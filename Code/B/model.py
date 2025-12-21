import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import time
from torch.utils.data import TensorDataset, DataLoader, Dataset, Subset
from sklearn.metrics import accuracy_score, classification_report
from torchvision import transforms

# Global configs
SEED = 25072441
BATCH_SIZE = 32
LR = 0.001
MAX_EPOCHS = 50

# Set device using cppu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

# Custom Transform: Add noise to images
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return torch.clamp(tensor + noise, 0., 1.)

# Dataset processing
class BreastDataset(Dataset):
    def __init__(self, imgs, lbls, transform=None):
        if len(imgs.shape) == 3:
            imgs = np.expand_dims(imgs, axis=1)
            
        self.x = torch.tensor(imgs, dtype=torch.float32) / 255.0
        self.y = torch.tensor(lbls, dtype=torch.long).squeeze() 
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = self.x[idx]
        lbl = self.y[idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img, lbl

# CNN Architecture
class BreastNet(nn.Module):
    def __init__(self, num_filters=32):
        super(BreastNet, self).__init__()

        # Block 1: 1 -> 32
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        
        # Block 2: 32 -> 64
        self.conv2 = nn.Conv2d(num_filters, num_filters * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters * 2)

        # MaxPool
        self.pool = nn.MaxPool2d(2, 2)
        
        # Image size: 28 -> 14 -> 7
        flat_size = (num_filters * 2) * 7 * 7
        self.fc1 = nn.Linear(flat_size, 128)
        self.fc2 = nn.Linear(128, 2)

        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        # Conv 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        # Conv 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # 全连接
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        
        return x

def train_one_epoch(net, loader, crit, opt):
    net.train()
    loss_sum = 0
    correct = 0
    total = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        opt.zero_grad()
        out = net(x)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        
        loss_sum += loss.item()
        _, pred = torch.max(out, 1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        
    return loss_sum / len(loader), 100 * correct / total

def eval_model(net, loader, crit, detailed=False):
    net.eval()
    loss_sum = 0
    correct = 0
    total = 0
    
    # store for ROC
    preds, labels, probs = [], [], []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = net(x)
            loss = crit(out, y)
            
            loss_sum += loss.item()
            _, p = torch.max(out, 1)
            prob = F.softmax(out, dim=1)
            
            correct += (p == y).sum().item()
            total += y.size(0)
            
            if detailed:
                preds.extend(p.cpu().numpy())
                labels.extend(y.cpu().numpy())
                probs.extend(prob.cpu().numpy())
                
    acc = 100 * correct / total
    avg_loss = loss_sum / len(loader)
    
    if detailed:
        return avg_loss, acc, labels, preds, probs
    return avg_loss, acc

def run_cnn_experiment(tr_img, tr_lbl, val_img, val_lbl, te_img, te_lbl, 
                       mode='baseline', filters=32, data_pct=1.0):
    
    print(f"\n Mode={mode} | Filters={filters} | Data={data_pct*100}%")
    
    # 1. Transforms
    tr_trans = None
    if mode == 'geometric':
        tr_trans = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
        ])
    elif mode == 'noise':
        tr_trans = transforms.Compose([
            AddGaussianNoise(0., 0.05),
        ])
        
    # 2. Datasets
    full_ds = BreastDataset(tr_img, tr_lbl, transform=tr_trans)
    val_ds  = BreastDataset(val_img, val_lbl)
    test_ds = BreastDataset(te_img, te_lbl)
    
    # Data Budget
    if data_pct < 1.0:
        n_sub = int(len(full_ds) * data_pct)
        # pick first N samples
        idx = list(range(n_sub))
        train_ds = Subset(full_ds, idx)
        print(f"Subset Training: {n_sub} samples")
    else:
        train_ds = full_ds
    
    # Load
    tr_load = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_load = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    te_load  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    net = BreastNet(num_filters=filters).to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(net.parameters(), lr=LR)
    sched = optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.1)
    
    best_acc = 0
    best_wts = None
    hist = {'tr_loss':[], 'tr_acc':[], 'val_loss':[], 'val_acc':[]}
    
    start_t = time.time()
    for ep in range(MAX_EPOCHS):
        tl, ta = train_one_epoch(net, tr_load, crit, opt)
        vl, va = eval_model(net, val_load, crit)
        sched.step()
        
        hist['tr_loss'].append(tl)
        hist['tr_acc'].append(ta)
        hist['val_loss'].append(vl)
        hist['val_acc'].append(va)
        
        if va > best_acc:
            best_acc = va
            best_wts = net.state_dict()
            
        if (ep+1) % 10 == 0:
            print(f"Ep {ep+1}: Tr Acc={ta:.1f}%, Val Acc={va:.1f}%")
            
    print(f"Training Done in {time.time()-start_t:.1f}s. Best Val={best_acc:.2f}%")
    
    # Test
    net.load_state_dict(best_wts)
    _, te_acc, y_true, y_pred, y_prob = eval_model(net, te_load, crit, detailed=True)
    
    print(f"Test Acc: {te_acc:.2f}%")
    print(classification_report(y_true, y_pred, digits=4))
    
    return te_acc, hist, (y_true, y_pred, y_prob)

def run_model_B(tr_x, tr_y, val_x, val_y, te_x, te_y):
    print("Start Model B (CNN Analysis)")
    set_seed(SEED)
    
    # Configs: (Name, Mode, Filters, Data%, Color)
    exps = [
        ('Baseline',     'baseline',  32, 1.0, 'gray'),
        ('Geometric',    'geometric', 32, 1.0, 'blue'),
        ('Noise',        'noise',     32, 1.0, 'red'),
        ('Low Capacity', 'baseline',  16, 1.0, 'green'),
        ('Half Budget',  'geometric', 32, 0.5, 'orange')
    ]
    
    all_res = {}
    
    for name, mode, filt, pct, col in exps:
        acc, h, preds = run_cnn_experiment(
            tr_x, tr_y, val_x, val_y, te_x, te_y,
            mode=mode, filters=filt, data_pct=pct
        )
        all_res[name] = {
            'acc': acc, 
            'history': h, 
            'preds': preds, 
            'color': col
        }
        
    return all_res