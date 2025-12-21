import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Save path
IMG_DIR = "Plots"
if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR)

def plot_samples(data):
    print("Plotting Data Samples")
    x = data['train_images']
    y = data['train_labels']
    
    # Pick 5 random samples per class
    ben_idx = np.where(y == 0)[0][:5]
    mal_idx = np.where(y == 1)[0][:5]
    
    plt.figure(figsize=(10, 4))
    
    # Benign
    for i, idx in enumerate(ben_idx):
        plt.subplot(2, 5, i+1)
        plt.imshow(x[idx], cmap='gray')
        plt.title("Benign")
        plt.axis('off')
        
    # Malignant
    for i, idx in enumerate(mal_idx):
        plt.subplot(2, 5, i+6)
        plt.imshow(x[idx], cmap='gray')
        plt.title("Malignant")
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/01_Samples.png", dpi=150)
    plt.close()

def plot_model_a(res):
    print("Plotting Model A (RF)")
    
    # plot 1. Sensitivity Curve
    trees = res['trees']
    
    plt.figure(figsize=(8, 5))
    plt.plot(trees, res['raw_val_accs'], 'o-', label='Raw Pixels', color='#2c3e50')
    plt.plot(trees, res['pca_val_accs'], 's--', label='PCA (50)', color='#e74c3c')
    
    plt.xlabel('Num Trees')
    plt.ylabel('Val Acc')
    plt.title('RF Sensitivity Analysis')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"{IMG_DIR}/A_Sensitivity.png")
    plt.close()
    
    # plot 2. modelA ROC
    roc = res['roc_data']
    y_test = roc['y_test']
    
    plt.figure(figsize=(6, 5))
    
    # Raw
    fpr1, tpr1, _ = roc_curve(y_test, roc['y_prob_raw'])
    auc1 = auc(fpr1, tpr1)
    plt.plot(fpr1, tpr1, label=f'Raw (AUC={auc1:.3f})')
    
    # PCA
    fpr2, tpr2, _ = roc_curve(y_test, roc['y_prob_pca'])
    auc2 = auc(fpr2, tpr2)
    plt.plot(fpr2, tpr2, '--', label=f'PCA (AUC={auc2:.3f})')
    
    plt.plot([0,1],[0,1], 'k:', alpha=0.3)
    plt.title('Model A ROC')
    plt.legend(loc='lower right')
    plt.savefig(f"{IMG_DIR}/A_ROC.png")
    plt.close()

def plot_model_b(res):
    print("Plotting Model B (CNN)")
    names = list(res.keys())
    
    # 1. Training Dynamics
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for i, name in enumerate(names):
        ax = axes[i]
        h = res[name]['history']
        
        epochs = range(1, len(h['tr_loss'])+1)
        
        # Plot Loss (Red)
        ax.plot(epochs, h['tr_loss'], 'r:', alpha=0.6, label='Tr Loss')
        ax.plot(epochs, h['val_loss'], 'r-', label='Val Loss')
        ax.set_ylabel('Loss', color='r')
        ax.tick_params(axis='y', labelcolor='r')
        
        # Plot Acc (Blue) on twin axis
        ax2 = ax.twinx()
        ax2.plot(epochs, h['tr_acc'], 'b:', alpha=0.6, label='Tr Acc')
        ax2.plot(epochs, h['val_acc'], 'b-', label='Val Acc')
        ax2.set_ylabel('Acc %', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        ax2.set_ylim(40, 100)
        
        ax.set_title(f"{name} ({res[name]['acc']:.1f}%)")
        
    if len(names) < 6:
        for j in range(len(names), 6):
            axes[j].axis('off')
            
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/B_Dynamics.png")
    plt.close()
    
    # 2. ROC Comparison of MODEL 吧
    plt.figure(figsize=(8, 6))
    for name in names:

        # Unpack the tuple (y_true, y_pred, y_prob) from 'preds' key
        y_true, y_pred, probs_raw = res[name]['preds']
        
        probs = np.array(probs_raw)
        
        # Safety check for dimensions
        if probs.ndim == 2:
            scores = probs[:, 1]
        else:
            scores = probs
            
        fpr, tpr, _ = roc_curve(y_true, scores)
        
        score = auc(fpr, tpr)
        
        ls = '-'
        if 'Baseline' in name: ls = '--'
        if 'Noise' in name: ls = '-.'
        
        plt.plot(fpr, tpr, label=f"{name} ({score:.3f})", 
                 color=res[name]['color'], linestyle=ls, lw=1.5)
                 
    plt.plot([0,1],[0,1], 'k:', alpha=0.3)
    plt.legend()
    plt.title("CNN ROC Comparison")
    plt.savefig(f"{IMG_DIR}/B_ROC.png")
    plt.close()

def plot_summary(res_a, res_b):
    print("Plotting Grand Summary")
    
    # Get A stats
    acc_raw = res_a['roc_data']['acc_raw']
    acc_pca = res_a['roc_data']['acc_pca']
    
    # Get B stats
    b_names = list(res_b.keys())
    b_accs = [res_b[n]['acc'] for n in b_names]
    b_cols = [res_b[n]['color'] for n in b_names]
    
    # Combine
    labels = ['RF\nRaw', 'RF\nPCA'] + [f'CNN\n{n}' for n in b_names]
    values = [acc_raw, acc_pca] + b_accs
    colors = ['gray', 'lightgray'] + b_cols
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=colors, edgecolor='black', alpha=0.8)
    
    # Add text
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h + 1, 
                 f"{h:.1f}%", ha='center', va='bottom', fontsize=9)
                 
    plt.ylim(50, 100)
    plt.ylabel("Test Acc %")
    plt.title("Final Performance Summary")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/Grand_Summary.png")
    plt.close()

def plot_all_results(data, res_a, res_b):
    plot_samples(data)
    plot_model_a(res_a)
    plot_model_b(res_b)
    plot_summary(res_a, res_b)
    print(f"\nAll plots saved to /{IMG_DIR}")