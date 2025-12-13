import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

IMG_DIR = "Report_Images"
os.makedirs(IMG_DIR, exist_ok=True)


def plot_data_samples(data):
    print("Generating Figure: Data Samples")
    images = data['train_images']
    labels = data['train_labels']
    benign_idxs = np.where(labels == 0)[0][:5]
    malignant_idxs = np.where(labels == 1)[0][:5]
    
    plt.figure(figsize=(10, 4))
    for i, idx in enumerate(benign_idxs):
        plt.subplot(2, 5, i+1); plt.imshow(images[idx], cmap='gray'); plt.title("Benign"); plt.axis('off')
    for i, idx in enumerate(malignant_idxs):
        plt.subplot(2, 5, i+6); plt.imshow(images[idx], cmap='gray'); plt.title("Malignant"); plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/01_Data_Samples.png", dpi=300)
    plt.close()

# Various comparison diagrams of Model A
def plot_model_a_results(results_a):
    print("Generating Figure: Model A Analysis")
    trees = results_a['trees']
    
    # Model A Curve Diagram
    plt.figure(figsize=(10, 6))
    plt.plot(trees, results_a['raw_val_accs'], marker='o', label='Raw Pixels', color='#2c3e50', lw=2)
    plt.plot(trees, results_a['pca_val_accs'], marker='s', label='PCA Features', color='#e74c3c', linestyle='--', lw=2)
    plt.xlabel('Number of Trees'); plt.ylabel('Validation Accuracy')
    plt.title('Model A: Sensitivity Analysis')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(f"{IMG_DIR}/ModelA_Sensitivity.png", dpi=300)
    plt.close()
    
    # Model A ROC comparison chart
    roc_data = results_a['roc_data']
    y_test = roc_data['y_test']
    plt.figure(figsize=(8, 6))
    fpr1, tpr1, _ = roc_curve(y_test, roc_data['y_prob_raw'])
    fpr2, tpr2, _ = roc_curve(y_test, roc_data['y_prob_pca'])
    plt.plot(fpr1, tpr1, label=f'Raw (AUC={auc(fpr1, tpr1):.3f})')
    plt.plot(fpr2, tpr2, label=f'PCA (AUC={auc(fpr2, tpr2):.3f})', linestyle='--')
    plt.plot([0,1],[0,1],'k--',alpha=0.3)
    plt.title('Model A: ROC Comparison'); plt.legend(loc='lower right')
    plt.savefig(f"{IMG_DIR}/ModelA_ROC.png", dpi=300)
    plt.close()

# Various comparison diagrams of Model B
def plot_model_b_results(results_b):
    print("Generating Figures: Model B Analysis")
    names = list(results_b.keys())
    
    # Model B Curve Diagram
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, name in enumerate(names):
        ax1 = axes[i]
        hist = results_b[name]['history']
        epochs = range(1, len(hist['train_loss'])+1)
        ax1.plot(epochs, hist['train_loss'], 'r:', label='Train Loss')
        ax1.plot(epochs, hist['val_loss'], 'r-', label='Val Loss')
        ax1.set_ylabel('Loss', color='r')
        ax2 = ax1.twinx()
        ax2.plot(epochs, hist['train_acc'], 'b:', label='Train Acc')
        ax2.plot(epochs, hist['val_acc'], 'b-', label='Val Acc')
        ax2.set_ylabel('Acc (%)', color='b'); ax2.set_ylim(40, 100)
        ax1.set_title(f"{name} ({results_b[name]['acc']:.2f}%)")
    axes[5].axis('off'); axes[5].text(0.5,0.5,"Training Dynamics",ha='center')
    plt.tight_layout(); plt.savefig(f"{IMG_DIR}/ModelB_Dynamics.png", dpi=300); plt.close()
    
    # Model B ROC comparison chart
    plt.figure(figsize=(10, 8))
    for name, res in results_b.items():
        y_scores = np.array(res['y_probs'])[:, 1]
        fpr, tpr, _ = roc_curve(res['y_true'], y_scores)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})", 
                 color=res['color'], linestyle=res['ls'], lw=2)
    plt.plot([0,1],[0,1],'k--',alpha=0.2); plt.legend(loc='lower right')
    plt.title('Model B: Comparative ROC'); plt.savefig(f"{IMG_DIR}/ModelB_ROC.png", dpi=300); plt.close()

    # Model B: Confusion Matrix
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, name in enumerate(names):
        cm = confusion_matrix(results_b[name]['y_true'], results_b[name]['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
        axes[i].set_title(f"{name}")
    axes[5].axis('off')
    plt.tight_layout(); plt.savefig(f"{IMG_DIR}/ModelB_CM.png", dpi=300); plt.close()

# Comparison of the two models
def plot_grand_summary(results_a, results_b):
    print("Generating Figure: Grand Summary")
    acc_raw = results_a['roc_data']['acc_raw']
    acc_pca = results_a['roc_data']['acc_pca']
    
    names_b = list(results_b.keys())
    accs_b = [results_b[n]['acc'] for n in names_b]
    
    all_names = ['Model A\nRaw', 'Model A\nPCA'] + [f'Model B\n{n}' for n in names_b]
    all_accs = [acc_raw, acc_pca] + accs_b
    colors = ['#34495e', '#7f8c8d'] + [results_b[n]['color'] for n in names_b]
    
    plt.figure(figsize=(12, 7))
    bars = plt.bar(all_names, all_accs, color=colors, edgecolor='k', alpha=0.9)
    plt.ylim(60, 95); plt.ylabel('Test Accuracy (%)')
    plt.title('Overall Performance Comparison')
    for bar in bars:
        plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f"{bar.get_height():.2f}%", ha='center')
    plt.tight_layout(); plt.savefig(f"{IMG_DIR}/Grand_Summary.png", dpi=300); plt.close()

def plot_all_results(data, results_a, results_b):
    print("\n[Step 4] Visualizing Results...")
    plot_data_samples(data)
    plot_model_a_results(results_a)
    plot_model_b_results(results_b)
    plot_grand_summary(results_a, results_b)
    print(f"All images saved to {IMG_DIR}/")