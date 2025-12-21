import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report

# Global configs
RF_SEED = 2447
PCA_SEED = 2887
TREE_COUNTS = [10, 30, 50, 100, 200]

def preprocess(imgs):
    # Flatten 28x28 images to 1D vector
    N = imgs.shape[0]
    flat = imgs.reshape(N, -1)
    # Scale to 0-1
    return flat.astype('float32') / 255.0

def train_rf(X_train, y_train, X_val, y_val, X_test, y_test, name):
    print(f"\nTraining {name}")
    
    best_acc = 0
    best_rf = None
    history = []
    
    # Grid search for n_estimators
    for n in TREE_COUNTS:
        rf = RandomForestClassifier(n_estimators=n, random_state=RF_SEED)
        rf.fit(X_train, y_train)
        
        # Check validation acc
        val_pred = rf.predict(X_val)
        acc = accuracy_score(y_val, val_pred)
        history.append(acc)
        
        print(f"Trees: {n}, Val_Acc: {acc:.4f}.")
        
        if acc > best_acc:
            best_acc = acc
            best_rf = rf
            
    print(f"Best Val_Acc: {best_acc:.4f}")
    
    # test
    t0 = time.time()
    test_pred = best_rf.predict(X_test)
    t1 = time.time()
    
    final_acc = accuracy_score(y_test, test_pred)
    print(f"Test Acc: {final_acc:.4f}")
    print(f"Inference Time: {t1 - t0:.4f}s")
    
    # Detailed report
    print(classification_report(y_test, test_pred))
    
    return final_acc, history, best_rf

def run_model_A(train_imgs, train_lbls, val_imgs, val_lbls, test_imgs, test_lbls):
    print("Start Model A (RF vs PCA)")
    
    # 1. Prepare data (Flattened)
    train_flat = preprocess(train_imgs)
    val_flat   = preprocess(val_imgs)
    test_flat  = preprocess(test_imgs)
    
    # flatten labels
    y_train = train_lbls.ravel()
    y_val   = val_lbls.ravel()
    y_test  = test_lbls.ravel()
    
    # Experiment 1: Raw Pixels
    acc_raw, hist_raw, model_raw = train_rf(
        train_flat, y_train, 
        val_flat, y_val, 
        test_flat, y_test, 
        "Baseline (Raw)"
    )
    
    # Experiment 2: PCA
    # reduce dim from 784 -> 50
    pca = PCA(n_components=50, random_state=PCA_SEED)
    
    train_pca = pca.fit_transform(train_flat)
    val_pca   = pca.transform(val_flat)
    test_pca  = pca.transform(test_flat)
    
    print(f"PCA Reduction: {train_flat.shape} to {train_pca.shape}")
    
    acc_pca, hist_pca, model_pca = train_rf(
        train_pca, y_train, 
        val_pca, y_val, 
        test_pca, y_test, 
        "PCA Features"
    )
    
    # Summary
    print("\nModel A Results: ")
    print(f"Raw: {acc_raw:.4f}")
    print(f"PCA: {acc_pca:.4f}")
    
    # Get probs for ROC later
    prob_raw = model_raw.predict_proba(test_flat)[:, 1]
    prob_pca = model_pca.predict_proba(test_pca)[:, 1]
    
    # Pack everything for plotting
    results = {
        'trees': TREE_COUNTS,
        'raw_val_accs': hist_raw,
        'pca_val_accs': hist_pca,
        'roc_data': {
            'y_test': y_test,
            'y_prob_raw': prob_raw,
            'y_prob_pca': prob_pca,
            'acc_raw': acc_raw * 100,
            'acc_pca': acc_pca * 100
        }
    }
    
    return results