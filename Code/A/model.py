import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
import time

def flatten_and_normalize(images):
    """

    Basic preprocessing: flattening & normalisation
    
    """
    n_samples = images.shape[0]
    # Flatten: (N, 28, 28) -> (N, 784)
    flattened = images.reshape(n_samples, -1)
    # Normalize
    return flattened.astype('float32') / 255.0

def run_rf_experiment(X_train, y_train, X_val, y_val, X_test, y_test, exp_name):
    """

    General function for running a single random forest experiment

    """
    print(f"\nprogressing: {exp_name}")
    
    # Define the parameters to be tested
    estimators_list = [10, 30, 50, 100, 200]
    best_acc = 0
    best_model = None
    val_acc_history = []

    # Tuning parameters on the validation set
    for n in estimators_list:
        clf = RandomForestClassifier(n_estimators=n, random_state=2447)
        clf.fit(X_train, y_train)
        
        val_pred = clf.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        val_acc_history.append(val_acc)
        print(f"   [Tuning] Trees={n}, Val Acc={val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = clf
            
    print(f"Best validation set accuracy: {best_acc:.4f}")
    
    # Evaluate the best model on the test set
    start_time = time.time()
    test_pred = best_model.predict(X_test)
    end_time = time.time()
    
    test_acc = accuracy_score(y_test, test_pred)
    inference_time = end_time - start_time
    
    print(f"Test set accuracy: {test_acc:.4f}")
    print(f"Time-consuming: {inference_time:.4f} s")
    
    print(classification_report(y_test, test_pred))
    
    return test_acc, val_acc_history, best_model

def run_model_A(train_images, train_labels, val_images, val_labels, test_images, test_labels):
    print("Initiating Model A: Feature Engineering Comparative Testing")
    
    # 1. basic data processing
    X_train_raw = flatten_and_normalize(train_images)
    X_val_raw   = flatten_and_normalize(val_images)
    X_test_raw  = flatten_and_normalize(test_images)
    
    # Label flatten
    y_train = train_labels.ravel()
    y_val   = val_labels.ravel()
    y_test  = test_labels.ravel()
    
    # test 1: Raw Pixels
    acc_raw, val_accs_raw, best_model_raw = run_rf_experiment(
        X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test, 
        "Baseline (Raw Pixels)"
    )
    

    # test 2: PCA Feature extraction
    pca = PCA(n_components=50, random_state=2887)
    
    # Fit on Train
    X_train_pca = pca.fit_transform(X_train_raw)
    # Transform Val & Test
    X_val_pca   = pca.transform(X_val_raw)
    X_test_pca  = pca.transform(X_test_raw)
    
    print(f"Data shape variation: {X_train_raw.shape} to {X_train_pca.shape}")
    
    acc_pca, val_accs_pca, best_model_pca = run_rf_experiment(
        X_train_pca, y_train, X_val_pca, y_val, X_test_pca, y_test, 
        "PCA Features (50 components)"
    )
    

    print("\nModel A final comparison results:")
    print(f"Raw Pixels Accuracy: {acc_raw:.4f}")
    print(f"PCA Features Accuracy: {acc_pca:.4f}")
    
    y_prob_raw = best_model_raw.predict_proba(X_test_raw)[:, 1]
    y_prob_pca = best_model_pca.predict_proba(X_test_pca)[:, 1]
    
    results = {
        'trees': [10, 30, 50, 100, 200],
        'raw_val_accs': val_accs_raw,
        'pca_val_accs': val_accs_pca,
        'roc_data': {
            'y_test': y_test,
            'y_prob_raw': y_prob_raw,
            'y_prob_pca': y_prob_pca,
            'acc_raw': acc_raw * 100,
            'acc_pca': acc_pca * 100
        }
    }
    
    return results

