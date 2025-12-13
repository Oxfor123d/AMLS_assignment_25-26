import os
import numpy as np
import sys
import medmnist
from medmnist import INFO


from Code.A.model import run_model_A
from Code.B.model import run_model_B
from Code.visualization import plot_all_results

def load_data():
    """
    Intelligent Data Loader:
    1. Verify presence of breastmnist.npz within the Datasets folder
    2. If absent, initiate automatic download
    3. Return loaded data dictionary
    """
    data_flag = 'breastmnist'
    output_folder = 'Datasets'
    os.makedirs(output_folder, exist_ok=True)
    
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    
    print(f"\n[Info] Checking/Downloading {data_flag}...")
    
    DataClass(split='train', download=True, root=output_folder)
    DataClass(split='val', download=True, root=output_folder)
    DataClass(split='test', download=True, root=output_folder)
    
    npz_file = f"{data_flag}.npz"
    data_path = os.path.join(output_folder, npz_file)
    
    if os.path.exists(data_path):
        print(f"[Success] Data file ready at: {data_path}")
        return np.load(data_path)
    else:
        raise FileNotFoundError(f"Failed to load data from {data_path}")

def main():
    print("AMLS Assignment 25-26: BreastMNIST Classification")
    print(f"Student Number: SN25072441") 
    
    # 1. load data
    print("\nStep 1: Loading Dataset...")
    data = load_data()

    train_images = data['train_images']
    train_labels = data['train_labels']
    val_images = data['val_images']
    val_labels = data['val_labels']
    test_images = data['test_images']
    test_labels = data['test_labels']
    
    print(f"Data Loaded. Train: {train_images.shape}, Validation: {val_images.shape}, Test: {test_images.shape}")

    # 2. Model A (Random Forest)
    print("\nStep 2: Running Model A (Classic Machine Learning)")
    results_A = run_model_A(train_images, train_labels, val_images, val_labels, test_images, test_labels)
    
    # 3. Model B (CNN)
    print("\nStep 3: Running Model B (Deep Learning)")
    results_B = run_model_B(train_images, train_labels, val_images, val_labels, test_images, test_labels)
    # 4. plot
    plot_all_results(data, results_A, results_B)
    
    print("\nAll Tasks Completed")

if __name__ == '__main__':
    main()