import os
import numpy as np
import medmnist
from medmnist import INFO

from Code.A.model import run_model_A
from Code.B.model import run_model_B
from Code.visualization import plot_all_results

def get_data():
    # Setup path
    data_flag = 'breastmnist'
    root = 'Datasets'
    
    # Check/Download using medmnist api
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    
    # Download if missing
    DataClass(split='train', download=True, root=root)
    DataClass(split='val', download=True, root=root)
    DataClass(split='test', download=True, root=root)
    
    # Load .npz file
    npz_path = os.path.join(root, f"{data_flag}.npz")
    if os.path.exists(npz_path):
        print(f"Data found at: {npz_path}")
        return np.load(npz_path)
    else:
        raise FileNotFoundError("NPZ file not found, download failed.")

def main():
    print("\n AMLS Assignment 25-26 | SN: 25072441 ")
    
    # 1. Load Data
    print("\nLoading BreastMNIST")
    data = get_data()

    # Unpack
    tr_x = data['train_images']
    tr_y = data['train_labels']
    val_x = data['val_images']
    val_y = data['val_labels']
    te_x = data['test_images']
    te_y = data['test_labels']
    
    print(f"Train: {tr_x.shape} | Val: {val_x.shape} | Test: {te_x.shape}")

    # 2. Run Model A (RF)
    print("\nModel A: Random Forest vs PCA")
    res_a = run_model_A(tr_x, tr_y, val_x, val_y, te_x, te_y)
    
    # 3. Run Model B (CNN)
    print("\nModel B: CNN Experiments")
    res_b = run_model_B(tr_x, tr_y, val_x, val_y, te_x, te_y)
    
    # 4. Visualization
    print("\nPlotting Results")
    # Generate charts and save to /Plots folder
    plot_all_results(data, res_a, res_b)
    
    print("\nAll Tasks Finished Successfully, yeah!")

if __name__ == '__main__':
    main()