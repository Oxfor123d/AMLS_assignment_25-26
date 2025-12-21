# AMLS Assignment 2025-26

**Student Number:** 25072441  
**Project:** Breast Ultrasound Image Classification (BreastMNIST)

---

## Project Overview

This project implements a comparative study between **Classical Machine Learning** and **Deep Learning** for medical image classification. Using the **BreastMNIST** dataset , the goal is to classify ultrasound images into **Benign (0)** or **Malignant (1)** categories.

The study investigates:
1.  **Feature Engineering:** Comparing Raw Pixels vs. PCA Features using Random Forest.
2.  **Deep Learning Dynamics:** Analyzing how Data Augmentation, Model Capacity, and Training Budget affect CNN performance on small datasets.

---

## Project Structure

The project is organized as follows:


.
├── Code/
│   ├── A/
│   │   └── model.py         # Model A: Random Forest implementation & Grid Search
│   ├── B/
│   │   └── model.py         # Model B: Custom CNN architecture & Experiment pipeline
│   └── visualization.py     # Helper functions to generate plots/charts
├── Datasets/                # Folder for automatic dataset download (BreastMNIST.npz)
├── Plots/                   # All generated result graphs are saved here
├── main.py                  # Main entry point to run the entire project
├── requirements.txt         # List of dependencies
└── README.md                # Project documentation

---

## Environment Setup & InstallationTo

To reproduce the results exactly as reported, please follow these setup steps.

1. **Prerequisites**

Python 3.8 or higher. I use the Python 3.12.7
It is recommended to use a virtual environment venv to avoid conflicts.

Bash: 

python -m venv venv
.\venv\Scripts\activate

2. **Install Dependencies**

Navigate to the project root directory and install the required packages:

Bash: 
pip install -r requirements.txt

---

## How to Run

To run the entire pipeline (Data Loading -> Model A -> Model B -> Visualization), simply execute:

Bash:
python main.py


