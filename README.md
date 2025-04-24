# Cybersecurity Intrusion Detection

## Project Description
This repository implements supervised machine learning models for intrusion detection using network traffic and user behavior features. It covers data preprocessing, model training (including ensemble & stacking), evaluation, and visualization.

## Dataset
Download the dataset from Kaggle:
https://www.kaggle.com/datasets/username/cybersecurity-intrusion-detection-dataset

Place the `cybersecurity_intrusion_data.csv` file into the `dataset/` folder.

## Repository Structure
```
intrusion_detection_project/
├── dataset/
│   └── README.md
├── data_processed/
├── results/
├── src/
│   ├── preprocess_and_train.py
│   ├── ensemble_and_stack.py
│   └── visualize.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
# Preprocess data and train base models
python src/preprocess_and_train.py

# Train ensemble & stacking models
python src/ensemble_and_stack.py

# Generate performance visualizations
python src/visualize.py
```
