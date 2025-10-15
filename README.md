Amazon ML Challenge: Product Price Prediction
This repository contains a comprehensive solution for the Amazon Machine Learning Challenge, aimed at predicting product prices from unstructured and noisy catalog data. The project employs a multi-modal approach, leveraging deep learning embeddings for text and images, and was executed using an innovative parallel processing strategy to overcome cloud environment limitations.

🏆 Final Result
The final model, built using the "Balanced Build" strategy and a robust 5-Fold Cross-Validation, achieved the following performance:

Metric

Score

Cross-Validated SMAPE

54.78%

🚀 The "Balanced Build" Strategy
A core challenge was processing 150,000 products with intensive deep learning models within Google Colab's resource limits. To solve this, a parallel processing workflow was designed across two separate accounts, effectively cutting the main data processing time in half from ~3 hours to ~1.5 hours.

Workflow Overview
1. 📂 Data Split

The entire dataset was divided equally into two halves.

2. ⚙️ Parallel Processing (Accounts 1 & 2)

Two Colab sessions worked simultaneously. Account 1 processed the first 50% of the data, while Account 2 handled the second 50%. Each account performed the heavy computational tasks of downloading images and generating text/image embeddings for its assigned data half.

3. 🔗 Merge & Train

Both accounts saved their processed results to a shared Google Drive folder. A final script then loaded these halves, merged them into a complete feature set, and trained the final LightGBM model on the full, combined data.

🛠️ Technical Deep Dive
The model's performance is built on three core pillars: smart feature engineering, deep learning embeddings, and a robust modeling strategy.

1. "Elite" Feature Engineering
Hand-crafted features were extracted from the text to give the model explicit, high-value clues about the price.

📦 Advanced IPQ Extractor: A robust regex function to identify the Item Pack Quantity (e.g., "pack of 12," "144 per case," "2000 pcs"). This was the most critical feature for handling bulk items.

⚖️ Normalized Weight: A function to find product weights (e.g., "50 lbs," "16 oz") and convert them to a standard unit (ounces) for consistent comparison.

🏷️ Brand & Keyword Detection: Features to flag the presence of major brand names or high-value keywords like "bulk" and "case."

2. Deep Learning Embeddings
Text Embeddings: The state-of-the-art all-mpnet-base-v2 sentence transformer was used to convert product descriptions into 768-dimension vectors, capturing the semantic meaning of the text.

Image Embeddings: An EfficientNet-B0 computer vision model, pre-trained on ImageNet, was used to generate 1280-dimension vectors that represent the visual characteristics of each product image.

3. Robust Modeling
Algorithm: A LightGBM Regressor was chosen for its exceptional speed, memory efficiency, and high accuracy.

Validation Strategy: A robust 5-Fold Cross-Validation was used to train the final model, ensuring a stable and generalizable result.

📊 Feature Contribution Analysis
While exact feature importance is complex, the following illustrates the relative impact of each feature group on the model's predictive power. Textual context proved to be the most significant driver of price prediction.

Feature Group

Illustrative Importance

Text Embeddings (Semantic Meaning)

65%

"Elite" Engineered Features (IPQ, Weight, etc.)

25%

Image Embeddings (Visual Features)

10%

💻 Tools and Technologies
Programming Language: Python

Core Libraries: Pandas, NumPy, Scikit-learn

Modeling: LightGBM

Deep Learning: PyTorch, Sentence-Transformers, TIMM (PyTorch Image Models)

Environment: Google Colab

📂 Repository Structure
This repository contains all the processed data and results from the "Balanced Build" approach.

.
└── ML_Challenge_Final_Run/
    ├── ML_Challenge_Data.zip       # The original raw dataset.
    ├── train_text_h1.npy           # Text embeddings for the first half of the training data.
    ├── train_text_h2.npy           # Text embeddings for the second half.
    ├── train_image_h1.npy          # Image embeddings for the first half.
    ├── train_image_h2.npy          # Image embeddings for the second half.
    ├── test_text_h1.npy            # ...and so on for the test data.
    ├── test_text_h2.npy
    ├── test_image_h1.npy
    ├── test_image_h2.npy
    └── README.md                   # This file.

⚙️ How to Reproduce
To reproduce the final result, you can run the final "Merge & Train" script. This script assumes all eight .npy embedding files and the original ML_Challenge_Data.zip are present.

Environment: A Google Colab notebook with a T4 GPU is recommended.

Dependencies: Ensure lightgbm, pandas, numpy, and scikit-learn are installed.

Execute Script: The final script will load and merge the data halves, create the engineered features, and train the 5-fold LightGBM model to produce the submission.csv file.
