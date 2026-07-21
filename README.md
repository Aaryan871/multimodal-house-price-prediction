# 🏠 Multimodal House Price Prediction using CNN and Tabular Data

> A Deep Learning-based Multimodal House Price Prediction system that combines **property images** and **tabular features** to accurately estimate real estate prices.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![Keras](https://img.shields.io/badge/Keras-CNN-red)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

---

# 📌 Project Overview

House prices are influenced by many factors. Traditional machine learning models usually rely only on structured numerical features such as:

- Bedrooms
- Bathrooms
- squt

However, the **visual appearance of a property** also plays a significant role in determining its value.

This project introduces a **Multimodal Deep Learning Model** that learns from:

- 🖼 Property Images (Custom CNN)
- 📊 Tabular Property Features

Both feature sets are fused together to predict the final house price.

---

## 📸 Application Preview

Add screenshots of your Streamlit application here.

<p align="center">
  <img src="House Price Prediction.gif" alt="house-price-prediction" width="900">
</p>

# 🚀 Features

- Custom CNN (No Pretrained Model)
- Image Feature Extraction
- Tabular Data Processing
- Feature Fusion
- Deep Learning Regression Model
- Price Prediction
- End-to-End Pipeline
- TensorFlow/Keras Implementation

---

# 🧠 Model Architecture

```
                 Property Images
                        │
                 Custom CNN Network
                        │
               Image Feature Vector
                        │
                        ├──────────────┐
                        │              │
                        │      Tabular Features
                        │              │
                        │      Dense Layers
                        │              │
                        └──── Feature Fusion ────► Dense Layers ───► House Price
```

---

# 📂 Dataset

The project uses a multimodal dataset containing:

## Image Data

- Interior Images
- Exterior Images
- Property Photos

## Tabular Data

- Bedrooms
- Bathrooms
- squt
- Price (Target)

---

# 🛠 Tech Stack

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn
- OpenCV

---

# 📁 Project Structure

```
multimodal-house-price-prediction/

│
├── dataset/
│   ├── images/
│   └── house_data.csv
│
├── notebook/
│   └── House_Price_Prediction.ipynb
│
├── models/
│   └── trained_model.keras
│
├── results/
│   ├── training_loss.png
│   ├── prediction_plot.png
│   └── evaluation.png
│
├── README.md
│
└── requirements.txt
```

---

# ⚙️ Workflow

```
Dataset
    │
    ▼
Load Images
    │
    ▼
Image Preprocessing
    │
    ▼
Custom CNN
    │
    ▼
Image Features
    │
    ▼
Load Tabular Features
    │
    ▼
Feature Fusion
    │
    ▼
Dense Neural Network
    │
    ▼
House Price Prediction
```

---

# 📈 Training Pipeline

1. Load Dataset
2. Preprocess Images
3. Normalize Pixel Values
4. Process Tabular Features
5. Train Custom CNN
6. Extract Image Features
7. Merge Features
8. Train Regression Model
9. Evaluate Performance
10. Predict House Price

---

# 📊 Evaluation Metrics

The model is evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

---

# 💻 Installation

Clone the repository

```bash
git clone https://github.com/Aaryan871/multimodal-house-price-prediction.git
```

Move into project directory

```bash
cd multimodal-house-price-prediction
```

---

# ▶️ Run the Notebook

```bash
jupyter notebook
```

Open:

```
House_Price_Prediction.ipynb
```

---

# 📷 Sample Pipeline

```
Property Image
        │
        ▼
Custom CNN
        │
        ▼
Image Features
        │
        ▼
Feature Fusion
        │
        ▼
Regression Network
        │
        ▼
Predicted House Price
```

---

# 🎯 Future Improvements

- Vision Transformer (ViT)
- Attention-based Feature Fusion
- Location Embeddings
- Web Deployment using Streamlit
- Hyperparameter Optimization

---

# 📌 Project Highlights

✅ Multimodal Deep Learning

✅ Custom CNN (No Transfer Learning)

✅ Image + Tabular Feature Fusion

✅ End-to-End House Price Prediction

✅ TensorFlow/Keras Implementation

---

# 👨‍💻 Author

**Aaryan Chauhan**

BCA Student | Data Science Enthusiast

Interested in:

- Machine Learning
- Deep Learning
- Computer Vision
- Multimodal AI

---

# ⭐ If you like this project

Please consider giving this repository a ⭐ on GitHub.

It motivates me to build more AI and Deep Learning projects.
