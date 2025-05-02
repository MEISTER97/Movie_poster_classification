# 🎬 Movie Poster Genre Classification (Multi-label) with ResNet-50

This project classifies movie posters into multiple genres using deep learning. Leveraging a pretrained ResNet-50 model, the system identifies visual patterns in posters and predicts applicable genres in a multi-label setup.

---

## 📁 Dataset

The dataset used is from [Kaggle: Movie Genre Classifier Dataset](https://www.kaggle.com/datasets/raman77768/movie-classifier), which contains movie posters and their associated genres.

Each poster may belong to multiple genres (e.g., Action, Drama, Comedy).
---

## 🧠 Model

- **Backbone**: ResNet-50 pretrained on ImageNet
- **Modifications**:
  - Replaced the final classification layer with:
    ```python
    nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    ```
- **Loss Function**: `BCEWithLogitsLoss` (for multi-label classification)
- **Alternative**: Option to use a custom-built CNN for experimentation

---

## ⚙️ Features

-  Multi-label classification for 25+ genres  
-  Fine-tuning with transfer learning  
-  Grad-CAM visualization for model explainability  
-  Prediction support for single images or entire folders  
-  Confusion matrix and performance plots

---


