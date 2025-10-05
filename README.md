# 🔬 Pneumonia Detection using Deep Learning (DenseNet121)

### Overview

This project implements a Convolutional Neural Network (CNN) model using **Transfer Learning** with the **DenseNet121** architecture to automatically detect pneumonia from chest X-ray images. The goal is to provide an efficient and accurate diagnostic aid, particularly valuable in resource-constrained medical environments.

The model is trained on a structured dataset containing images classified as **NORMAL** and **PNEUMONIA**. Techniques like **data augmentation** and **class weighting** were employed to handle the significant class imbalance inherent in medical imaging datasets and improve the model's generalization capabilities.

---

### 🚀 Key Features

* **Architecture:** Fine-tuned **DenseNet121** (pre-trained on ImageNet).
* **Transfer Learning:** Two-stage training process (feature extraction, followed by fine-tuning of the final layers).
* **Data Handling:** Extensive **data augmentation** (rotation, shifts, flips, zoom) to prevent overfitting.
* **Imbalance Handling:** Utilizes **class weighting** to mitigate the effects of class imbalance.
* **Evaluation:** Comprehensive performance analysis using **Classification Report**, **Confusion Matrix**, and **ROC-AUC** curve.

---

### 🛠️ Technology Stack

* **Language:** Python
* **Framework:** TensorFlow / Keras
* **Libraries:** NumPy, Matplotlib, Seaborn, Scikit-learn
* **Model:** `DenseNet121`

---

### 📂 Dataset

The project utilizes a chest X-ray image dataset from Kaggle, structured into `train`, `val`, and `test` directories.

* **Source:** [Insert the actual Kaggle Dataset Link Here]
* **Structure:**
    ```
    /Kaggle_Date_Pneu
    ├── /train
    │   ├── /NORMAL
    │   └── /PNEUMONIA
    ├── /val
    │   ├── /NORMAL
    │   └── /PNEUMONIA
    └── /test
        ├── /NORMAL
        └── /PNEUMONIA
    ```
* ***Note:** Ensure the data path in the script is updated to your local directory.*

---

### 📊 Model Performance

After fine-tuning, the model achieved the following strong performance metrics on the independent **Test Set**:

| Metric | Result | Interpretation |
| :--- | :--- | :--- |
| **Test Accuracy** | ~88.46% | High overall prediction correctness. |
| **AUC (Area Under ROC Curve)** | > 0.95 | Excellent discriminative ability (strong separation between classes). |
| **Classification Scores** | High (Precision, Recall, F1-score) | Indicating high performance, especially for detecting the PNEUMONIA class. |
| **Confusion Matrix** | Minimal False Positives/Negatives | Model accurately identifies most cases with low error rates. |

***


