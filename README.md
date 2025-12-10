# Pathology Image Classification: Random Forest vs. MLP vs. CNN

## 1. Project Overview
Automated classification of biomedical images is crucial for assisting diagnosis and reducing the workload of healthcare professionals. This project focuses on classifying 9 types of body tissues (e.g., adipose tissue, mucus, cancer-associated stroma) using the **PathMNIST** dataset (a subset of MedMNIST).

**Objective:** To implement, tune, and compare the performance of three distinct machine learning algorithms:
1.  **Random Forest (RF):** Representing traditional ensemble learning.
2.  **Multilayer Perceptron (MLP):** Representing fully connected neural networks.
3.  **Convolutional Neural Network (CNN):** Representing deep learning specifically designed for image data.

**Key Results:** The **CNN** model outperformed others with an accuracy of **83.24%**, demonstrating the superiority of spatial feature extraction in medical imaging tasks.

## 2. Dataset & Setup
* **Source:** [PathMNIST (MedMNIST v2)](https://medmnist.com/)
* **Data Structure:** 28x28 RGB images.
* **Classes:** 9 tissue types (Label 0-8).
* **Size:** Training set (32,000 images), Test set (8,000 images).

**Note:** Due to file size limitations, the dataset is not hosted directly in this repository.
**How to run this project:**
1.  Download the **PathMNIST** dataset (part of MedMNIST v2) from the official source: [MedMNIST Website](https://medmnist.com/) or [Zenodo](https://zenodo.org/record/6496656).
2.  Create a folder named `Data` in the same directory as the notebook.
3.  Place the downloaded `.npy` files (`X_train.npy`, `y_train.npy`, etc.) into that folder.
4.  Run the notebook.

## 3. Methodology & Tech Stack
* **Python Libraries:** TensorFlow (Keras), Scikit-learn, Pandas, NumPy, Seaborn, Keras-Tuner.
* **Hyperparameter Tuning:**
    * **Grid Search** (5-fold CV) for Random Forest.
    * **Bayesian Optimization** for Neural Networks (MLP & CNN).

### Model Architectures:
* **Random Forest:** Tuned `n_estimators`, `max_depth`, and `class_weight`.
* **MLP:** 3 Hidden Layers (ReLu activation) with Dropout for regularization.
* **CNN:** 2 CONV-RELU-POOL blocks + Fully Connected Layer.

## 4. Performance Comparison
| Model | Test Accuracy | Training Time | Key Observation |
|-------|---------------|---------------|-----------------|
| **CNN** | **83.24%** | ~392s | Best performance; captures spatial patterns effectively. |
| **Random Forest** | 66.05% | **~48s** | Fastest training; robust baseline but struggles with complex image features. |
| **MLP** | 59.19% | ~97s | Lowest accuracy; fails to preserve spatial structures of images. |

## 5. Key Insights & Visualizations

### Confusion Matrix Analysis
The confusion matrices reveal that all models struggled with **Class 6 (Normal Colon Mucosa)**, often confusing it with Class 4 or 7. However, the CNN showed significantly better discrimination for Class 1 and Class 8.

*Figure: Confusion Matrix of the final CNN model.*

### Hyperparameter Tuning Analysis
* **Random Forest:** Increasing tree depth beyond 15 yielded diminishing returns, indicating potential overfitting.
* **CNN:** The Bayesian Optimization highlighted that a **Learning Rate of 0.001** provided the best balance between convergence speed and stability.

## 6. Conclusion & Future Work
While the **CNN** is the clear winner for accuracy, the **Random Forest** offers a viable alternative for scenarios requiring extremely low computational resources.

**Future Improvements:**
1.  **Data Augmentation:** To address the misclassification in Class 6, techniques like rotation, flipping, and zooming could be applied to enrich the training set.
2.  **Transfer Learning:** Implementing ResNet50 or VGG16 (pre-trained on ImageNet) could likely boost accuracy beyond 90%.
3.  **Feature Engineering:** For the Random Forest, extracting Histogram of Oriented Gradients (HOG) features might improve performance compared to using raw flattened pixels.

## 7. How to Run
1.  Clone this repository.
2.  Install dependencies: `pip install tensorflow scikit-learn pandas seaborn keras-tuner`
3.  Run the notebook `pathology_classification.ipynb`.
