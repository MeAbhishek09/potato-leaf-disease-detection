# Image Classification using CNN (PLD 3 Classes)

This repository contains a **Jupyter Notebook** for training and evaluating a **Convolutional Neural Network (CNN)** on an image classification dataset with **3 classes**. The project demonstrates an end-to-end deep learning workflow including data loading, exploration, visualization, model building, training, and evaluation using **TensorFlow/Keras**.

---

## 📌 Project Overview

* **Task**: Multi-class image classification (3 classes)
* **Framework**: TensorFlow / Keras
* **Input Size**: 256 × 256 RGB images
* **Model Type**: Custom CNN
* **Loss Function**: Sparse Categorical Crossentropy
* **Optimizer**: Adam

The dataset used in this project is structured as directories of images (one folder per class) and is loaded using `image_dataset_from_directory`.

---

## 📂 Dataset Structure

The notebook expects the dataset to be organized as follows:

```
PLD_3_Classes_256/
│
├── Training/
│   ├── Class_1/
│   ├── Class_2/
│   └── Class_3/
│
└── Validation/
    ├── Class_1/
    ├── Class_2/
    └── Class_3/
```

Each class directory should contain the corresponding image samples.

---

## ⚙️ Configuration

Key parameters used in the notebook:

```python
IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 20
```

---

## 🧠 Model Architecture

The CNN model is built using Keras `Sequential` API and includes:

* Input layer
* Multiple convolutional layers
* Max pooling layers
* Fully connected (Dense) layers
* Softmax output layer for 3-class classification

The model is compiled with:

* **Optimizer**: Adam
* **Loss**: Sparse Categorical Crossentropy
* **Metric**: Accuracy

---

## 📊 Data Visualization

The following figure shows sample images from the dataset used for training and validation. This helps in visually understanding class characteristics, image quality, and variability before model training.

<p align="center">
 <img width="795" height="500" alt="image" src="https://github.com/user-attachments/assets/75992f0c-f6a3-451e-bf55-4e1797674d05" />
</p>

<p align="center">
  <img width="554" height="457" alt="image" src="https://github.com/user-attachments/assets/3d8d97ff-7284-442d-bf4b-1d838b7f7732" />

</p>

* Class distribution analysis using `Counter`
* Visualization of class imbalance using **Matplotlib** and **Seaborn**

These steps help in understanding the dataset before training.

---

## 🚀 Training

The model is trained using:

```python
model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds
)
```

Training and validation accuracy/loss are tracked over epochs.

---

## 📈 Results

* The proposed CNN model was evaluated on an independent test dataset to assess its generalization capability. Quantitative evaluation shows that the model achieved a test accuracy of 78.79% with a corresponding test loss of 0.68, indicating effective learning of discriminative features from the input images.
* Overall, these results demonstrate that the proposed CNN architecture, trained on 256×256 RGB images, provides reliable performance for three-class PLD classification. The achieved accuracy establishes a strong baseline and highlights the potential of deep learning–based approaches for automated plant disease detection. Further improvements can be achieved through data augmentation, deeper architectures, and transfer learning–based methods.n

---

## 🛠️ Requirements

Make sure you have the following installed:

* Python 3.8+
* TensorFlow
* NumPy
* OpenCV (`cv2`)
* Matplotlib
* Seaborn

Install dependencies using:

```bash
pip install tensorflow opencv-python matplotlib seaborn numpy
```

---

## ▶️ How to Run

1. Clone the repository
2. Place the dataset in the expected directory structure
3. Open the notebook:

```bash
jupyter notebook
```

4. Run all cells sequentially

---

## 📌 Notes

* Ensure sufficient system memory and preferably a GPU for faster training
* Dataset size and balance can significantly affect performance

---

## 👤 Author

**Abhishek**
B.Tech CSE | AI & Deep Learning Enthusiast

---

## 📄 License

This project is for educational and research purposes.
