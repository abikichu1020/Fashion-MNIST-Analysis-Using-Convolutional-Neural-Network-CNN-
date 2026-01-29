# Fashion MNIST Analysis Using Convolutional Neural Network (CNN)

## Overview
This project focuses on analyzing and classifying images from the **Fashion MNIST dataset** using a **Convolutional Neural Network (CNN)**. CNNs are especially well-suited for image data as they automatically learn spatial features such as edges, textures, and shapes.

All implementation and analysis are contained in the Jupyter Notebook:

**`fashion mnist cnn.ipynb`**

---

## Dataset: Fashion MNIST
Fashion MNIST is a popular benchmark dataset for evaluating image classification models.

**Dataset Details:**
- 70,000 grayscale images
- Image size: 28 × 28 pixels
- 10 classes
- 60,000 training images
- 10,000 testing images

**Class Labels:**
- T-shirt / Top  
- Trouser  
- Pullover  
- Dress  
- Coat  
- Sandal  
- Shirt  
- Sneaker  
- Bag  
- Ankle Boot  

---

## Objective
The main objectives of this project are to:
- Apply Convolutional Neural Networks for image classification
- Learn spatial feature extraction using convolution and pooling layers
- Compare CNN performance with traditional deep learning (ANN) models
- Achieve higher accuracy through hierarchical feature learning

---

## Technologies Used
- Python  
- Jupyter Notebook  
- NumPy  
- Matplotlib  
- TensorFlow / Keras (or PyTorch, depending on implementation)

---

## Project Workflow
1. **Library Imports**
   - Import required libraries for deep learning and visualization.

2. **Dataset Loading**
   - Load the Fashion MNIST dataset using built-in dataset utilities.

3. **Data Preprocessing**
   - Normalize pixel values to range [0, 1]
   - Reshape images to include channel dimension (28 × 28 × 1)
   - Encode labels if required

4. **CNN Model Architecture**
   - Convolutional layers for feature extraction
   - Activation function: ReLU
   - MaxPooling layers for spatial downsampling
   - Flatten layer to convert 2D features into 1D
   - Fully connected (Dense) layers
   - Output layer with Softmax activation (10 classes)

5. **Model Compilation**
   - Optimizer: Adam
   - Loss function: Sparse Categorical Crossentropy
   - Evaluation metric: Accuracy

6. **Model Training**
   - Train the CNN using training data
   - Validate performance using test data
   - Track loss and accuracy across epochs

7. **Evaluation and Analysis**
   - Evaluate final model on test dataset
   - Plot training vs validation accuracy and loss
   - Analyze misclassified samples

8. **Prediction**
   - Predict labels for unseen images
   - Visualize predictions alongside true labels

---

## Results
- CNN significantly improves classification accuracy compared to basic neural networks
- Convolutional layers effectively learn visual features
- Reduced overfitting with pooling and deeper representations

---

## How to Run the Notebook
1. Clone or download the project files
2. Install the required dependencies
3. Launch Jupyter Notebook
4. Run `fashion mnist cnn.ipynb`

```bash
pip install numpy matplotlib tensorflow
jupyter notebook
