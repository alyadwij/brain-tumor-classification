# brain-tumor-classification

A deep learning-based system for classifying brain MRI images into four classes, named glioma tumor, meningioma tumor, pituitary tumor, and no tumor brain.

This project was developed as a final project for my applied bachelor degree in Computer Engineering program.

## project overview
This project is used to classify brain tumor from MRI images based on image patterns. This project implements and compares a custom CNN with several transfer learning models for multi-class brain MRI classification.

The system also includes a graphical user interface (GUI) that allows users to upload an MRI image and obtain the predicted class along with the prediction probabilities.

**Note:** This project is intended for academic purpose only and is not a medical diagnostic tool. 

## classification classes
The models classify MRI images into four categories:
1. Glioma
2. Meningioma
3. No tumor
4. Pituitary

## dataset
The dataset consists of 8037 T-1 weighted brain MRI images in JPG format. 

The dataset was compiled from publicly available brain MRI datasets and combined to increase the amount of training data and reduce class imbalance.

**Dataset**                **Num of Images**
1. Training                      5464
2. Validation                    965
3. Testing                       1068
4. **Total**                   **8037**

The data was split using stratified sampling to maintain the class distribution accross datasets.

## preprocessing
The following preprocessing pipeline was applied to the brain MRI images:
1. Brain region cropping
2. Square padding
3. Noise reduction using bilateral filtering
4. Contrast enhancement using CLAHE
5. Resizing to 224x224 pixels
6. Pixel normalization

Data augmentation was applied to the training data by using rotation, zoom, horizontal flipping, shear, and shifting.

## models
Four deep learning approaches were implemented and evaluated:
1. Custom CNN
2. ResNet50
3. VGG16
4. VGG19

The transfer learning models use ImageNet-pretrained weights followed by additional classification layers and fine tuning.

## results
The models were evaluated using the test dataset.
**Model**              **Test Acc**
Custom CNN              97.38%
ResNet50                97.89%
VGG16                   97.76%
VGG19                   98.45%

Additional evaluation was performed using precision, recall, F1-score, and confusion matrix.

## gui application
The project includes a desktop GUI developed using Python Tkinter.

Main features include:
1. MRI image upload
2. Direct image classification
3. Prediction probabilities
4. Model selection (Custom CNN, ResNet50, VGG16, VGG19)
5. Preprocessing preview
6. Diagnosis result display
7. Reset functionality

## technologies
1. Python
2. TensorFlow/Keras
3. OpenCV
4. NumPy
5. Scikit-learn
6. Matplotlib
7. Tkinter
