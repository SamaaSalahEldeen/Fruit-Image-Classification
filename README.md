overview 
This repository contains Python scripts for image classification using three different deep learning models: AlexNet, Basic Convolutional Neural Network (CNN), and a Transformer-based model. The code is implemented using the Keras and TensorFlow libraries.
Requirements
Python 3.x
Keras
TensorFlow
OpenCV
NumPy
Pandas
Scikit-Learn
1. Basic Convolutional Neural Network (CNN)
Overview
The script defines a basic CNN model using Keras for image classification.
Training is performed using the ImageDataGenerator for data augmentation.
The model is trained and evaluated on the specified dataset.
Execution
Set the 'path' variable to the location of your training dataset.
Run the script.
2. AlexNet
Overview
The script implements the AlexNet architecture for image classification.
Training is conducted with early stopping and model checkpoint callbacks.
The model is saved as 'alexnet_best.h5'.
Execution
Set the 'path' variable to the location of your training dataset.
Run the script.
3. Transformer-based Model
Overview
The script defines a Transformer-based model using the ResNet50 backbone.
The model is trained with a learning rate schedule.
The trained model is saved as 'transformer2.h5'.
Execution
Set the 'path' variable to the location of your training dataset.
Run the script.
Model Evaluation
Overview
Model evaluation is performed on a test dataset located at 'test_dataset_path'.
The 'test_images' function generates predictions and saves results to a CSV file.
Execution
Set the 'test_dataset_path' variable to the location of your test dataset.
Run the 'test_images' function for each model.
