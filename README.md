# Autonomous Driving Scene Segmentation with U Net
![Project Status](https://img.shields.io/badge/status-Completed-brightgreen.svg)
![Platform](https://img.shields.io/badge/platform-Kaggle-pink.svg)
![Environment](https://img.shields.io/badge/environment-Jupyter%20Notebook-orange.svg)
![Language](https://img.shields.io/badge/language-Python-blue.svg) 
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

Autonomous vehicles require pixel level understanding of the environment for safe navigation. Traditional computer vision methods fail in complex situations like occlusions, shadows, bright lights, rain, and urban clutter. The challenge is to segment CARLA road scene images into meaningful classes including roads, vehicles, pedestrians, sidewalks, buildings, vegetation, sky, and background. U Net provides an effective end to end learning solution for pixelwise segmentation. This project uses a U Net based deep learning model to perform pixel level semantic segmentation on CARLA simulator images. The goal is to classify each pixel of a driving scene into categories such as road, vehicles, pedestrians, buildings, vegetation, and background. The project demonstrates how deep learning improves scene understanding for autonomous driving systems.

---

## Dataset Information

Dataset Source: **CARLA Simulator Semantic Segmentation Dataset**
This dataset includes synthetic driving scenes with perfectly labeled ground truth masks.

Contents:

* Around 5000 RGB images
* Each image has a corresponding pixel labeled segmentation mask
* Classes include road, vehicles, pedestrians, buildings, vegetation, road lines, poles, sky, etc
* Data is divided into: dataA, dataB, dataC, dataD, dataE (folders)

Advantages:

* Pixel perfect labels
* No noise or annotation errors
* Helps build strong baseline models before moving to real datasets

---

## Data Preprocessing

Steps included in the project:

1. Load RGB input images and their ground truth segmentation masks
2. Resize all images and masks to a consistent shape (256 × 256)
3. Normalize pixel values to [0, 1]
4. Convert masks into one hot encoded format for multi class prediction
5. Visualize image–mask pairs to ensure correct alignment
6. Split into training, validation, and test sets

Purpose:
Preprocessing ensures the model receives uniform and clean inputs, improving accuracy and convergence.

---

## U Net Model Architecture

The model follows the original U Net structure:

### Encoder

* Repeated Conv2D + ReLU
* Max Pooling layers
* Extracts features and reduces spatial dimension

### Decoder

* Transposed Convolutions (upsampling)
* Reconstructs image resolution
* Combines encoder features using skip connections

### Skip Connections

* Transfer fine spatial details directly from encoder to decoder
* Help the network learn thin and small structures such as road markings or poles

### Output Layer

* Produces a multi class segmentation map
* Each pixel receives a probability for each class

---

## Training Configuration

Loss Function: **Categorical Cross Entropy**
Optimizer: **Adam**
Hyperparameters:

* Batch size: 32
* Epochs: 20
* Learning Rate: 0.001
* Validation split: 20 percent

Training Tools:

* ModelCheckpoint: saves best model
* EarlyStopping: prevents overfitting

---

## Model Performance

Training Accuracy: **98.77 percent**
Validation Accuracy: **98.67 percent**
Test Accuracy: **98.58 percent**

These results show that the model correctly learns scene features and generalizes well across unseen images.

---

## Evaluation Metrics

The following metrics were used:

* **Accuracy**: percentage of correctly predicted pixels
* **Precision**: correctness of predicted positive pixels
* **Recall**: ability to detect all true positive pixels
* **Specificity**: ability to avoid false positives
* **Mean Intersection over Union (mIoU)**: overlap between prediction and ground truth
* **F1 Score / Dice Coefficient**: harmonic balance between precision and recall
* **True Detection Rate (TDR)**: how accurately the target object is segmented

These metrics provide both pixel level and class wise performance insights.

---

## Results and Visualizations

The project demonstrates:

* Predicted segmentation maps compared with ground truth masks
* Overlay of predictions on original RGB images
* Clear separation of objects such as roads, vehicles, sidewalks, and buildings
* High quality reconstruction of scene elements

Visual outputs prove that U Net can reliably interpret complex driving scenes.

---

## Key Observations

* Strong performance on structured objects like roads and buildings
* Good detection of vehicles and pedestrians
* Minor confusion in thin structures such as poles and wires
* Dataset quality helps training stability

---


## Tools Used

* Python
* TensorFlow or PyTorch
* NumPy
* Matplotlib
* CARLA Simulator dataset
* Jupyter Notebook

---

## Conclusion

This project successfully implements U Net for multi class semantic segmentation on CARLA driving scenes.
The model achieves high accuracy and strong segmentation quality, demonstrating the usefulness of U Net for autonomous driving perception tasks.

By converting raw images into pixel level class maps, the system supports decision making in tasks like navigation, lane detection, path planning, and obstacle avoidance.

---

## Future Improvements

* Add data augmentation (brightness, rotation, weather effects)
* Use Dice loss or combined BCE + Dice loss
* Incorporate Attention U Net or DeepLabV3+
* Train on real world datasets for better robustness
* Convert the model for edge deployment or real time inference

---
