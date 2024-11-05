# Cat and Dog Classifier using VGG16 and PyTorch

This project implements a cat and dog image classifier using PyTorch and a pre-trained VGG16 model. It includes scripts for training the model, applying Grad-CAM for visualization, and predicting new images.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Data Preparation](#data-preparation)
- [Training the Model](#training-the-model)
- [Applying Grad-CAM](#applying-grad-cam)
- [Making Predictions](#making-predictions)
- [Acknowledgments](#acknowledgments)

---

## Prerequisites

Before running the scripts, ensure you have the following installed:

- **Python 3.x**
- **PyTorch**
- **Torchvision**
- **NumPy**
- **Matplotlib**
- **OpenCV (cv2)**
- **Pillow**

Install the required packages using pip:


pip install torch torchvision numpy matplotlib opencv-python pillow
Project Structure
├── split_dataset.py       # Script to split data into training and validation sets
├── train_model.py         # Script to train the VGG16 model
├── grad_cam.py            # Script to apply Grad-CAM visualization
├── predict.py             # Script to predict new images
├── vgg16_cats_dogs.pth    # Trained model weights (generated after training)
├── README.md              # This README file
## Data Preparation
1. Organize Your Dataset
Prepare a dataset of cat and dog images organized into separate folders:

markdown
dataset/
    cats/
        cat001.jpg
        cat002.jpg
        ...
    dogs/
        dog001.jpg
        dog002.jpg
        ...
2. Split Dataset into Training and Validation Sets
If you have a single folder for each class and need to split the data:

Modify split_dataset.py:
original_dataset_dir = 'path_to_original_dataset'  # Your dataset path
base_dir = 'path_to_split_dataset'                 # Destination path for split data
split_ratio = 0.8                                  # 80% training, 20% validation
Run the script:
python split_dataset.py
Resulting Directory Structure:
path_to_split_dataset/
    train/
        cats/
            cat001.jpg
            ...
        dogs/
            dog001.jpg
            ...
    validation/
        cats/
            cat101.jpg
            ...
        dogs/
            dog101.jpg
            ...
## Training the Model
1. Modify train_model.py
Set the Data Directory:
data_dir = 'path_to_split_dataset'  # Replace with your actual path
Adjust Training Parameters (Optional):
num_epochs = 10  # Number of training epochs
2. Run the Training Script
python train_model.py
Output:

The trained model weights will be saved as vgg16_cats_dogs.pth.
## Applying Grad-CAM
1. Modify grad_cam.py
Set the Image Path:
img_path = 'path_to_image'  # Replace with the path to your image
2. Run the Grad-CAM Script
python grad_cam.py
Output:

A visualization image cam.jpg showing the heatmap overlay on the original image.
## Making Predictions
1. Modify predict.py
Set the Image Path:
img_path = 'path_to_image'  # Replace with the path to your image
2. Run the Prediction Script
python predict.py
Output:

The script will print whether the image is predicted to be a cat or a dog.
模型预测结果：cat
## Acknowledgments
This project utilizes the pre-trained VGG16 model from PyTorch's model zoo.

Special thanks to the open-source community for providing resources and tools that make projects like this possible.

Note: Ensure all scripts are in the same directory, and paths in the scripts are correctly set to reflect your local file system. If you encounter any issues, please check that all dependencies are installed and that your data is correctly organized.

