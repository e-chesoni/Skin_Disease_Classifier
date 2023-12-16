# Skin Disease Detection Using PyTorch
This is Elaine Chesoni, Meiru Han, and Ankita Reddy's final project for CIS 5810.

## Download Data

You need will need to download the [Diverse Dermatology Images](https://ddi-dataset.github.io/) and place a folder called 'dataset' located at the project root.

## Installation

To run the notebook, you need to install the required packages:

```bash
pip install torchsummary
```

## Dataset

The dataset used in this project is a collection of dermatology images, each labeled with the corresponding skin condition. The data is preprocessed and augmented to improve the model's performance.

## Preprocessing and Data Augmentation
- Images are loaded and preprocessed to a 3-channel RGB format.
- Data augmentation is performed using `ImageDataGenerator` from Keras, including rotations, shifts, flips, and zooms.

## Pretrained Models

### Overview
We utilize pretrained models from PyTorch, specifically VGG16, VGG19, and ResNet50, due to their proven effectiveness in image classification tasks. These models are pretrained on the ImageNet dataset, which includes a vast array of images across numerous categories.

#### Inspiration

The Pretrained_Models.ipynb notebook contains the implementation of a skin disease detection system using PyTorch. The project is inspired by a [Pneumonia detection study](https://www.kaggle.com/code/dnik007/pneumonia-detection-using-pytorch) and aims to differentiate between malignant (neoplastic) and benign (non-neoplastic) dermatological conditions.

### Customization of Pretrained Models

The models are customized by modifying their lower layers. This approach, inspired by a pneumonia detection study, involves freezing the initial layers and replacing the classification module with a custom-designed one. This strategy helps mitigate overfitting and leverages the pretrained models' ability to capture low-level features.

### Training

The models are trained using a negative log likelihood loss function and an Adam optimizer. The training process includes early stopping to prevent overfitting.

### Challenges and Observations

- The models performed well on the training set but underperformed on the validation set, indicating potential overfitting.
- The dataset's smaller size and greater variability in images compared to the pneumonia dataset might have contributed to these challenges.

### Future Work

- Exploring different modifications to the bottom layers of the models to enhance performance for skin disease classification.
- Implementing more targeted and uniform photo preprocessing techniques, such as centering the area of interest.

### Pre-training Notebook Usage

To train and evaluate the models, follow the steps outlined in the Colab Notebook. Ensure that you have the necessary data and modify the folder paths as per your setup.

## Skin Disease Detection Using VAE

### Overview
The St_Classification_with_VAE.ipynb Colab Notebook aims to create an image classifier using Variational Autoencoder (VAE) extracted features to distinguish between malignant and benign skin diseases.

## VAE Model Development
The project involves training Variational Autoencoders (VAEs) with different latent dimensions (32, 64, 128, 256) to learn efficient representations of the input images. These representations are then used for the classification task.

### Feature Extraction
- Latent representations are extracted from the trained VAE models.
- These features, along with one-hot-encoded skin tone data, are used for the classification task.

### Classification
Two Logistic Regression classifiers are trained:
1. Using only the extracted features from VAE.
2. Using both extracted features and skin tone information.

### Results and Observations
- The accuracy of the classifiers is evaluated on the test set.
- The impact of including skin tone information on the classification performance is analyzed.

### VAE Notebook Usage
To train the VAE models, extract features, and perform classification, follow the steps outlined in the Jupyter Notebook. Ensure that you have the necessary data and modify the folder paths as per your setup.

## Contributions

Contributions to this project are welcome. Please ensure that any pull requests or issues adhere to the project's guidelines.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

*Note: This README is a brief overview of the project. For detailed instructions and explanations, refer to the Colab Notebook provided.*
