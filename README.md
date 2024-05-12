
# Medical Scan Classification using TinyVGG

This repository contains the implementation of a convolutional neural network (CNN) based on the TinyVGG architecture to classify six different types of medical scans. The model is trained and tested on the Medical MNIST dataset which includes various types of medical images such as X-rays and CT scans.

## Project Overview

- **Model**: Utilizes a TinyVGG-based architecture adapted for medical image classification.
- **Dataset**: Medical MNIST from Kaggle, containing 60,000 images of size 64x64 pixels, labeled across six categories: Head CT, Hand X-Ray, Chest CT, CXR (Chest X-Ray), Breast MRI, and Abdomen CT.
- **Features**: Includes implementation of a confusion matrix to analyze model performance and misclassifications, particularly between Hand X-Ray and CXR.
- **Utility**: The code also includes a feature attribution mechanism prepared for integration with Grad-CAM using OpenCV (not fully implemented).

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/TommasoMonicelli/MedicalMNIST_TinyVGG.git
   cd MedicalMNIST_TinyVGG
   cd repo
   ```

2. **Install required libraries:**
   ```bash
   pip install torch torchvision torchmetrics sklearn matplotlib mlxtend
   ```

3. **Download the dataset:**
   - Download the Medical MNIST dataset from [Kaggle](https://www.kaggle.com/datasets/andrewmvd/medical-mnist/data).
   - Extract the dataset into a suitable directory, e.g., `C:/Users/yourusername/Downloads/archive_MedMnist`.

4. **Set up the paths:**
   - Update the dataset directory path and the path where model parameters will be saved in the script accordingly.

## Usage

To run the training and evaluation of the model, execute the main script:

```bash
python Project_Main_1.py
```

This will:
- Train the model for a default of 5 epochs (modifiable).
- Output training and validation losses and accuracies.
- Save the trained model.
- Plot and display a confusion matrix of the predictions.
- Optionally, visualize some misclassified samples.

## Customization

- **Number of Epochs**: Modify the number of epochs in the training loop as required.

## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your proposed changes.

## License

This project is open source and available under the [MIT License](LICENSE).

## Authors

- Tommaso Monicelli - Initial work and development.

## Acknowledgments

- This project would not have been possible without the foundational models and code provided by the following individuals and resources:

- **Daniel Bourke**: The training functions in this project were adapted from Daniel Bourke's work on his PyTorch deep learning tutorial. His material has been instrumental in guiding the structure of our model's training process. More information and the original code can be found on his GitHub repository: [PyTorch Deep Learning by Daniel Bourke](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/03_pytorch_computer_vision.ipynb).
- Thanks to the creators of the Medical MNIST dataset for providing a rich dataset for medical image analysis.
- Inspired by the TinyVGG architecture and adapted to suit medical imaging needs.

```
