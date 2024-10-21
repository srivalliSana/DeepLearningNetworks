# Deep Learning Networks

This repository contains implementations of popular deep learning architectures. The models are presented in Jupyter notebooks and cover a wide range of neural network architectures for deep learning tasks.

## Repository Structure
This repository includes the following network implementations:
- **ALEXNET.ipynb**: A notebook implementing the AlexNet architecture, designed for image classification tasks.
- **EFFNET AND VGG.ipynb**: A notebook that includes implementations of EfficientNet and VGGNet, two powerful convolutional neural networks (CNNs).
- **INCEPTION NET.ipynb**: A notebook for the InceptionNet (GoogLeNet) model, known for its multi-scale processing in image recognition tasks.
- **RESNET.ipynb**: A notebook implementing ResNet (Residual Networks), which addresses the vanishing gradient problem in deep networks by introducing skip connections.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Models Overview](#models-overview)
  - [AlexNet](#alexnet)
  - [EfficientNet and VGGNet](#efficientnet-and-vggnet)
  - [InceptionNet](#inceptionnet)
  - [ResNet](#resnet)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction
This repository contains implementations of popular deep learning models, each designed to solve image classification tasks. These models are widely used in computer vision and are implemented in Jupyter notebooks for ease of use, with detailed comments and explanations for learning purposes.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/srivalliSana/DeepLearningNetworks.git
    cd DeepLearningNetworks
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

   The `requirements.txt` should list the necessary Python packages (such as `tensorflow`, `keras`, `torch`, `numpy`, etc.) that are used in the notebooks.

3. Open Jupyter notebooks:

    ```bash
    jupyter notebook
    ```

4. Open any notebook from the list above to explore the models.

## Models Overview

### AlexNet
- **File**: `ALEXNET.ipynb`
- **Overview**: AlexNet is a convolutional neural network designed for image classification. It was the breakthrough model in the ImageNet competition, achieving significant improvements in accuracy.

### EfficientNet and VGGNet
- **File**: `EFFNET AND VGG.ipynb`
- **Overview**: 
  - **EfficientNet**: A family of models that balance accuracy and efficiency by scaling network width, depth, and resolution in a principled way.
  - **VGGNet**: A deeper convolutional network that uses small 3x3 filters to achieve state-of-the-art performance in image classification.

### InceptionNet (GoogLeNet)
- **File**: `INCEPTION NET.ipynb`
- **Overview**: InceptionNet is known for its innovative "Inception module", which processes data at multiple scales in parallel, significantly improving model performance without drastically increasing computation.

### ResNet
- **File**: `RESNET.ipynb`
- **Overview**: ResNet introduces residual learning with skip connections, allowing very deep networks to be trained by mitigating the vanishing gradient problem.

## Usage

To run a specific model, open the corresponding notebook and execute the cells in order. You can also modify the models or data loaders as per your requirement. Each notebook includes:
- A brief explanation of the architecture.
- Code for defining the model.
- Code for training and evaluating the model.

For example, to run the ResNet implementation:
1. Open `RESNET.ipynb`.
2. Follow the instructions in the notebook to load the dataset and train the model.

## Contributing

Contributions are welcome! If you'd like to improve the code, fix bugs, or add new models, please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add a new feature'`).
4. Push your branch (`git push origin feature-branch`).
5. Open a pull request.



## Contact

For any queries, feel free to reach out to:

- **Your Name**: srivalliSana
- **GitHub**: [srivalliSana](https://github.com/srivalliSana)

---


