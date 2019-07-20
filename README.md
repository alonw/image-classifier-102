# image-classifier-102

Project code for Udacity's AI Programming with Python Nanodegree program: an image classifier built with PyTorch, then converted into a command line application.

The image classifier is built with pytorch that classifies a flower image to it's category. The classifier currently has training, validation, and test data for 102 flowers and uses transfer learning with either VGG19 or Densenet161 to train and infer with.

Tools are packages: Python, Numpy, Pytorch, Torchvision, Matplotlib

## How It Works
1. An existing collection of classified flower images used to train a neural network model
2. Also using a pretrained network (transer learning)
3. Once the model is ready, it's saved and validated
4. Prediction uses the trained network and returns the most probable flower classifications (ordered). 



## Licence
[MIT](https://opensource.org/licenses/MIT)
