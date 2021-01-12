# face-generator-VAE
An attempt to generate faces using a convolutional variational autoencoder by training on the UTKFace dataset

# Weights

The weights can be downloaded at https://drive.google.com/drive/folders/1KDRJRzU0rXM8M6CFXn9glJ-BZhI4a4hZ?usp=sharing

# Dataset used
The dataset used is the cropped version of the UTKFace dataset. The dataset contains 128 by 128 images of human faces.  
It can be found [here](https://susanqq.github.io/UTKFace/)

# Objective
The objective of this project is to train a variational autoencoder to map an image to a multivariate probability distribution in a 2048
dimensional space whose distribution is a standard normal distribution. Random vectors can then b sampled from the distribution to generate random faces.

The model uses 'mean squared error' for it's reconstruction loss and KL divergence loss.

# Training
The model was trained for 300 epochs on a GPU instance.

# Libraries used
- [Tensorflow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [Numpy](https://numpy.org/)

# Acknowledgments
- Ahlad Kumar's series on [Youtube](https://www.youtube.com/watch?v=w8F7_rQZxXk&list=PLdxQ7SoCLQANizknbIiHzL_hYjEaI-wUe) on variational autoencoders on youtube do help understand the theory
- The article "Intuitively Understanding Variational Autoencoders" on [Medium](https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf)
