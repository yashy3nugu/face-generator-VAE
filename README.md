# face-generator-VAE
An attempt to generate faces using a convolutional variational autoencoder by training on the UTKFace dataset

# Dataset used
The dataset used is the cropped version of the UTKFace dataset. The dataset contains 128 by 128 images of human faces.  
It can be found [here](https://susanqq.github.io/UTKFace/)

# Objective
The objective of this project is to train a variational autoencoder to map an image to a multivariate probability distribution in a 2048
dimensional space whose distribution is a standard normal distribution. Random vectors can then be sampled from the distribution to generate random faces.

The objective of this project is to train a variational autoencoder on the UTKFace dataset so that new faces can be generated.  

- Unlike standard encoders which give a discrete values for encodings, Variational autoencoders create a latent space probability distribution.
 Reconstruction can then be done by sampling from the probability distributions in the encodings.  

- Variational autoencoders have a KL divergence term in their loss functions which forces the entire latent space for the encodings to be a standard normal multivariate        distribution.
  random sampling can then be done with ease from this distribution to generate newer faces.

- The model uses convolutional layers for generating encodings and transposed convolutional layers to decode.

- 'mean squared error' has been used for the reconstruction loss.

<img src="https://www.machinecurve.com/wp-content/uploads/2019/12/vae-encoder-decoder.png">


# Training
The model was trained for 300 epochs on a GPU instance.  
The results of the model can be seen below ⬇️.  
<img src="assets/faces.png" height=70% width=70%>

# Weights
The weights of the network after training can be downloaded [here](https://drive.google.com/drive/folders/1KDRJRzU0rXM8M6CFXn9glJ-BZhI4a4hZ?usp=sharing) for testing purposes

# Libraries used
- [Tensorflow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [Numpy](https://numpy.org/)

# Acknowledgments
- Ahlad Kumar's series on [Youtube](https://www.youtube.com/watch?v=w8F7_rQZxXk&list=PLdxQ7SoCLQANizknbIiHzL_hYjEaI-wUe) on variational autoencoders on youtube do help understand the theory
- The article "Intuitively Understanding Variational Autoencoders" on [Medium](https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf)
