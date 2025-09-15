import tensorflow.keras as keras
from tensorflow.keras import layers
import tensorflow as tf

def sampling(args):
    """
    Uses (z_mean, z_log_var) to sample latent vector z,
    implementing the reparameterization trick.
    """
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a Variational Autoencoder (VAE).

    Parameters:
        input_dims (int): Dimensions of the input data (e.g., 784 for flattened 28x28 images).
        hidden_layers (list[int]): Number of units in each hidden layer of the encoder.
        latent_dims (int): Dimension of the latent space.

    Returns:
        encoder (Model): Encoder model outputting (z, z_mean, z_log_var).
        decoder (Model): Decoder model reconstructing input from latent space.
        vae (Model): Full VAE model compiled with Adam optimizer and custom loss.
    """
    # Encoder
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for units in hidden_layers:
        x = layers.Dense(units, activation='relu')(x)

    # Mean and log variance layers for latent space
    z_mean = layers.Dense(latent_dims, activation=None, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dims, activation=None, name='z_log_var')(x)

    # Sampling layer using reparameterization trick
    z = layers.Lambda(sampling, output_shape=(latent_dims,), name='z')([z_mean, z_log_var])

    encoder = keras.Model(inputs, [z, z_mean, z_log_var], name='encoder')

    # Decoder
    latent_inputs = keras.Input(shape=(latent_dims,), name='z_sampling')
    x = latent_inputs
    for units in reversed(hidden_layers):
        x = layers.Dense(units, activation='relu')(x)
    outputs = layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(latent_inputs, outputs, name='decoder')

    # VAE model = encoder + decoder
    outputs = decoder(encoder(inputs)[0])
    vae = keras.Model(inputs, outputs, name='vae')

    # Define VAE loss (reconstruction loss + KL divergence)
    reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
    reconstruction_loss *= input_dims  # scale by number of input dims

    kl_loss = 1 + z_log_var - keras.backend.square(z_mean) - keras.backend.exp(z_log_var)
    kl_loss = keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)

    # Compile model with Adam optimizer
    vae.compile(optimizer='adam')

    return encoder, decoder, vae

