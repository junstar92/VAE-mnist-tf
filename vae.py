# -*- coding: utf-8 -*-
import tensorflow as tf

class VAE(tf.keras.Model):
    def __init__(self, input_shape, args):
        super(VAE, self).__init__()
                
        # encoder
        self.encoder = gaussian_encoder(input_shape, args)
        
        # z sampling layer
        self.sampling_layer = Sampling()
        
        # decoder
        self.decoder = bernoulli_decoder(args, input_shape)
    
    def KLD_loss(self, mu, sigma):
        loss = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.math.log(1e-8 + tf.square(sigma)) - 1, 1)
        loss = tf.reduce_mean(loss)
        
        return loss
    
    def BCE_loss(self, x, y):
        loss = tf.reduce_sum(x*tf.math.log(y) + (1-x)*tf.math.log(1-y), 1)
        
        return -tf.reduce_mean(loss)
    
    def call(self, x):
        mu, sigma = self.encoder(x)
        z = self.sampling_layer((mu, sigma))        
        x_hat = self.decoder(z)
        x_hat = tf.clip_by_value(x_hat, 1e-8, 1-1e-8)
        
        # loss
        bce_loss = self.BCE_loss(x, x_hat)
        kld_loss = self.KLD_loss(mu, sigma)
        
        return x_hat, bce_loss, kld_loss

def gaussian_encoder(input_shape, args):
    # initializer
    w_init = tf.keras.initializers.glorot_normal(args.seed)
    
    # input
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    # hidden layer
    x = tf.keras.layers.Dense(args.n_hidden, activation='elu', 
                              kernel_initializer=w_init)(inputs)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(args.n_hidden, activation='tanh',
                              kernel_initializer=w_init)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    
    # output
    mean = tf.keras.layers.Dense(args.dim_z, kernel_initializer=w_init)(x)
    stddev = tf.keras.layers.Dense(args.dim_z, kernel_initializer=w_init)(x)
    
    encoder = tf.keras.Model(inputs, [mean, stddev])
    encoder.summary()
    
    return encoder

def bernoulli_decoder(args, n_output):
    # initializer
    w_init = tf.keras.initializers.glorot_normal(args.seed)
    
    # input
    inputs = tf.keras.layers.Input(shape=args.dim_z)
    
    # hidden layer
    x = tf.keras.layers.Dense(args.n_hidden, activation='tanh',
                              kernel_initializer=w_init)(inputs)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(args.n_hidden, activation='elu',
                              kernel_initializer=w_init)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    
    # output
    x = tf.keras.layers.Dense(n_output, activation='sigmoid',
                              kernel_initializer=w_init)(x)
    
    decoder = tf.keras.Model(inputs, x)
    decoder.summary()
    
    return decoder

class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        mu, sigma = inputs
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]

        z = mu + sigma * tf.random.normal((batch, dim), 0, 1)

        return z