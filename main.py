# -*- coding: utf-8 -*-
import os
import glob

from vae import VAE
from utils import *

import tensorflow as tf
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--output', type=str, default='results', help='File path of output images')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'fashion-mnist'], help='The name of dataset')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    
    parser.add_argument('--add_noise', type=bool, default=False, help='boolean for adding noise to input image')
    parser.add_argument('--dim_z', type=int, default=2, help='Dimension of latent vector')#, required=True)
    parser.add_argument('--n_hidden', type=int, default=500, help='Number of hidden units in MLP')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate of adam optimizer')
    parser.add_argument('--num_epochs', type=int, default=20, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    
    #parser.add_argument('--')
    return check_args(parser.parse_args())
    
def check_args(args):
    # --output
    try:
        os.mkdir(args.output)
    except(FileExistsError):
        pass
    # delete all output files
    files = glob.glob(args.output + '/*')
    for file in files:
        os.remove(file)
    
    # --add_noise
    try:
        assert args.add_noise == True or args.add_noise == False
    except:
        print('add_noise must be boolean type')
        return None
    
    # --dim-z
    try:
        assert args.dim_z > 0
    except:
        print('dim_z must be positive interger')
    
    # --n_hidden
    try:
        assert args.n_hidden >= 1
    except:
        print('number of hidden units must be larger than or equal to one')
        
    # --learning_rate
    try:
        assert args.learning_rate > 0
    except:
        print('learning_rate must be positive')
    
    # --num_epochs
    try:
        assert args.num_epochs >= 1
    except:
        print('number of epochs must be larger than or equal to one')
    
    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    
    return args

def main(args):
    # dataset
    x_train, _, x_test, y_test, input_shape = prepare_dataset(args)
    n_samples = x_train.shape[0]
    
    # plot utils
    PRP = Plot_Reproduce_Result(args.output)
    PMLR = Plot_Manifold_Learning_Result(args.output)
    
    # build model
    vae = VAE(input_shape, args)
    
    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    
    # loss metric
    loss_metric = tf.keras.metrics.Mean()
        
    # train
    total_batch = int(n_samples / args.batch_size)
    
    PRP.save_image(x_test[:64], 'input.jpg')    
    for epoch in range(args.num_epochs):
        print(f'Start of epoch {epoch+1}')
        
        # random shuffling
        np.random.shuffle(x_train)
        
        for step in range(total_batch):
            # Compute the offset of the current minibatch in the data
            offset = (step * args.batch_size) % n_samples
            batch_x_train = x_train[offset:(offset + args.batch_size), :]
            with tf.GradientTape() as tape:
                if args.add_noise:
                    pass
                
                x_hat, bce_loss, kld_loss = vae(batch_x_train)
                total_loss = bce_loss + kld_loss
            
            grads = tape.gradient(total_loss, vae.trainable_weights)
            optimizer.apply_gradients(zip(grads, vae.trainable_weights))
            
            loss_metric(total_loss.numpy())
            
            if step % 100 == 0:
                print(f'Epoch: {epoch+1} step: {step} mean loss = {loss_metric.result().numpy()}')
                print(f'BCE loss: {bce_loss}, KL loss: {kld_loss}')
        
        x_hat, _, _ = vae(x_test[:64])
        PRP.save_image(x_hat.numpy(), f'PRR_epoch_{epoch+1:02d}.jpg')
        
        if args.dim_z == 2:
            y_PMLR = vae.decoder(PMLR.z)
            PMLR.save_image(y_PMLR.numpy(), f'PMLR_epoch_{epoch+1:02d}.jpg')
            
            mu, sigma = vae.encoder(x_test[:5000])
            z_PMLR = vae.sampling_layer((mu, sigma))
            PMLR.save_scattered_image(z_PMLR, y_test[:5000], name=f'PMLR_map_epoch_{epoch+1:02d}.jpg')

if __name__ == '__main__':
    # parse arguments
    args = parse_args()
    
    if args is None:
        exit()
    
    main(args)