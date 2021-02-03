# Variational Auto-Encoder for MNIST

This is an implementation of variational auto-encoder(VAE) for MNIST

## Usage

### requirements
```
tensorflow >= 2.0
numpy
argparse
matplotlib
cv2
```

### Command
```
python main.py --[argument] <value>
```

*Example* : `python main.py --dim_z 4`

### Arguments

* `--dim_z` : Dimension of latent vector. *Default* : `4`
* `--output` : File path of output images. *Default* : `results`
* `--dataset` : The name of Dataset. List : ['mnist', 'fashion-mnist']. *Default* : `mnist`
* `--seed` : Random Seed. *Default* : `0`
* `--add_noise` : Boolean for adding noise to input image(gaussian noise). *Default* : `False`
* `--noise_factor` : Factor of gaussian noise. *Default* : `0.7`
* `--n_hidden` : The number of hidden units in MLP. *Default* : `500`
* `--learning_rate` : Learning rate of adam optimizer. *Default* : `1e-3`
* `--num_epochs` : The number of epochs to run. *Default* : `40`
* `--batch_size` : Batch size. *Default* : `128`

* `--PRR_n_img_x` : Number of images along x-axis. *Default* : `10`
* `--PRR_n_img_y` : Number of images along y-axis. *Default* : `10`
* `--PMLR_n_img_x` : Number of images along x-axis. *Default* : `20`
* `--PMLR_n_img_y` : Number of images along y-axis. *Default* : `20`
* `--PMLR_z_range` : Rnage for uniformly distributed latent. *Default* : `2.0`
* `--PMLR_n_samples` : Number of samples in order to get distribution of labeled data. *Default* : `5000`

## Results

### Reproduce

Well trained VAE must be able to reproduce input images.

Images below show reproduce performance of learned generative models for different dimensionalities.

Command : `python main.py --dim_z <each value of letent space> --num_epochs 60`

|Input image|2-D latent space|4-D latent space|10-D latent space|20-D latent space|
|:---:|:---:|:---:|:---:|:---:|
|<img src="results/input.jpg">|<img src="results/PRR_epoch_60_dim2.jpg">|<img src="results/PRR_epoch_60_dim4.jpg">|<img src="results/PRR_epoch_60_dim10.jpg">|<img src="results/PRR_epoch_60_dim20.jpg">|

### Denoising

VAE can reduce noise and restore original input images if noise is added to input images when training.

Command : `python main.py --dim_z 20 --add_noise True --num_epochs 60`
|Original Input image|Input image with noise|Restored image via VAE|
|:---:|:---:|:---:|
|<img src="results/input.jpg">|<img src="results/input_noise.jpg">|<img src="results/PRR_epoch_60_dim20_denoising.jpg">|

### Learned MNIST manifold

VAE can learn data manifold via generative models with 2-dim.

If dimension of z is 2, images for visualization of learned data manifold should be generated.

Command : `python main.py --dim_z 2 --num_epochs 60`

|Learned MNIST manifold|Distribution of labeled data|
|:---:|:---:|
|<img src="results/PMLR_epoch_60.jpg">|<img src="results/PMLR_map_epoch_60.jpg">|

## Reference

The implementation is based on 
* https://github.com/hwalsuklee/tensorflow-mnist-VAE
* [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114.pdf)
