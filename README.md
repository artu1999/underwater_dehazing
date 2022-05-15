# Using GANs to reconstruct and enhance the quality of underwater hazed images

This model takes inspiration from the work proposed with FUnIE-GAN, with a few modifications:

- A perceptual colour distance function is used to enhance the quality of reconstructed colours.

- The generator architecture was changed to one inspired by the one used in EnlightenGAN, so instead of using a nearest-neighbour upsampling layer the generator is fully-convolutional.

- The generator loss was changed from MSE to MAE to improve its stability. Additionally, one-sided label smoothing was used to improve the generator's adversarial defense.

The generator architecture is composed by 8 upsampling and 8 downsampling blocks. 

![alt text](https://github.com/artu1999/underwater_dehazing/blob/main/images/3.5.2_UNet_CGAN.png?raw=true)

![alt text](https://github.com/artu1999/underwater_dehazing/blob/main/images/3.5.2_UNet_blocks.png?raw=true)

Additionally, a smaller version of the generator is also available, which is identical to this one but with only 5 and 5 blocks instead.


Some dehazing examples:

![alt text](https://github.com/artu1999/underwater_dehazing/blob/main/images/dehazed_examples.png?raw=true)


References:

- FUnIE-GAN [github](https://github.com/xahidbuffon/FUnIE-GAN), [paper](https://ieeexplore.ieee.org/document/9001231)

- EnlightenGAN [paper](https://arxiv.org/pdf/1906.06972.pdf)

- Colour perceptual distance [Definition](https://www.compuphase.com/cmetric.htm), [implementation](github.com/wandb/superres)


