# Using GANs to reconstruct and enhance the quality of underwater hazed images

This model takes inspiration from the work proposed with FUnIE-GAN, with a few modifications:

- A perceptual colour distance function is used to enhance the quality of reconstructed colours.

- The generator architecture was changed to one inspired by the one used in EnlightenGAN, so instead of using a nearest-neighbour upsampling layer the generator is fully-convolutional.

- The generator loss was changed from MSE to MAE to improve its stability. Additionally, one-sided label smoothing was used to improve the generator's adversarial defense.


