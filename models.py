import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow_addons.layers import InstanceNormalization
from losses import MSE, MAE, reduce_mean
import os
from tqdm import tqdm


def downsample(filters, size, input_layer):
    init = keras.initializers.RandomNormal(stddev=0.02)
    g = layers.Conv2D(filters, size, strides=2, padding='same',
                      kernel_initializer=init, use_bias=False)(input_layer)
    g = InstanceNormalization(axis=-1)(g)
    g = layers.Activation('relu')(g)
    return g


def upsample(filters, size, input_layer):
    init = keras.initializers.RandomNormal(stddev=0.02)
    g = layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                               kernel_initializer=init, use_bias=False)(input_layer)
    g = InstanceNormalization(axis=-1)(g)
    g = layers.Activation('relu')(g)
    return g


def large_generator(image_shape=(256, 256, 3)):
    init = keras.initializers.RandomNormal(stddev=0.02)
    in_image = keras.Input(shape=image_shape)

    down1 = downsample(64, 4, in_image)
    down2 = downsample(128, 4, down1)
    down3 = downsample(256, 4, down2)
    down4 = downsample(512, 4, down3)
    down5 = downsample(512, 4, down4)
    down6 = downsample(512, 4, down5)
    down7 = downsample(512, 4, down6)
    down8 = downsample(512, 4, down7)

    up1 = upsample(512, 4, down8)
    up1 = layers.Concatenate()([up1, down7])
    up2 = upsample(512, 4, up1)
    up2 = layers.Concatenate()([up2, down6])
    up3 = upsample(512, 4, up2)
    up3 = layers.Concatenate()([up3, down5])
    up4 = upsample(512, 4, up3)
    up4 = layers.Concatenate()([up4, down4])
    up5 = upsample(256, 4, up4)
    up5 = layers.Concatenate()([up5, down3])
    up6 = upsample(128, 4, up5)
    up6 = layers.Concatenate()([up6, down2])
    up7 = upsample(64, 4, up6)
    up7 = layers.Concatenate()([up7, down1])

    up8 = layers.Conv2DTranspose(3, 4, strides=2, padding='same',
                                 kernel_initializer=init, use_bias=False)(up7)
    out_image = layers.Activation('tanh')(up8)

    model = keras.models.Model(in_image, out_image)

    return model


def tiny_generator(image_shape=(256,256,3)):
    init = keras.initializers.RandomNormal(stddev=0.02)
    in_image = keras.Input(shape=image_shape)

    down1 = downsample(64, 4, in_image)
    down2 = downsample(128, 4, down1)
    down3 = downsample(256, 4, down2)
    down4 = downsample(512, 4, down3)
    down5 = downsample(512, 4, down4)

    up1 = upsample(512, 4, down5)
    up1 = layers.Concatenate()([up1, down4])
    up2 = upsample(256, 4, up1)
    up2 = layers.Concatenate()([up2, down3])
    up3 = upsample(128, 4, up2)
    up3 = layers.Concatenate()([up3, down2])
    up4 = upsample(64, 4, up3)
    up4 = layers.Concatenate()([up4, down1])

    up5 = layers.Conv2DTranspose(3, 4, strides=2, padding='same',
                      kernel_initializer=init, use_bias=False)(up4)
    out_image = layers.Activation('tanh')(up5)

    model = keras.models.Model(in_image, out_image)

    return model


def VGG19_Content(dataset='imagenet'):
    # Load VGG, trained on imagenet data
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights=dataset)
    vgg.trainable = False
    content_layers = ['block5_conv2']
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    return keras.models.Model(vgg.input, content_outputs)


def discriminator(image_shape):
    """
        Inspired by the pix2pix discriminator.
        Thanks to https://github.com/xahidbuffon/FUnIE-GAN
    """
    def d_layer(layer_input, filters, strides_=2,f_size=3, bn=True):
        ## Discriminator layers
        d = layers.Conv2D(filters, kernel_size=f_size, strides=strides_, padding='same')(layer_input)
        #d = LeakyReLU(alpha=0.2)(d)
        d = layers.Activation('relu')(d)
        if bn: d = layers.BatchNormalization(momentum=0.8)(d)
        return d

    img_A = keras.Input(shape=image_shape)
    img_B = keras.Input(shape=image_shape)
    ## input
    combined_imgs = layers.Concatenate(axis=-1)([img_A, img_B])
    ## Discriminator layers
    d1 = d_layer(combined_imgs, 32, bn=False)
    d2 = d_layer(d1, 32*2)
    d3 = d_layer(d2, 32*4)
    d4 = d_layer(d3, 32*8)
    validity = layers.Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
    # return model
    return keras.models.Model([img_A, img_B], validity)


class FUnIEGAN:
    def __init__(self,
                 image_shape,
                 n_epochs,
                 n_batch,
                 strategy=None,
                 generator = 'large',
                 ckpt_path = os.getcwd() + '/ckpt',
                 **kwargs):
        self.image_shape = image_shape
        self.n_epochs = n_epochs
        self.n_batch = n_batch
        self.strategy = strategy

        if self.strategy:
            with self.strategy.scope():
                if generator == 'large':
                    self.g = large_generator()
                else:
                    self.g = tiny_generator()
                self.d = discriminator(self.image_shape)
                self.g_optimizer = tf.keras.optimizers.Adam(
                    learning_rate=2e-4, beta_1=0.5, beta_2=0.9)
                self.d_optimizer = tf.keras.optimizers.Adam(
                    learning_rate=2e-4, beta_1=0.5, beta_2=0.9)
                self.vgg19 = VGG19_Content()
        else:
            if generator == 'large':
                self.g = large_generator()
            else:
                self.g = tiny_generator()
            self.d = discriminator(self.image_shape)
            self.g_optimizer = tf.keras.optimizers.Adam(
                learning_rate=2e-4, beta_1=0.5, beta_2=0.9)
            self.d_optimizer = tf.keras.optimizers.Adam(
                learning_rate=2e-4, beta_1=0.5, beta_2=0.9)
            self.vgg19 = VGG19_Content()

        self.checkpoint = tf.train.Checkpoint(g=self.g,
                                              d=self.d,
                                              g_optimizer=self.g_optimizer,
                                              d_optimizer=self.d_optimizer)

        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

        self.ckpt_path = ckpt_path

        self.ckpt_manager = tf.train.CheckpointManager(
            checkpoint=self.checkpoint,
            directory=self.ckpt_path,
            max_to_keep=self.n_epochs
        )

        self.ckpt_num = 1
        super(FUnIEGAN, self).__init__(**kwargs)

    def save_checkpoint(self):
        self.ckpt_manager.save(checkpoint_number=self.ckpt_num)
        self.ckpt_num += 1
        print(f"\nsaved checkpoint to {self.ckpt_path}\n")

    def load_checkpoint(self, ckpt_name):
        """ load checkpoint from checkpoint_dir if exists """
        if self.strategy:
            with self.strategy.scope():
                self.checkpoint.restore(self.ckpt_path)
        else:
            self.checkpoint.restore(self.ckpt_path + ckpt_name)
        print(f"Model loaded from {self.ckpt_path + ckpt_name}")
        return

    def train_step(self, datasetA, datasetB):
        result = {}

        with tf.GradientTape(persistent=True) as d_tape, tf.GradientTape(persistent=True) as g_tape:
            fake = self.g(datasetA, training=True)

            discriminate_real = self.d([datasetB, datasetA])
            discriminate_fake = self.d([fake, datasetA])

            d_loss = self.discriminator_loss(discriminate_real, discriminate_fake)
            aspect_loss = self.gen_aspect_loss(datasetB, fake)
            g_loss = self.generator_loss(discriminate_fake)

            g_total_loss = 0.6 * g_loss + 0.4 * aspect_loss

        self.d_optimizer.minimize(loss=d_loss,
                                  var_list=self.d.trainable_variables,
                                  tape=d_tape)
        self.g_optimizer.minimize(loss=g_total_loss,
                                  var_list=self.g.trainable_variables,
                                  tape=g_tape)

        result['g_loss'] = g_loss
        result['aspect_loss'] = aspect_loss
        result['g_total_loss'] = g_total_loss
        result['d_loss'] = d_loss

        return result

    @tf.function
    def distributed_train_step(self, x, y):
        results = self.strategy.run(self.train_step, args=(x, y))
        results = self.reduce_dict(results)
        return results

    def generator_loss(self, discriminate_fake):
        per_sample_loss = MAE(y_true=tf.ones_like(discriminate_fake),
                              y_pred=discriminate_fake)
        return reduce_mean(per_sample_loss, self.n_batch)

    def perceptual_distance(self, y_true, y_pred):
        """
          Calculating perceptual distance
          Thanks to github.com/wandb/superres
        """
        y_true = (y_true + 1.0) * 127.5  # [-1,1] -> [0, 255]
        y_pred = (y_pred + 1.0) * 127.5  # [-1,1] -> [0, 255]
        rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
        r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
        g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
        b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]
        return tf.reduce_mean(
            tf.sqrt((((512 + rmean) * r * r) / 256) + 4 * g * g + (((767 - rmean) * b * b) / 256)))

    def gen_aspect_loss(self, org_content, gen_content):
        # custom perceptual loss function
        vgg_org_content = self.vgg19(org_content)
        vgg_gen_content = self.vgg19(gen_content)
        content_loss = reduce_mean(
            tf.reduce_mean(tf.square(vgg_org_content - vgg_gen_content), axis=-1), self.n_batch)
        mae_gen_loss = reduce_mean(
            tf.reduce_mean(tf.abs(org_content - gen_content)), self.n_batch)
        perceptual_loss = self.perceptual_distance(org_content, gen_content)
        gen_total_err = 0.6 * mae_gen_loss + 0.3 * content_loss + 0.1 * perceptual_loss
        return gen_total_err

    def discriminator_loss(self, discriminate_real, discriminate_fake):
        real_loss = MSE(y_true=tf.ones_like(discriminate_real) - 0.1,
                        y_pred=discriminate_real)
        fake_loss = MSE(y_true=tf.zeros_like(discriminate_fake),
                        y_pred=discriminate_fake)
        per_sample_loss = 0.5 * (real_loss + fake_loss)
        # per_sample_loss = real_loss + fake_loss
        return reduce_mean(per_sample_loss, self.n_batch)

    def reduce_dict(self, d: dict):
        """ inplace reduction of items in dictionary d """
        return {
            k: self.strategy.reduce(tf.distribute.ReduceOp.SUM, v, axis=None)
            for k, v in d.items()
        }

def train(ds, GAN, TRAIN_STEPS, train_results):
    results = {}
    for x, y in tqdm(ds, desc='Train', total=TRAIN_STEPS):
        result = GAN.distributed_train_step(x, y)
        append_dict(results, result)

    for key, value in results.items():
        results[key] = tf.reduce_mean(value).numpy()

    append_dict_vals(train_results, results)


def append_dict(dict1: dict, dict2: dict, replace: bool = False):
    """ append items in dict2 to dict1 """
    for key, value in dict2.items():
        if replace:
            dict1[key] = value.numpy()
        else:
            if key not in dict1:
                dict1[key] = []
            dict1[key].append(value.numpy())


def append_dict_vals(dict1: dict, dict2: dict, replace: bool = False):
    """ append items in dict2 to dict1 """
    for key, value in dict2.items():
        if replace:
            dict1[key] = value
        else:
            if key not in dict1:
                dict1[key] = []
            dict1[key].append(value)

