import tensorflow as tf
import matplotlib.pyplot as plt
from models import FUnIEGAN
import argparse
import os


def load_image(file_path, resize=(256, 256), file_type='png'):
    byte = tf.io.read_file(file_path)
    if file_type == 'png':
        image = tf.io.decode_png(byte, channels=3)
    else:
        image = tf.io.decode_jpeg(byte, channels=3)
    image = tf.image.resize(image, resize)
    return image


def make_prediction(img_name, hazed_path, dehazed_path, model, file_type='png'):

    """ Plots dehazed estimated image against ground truth"""

    hazed_img = load_image(hazed_path + img_name, file_type=file_type)
    dehazed_img = load_image(dehazed_path + img_name, file_type=file_type)

    scaled_hazed = (hazed_img / 127.5) - 1
    scaled_hazed = tf.expand_dims(scaled_hazed, axis=0)
    generated_dehazed = model(scaled_hazed)
    generated_dehazed = tf.squeeze(generated_dehazed, axis=0)
    generated_dehazed = (generated_dehazed + 1) * 127.5

    plt.figure(figsize=(14, 14))
    plt.subplot(1, 3, 1)
    plt.title("hazed picture")
    plt.imshow(hazed_img.numpy().astype("uint8"))
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("dehazed reference")
    plt.imshow(dehazed_img.numpy().astype("uint8"))
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("generated picture")
    plt.imshow(generated_dehazed.numpy().astype("uint8"))
    plt.axis("off")

    plt.show()

parser = argparse.ArgumentParser()
parser.add_argument("hazed_path", help="Directory path for hazed images", type='str')
parser.add_argument("dehazed_path", help="Directory path for dehazed images", type='str')
parser.add_argument("n_files", help="number of examples to display", type=int)
parser.add_argument("ckpt_path", help="checkpoint path to save/load models", type='str')
parser.add_argument("ckpt_name", help="checkpoint name to load model from", type='str')
parser.add_argument("generator",
                    help="choose type of generator between large and tiny. large by default", nargs="?", default='large')
parser.add_argument("training_hardware",
                    help="specify hardware used for training. TPU by default", nargs="?", default='TPU')
parser.add_argument("strategy", help="Name of the hardware if needed. Mandatory when using GCP TPUs False by default.",
                    nargs="?", default=False)

args = parser.parse_args()

def main(args):

    if args.training_hardware == 'TPU':
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(args.strategy)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        print("All devices: ", tf.config.list_logical_devices('TPU'))
        strategy = tf.distribute.TPUStrategy(resolver)
    elif args.training_hardware == 'GPU':
        gpus = tf.config.list_logical_devices('GPU')
        strategy = tf.distribute.MirroredStrategy(gpus)
    else:
        strategy = None

    GAN = FUnIEGAN((256, 256, 3), 1, 1, ckpt_path=args.ckpt_path,
                   strategy=strategy, generator=args.generator)
    vgg_weights = GAN.vgg19.get_weights()
    GAN.vgg19.set_weights(vgg_weights)

    GAN.load_checkpoint(args.ckpt_name)

    filenames = tf.io.gfile.glob("gs://hazed/hazed/*png")
    for i in range(args.n_files):
        make_prediction(os.path.split(filenames[i])[1], GAN.g, file_type='png')



