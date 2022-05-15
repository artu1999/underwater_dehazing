import tensorflow as tf
from time import time
import math
import pandas as pd
from data_utils import load_paired
from models import FUnIEGAN, train
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("n_epochs", help="number of epochs to train the model for.", type=int)
parser.add_argument("n_batch", help="size of the batch to be used for training.", type=int)
parser.add_argument("paired_path", help="Directory path for hazed-dehazed paired tfrecord file")
parser.add_argument("ckpt_path", help="checkpoint path to save/load models")
parser.add_argument("generator",
                    help="choose type of generator between large and tiny. large by default", nargs="?", default='large')
parser.add_argument("training_hardware",
                    help="specify hardware used for training. TPU by default", nargs="?", default='TPU')
parser.add_argument("strategy", help="Name of the hardware if needed. Mandatory when using GCP TPUs False by default.",
                    nargs="?", default=False)

args = parser.parse_args()


def main(args):
    n_epochs = args.n_epochs
    n_batch = args.n_batch

    train_ds = load_paired(args.paired_path, batch_size=n_batch, img_type='png')

    if args.training_hardware == 'TPU':
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(args.strategy)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        print("All devices: ", tf.config.list_logical_devices('TPU'))
        strategy = tf.distribute.TPUStrategy(resolver)
        train_ds = strategy.experimental_distribute_dataset(train_ds)
    elif args.training_hardware == 'GPU':
        gpus = tf.config.list_logical_devices('GPU')
        strategy = tf.distribute.MirroredStrategy(gpus)
        train_ds = strategy.experimental_distribute_dataset(train_ds)
    else:
        strategy = None

    n_files = len(tf.io.gfile.glob(args.paired_path + '*png'))
    train_size = n_files
    train_steps = math.floor(train_size/n_batch)

    GAN = FUnIEGAN((256, 256, 3), n_epochs, n_batch, ckpt_path=args.ckpt_path,
                   strategy=strategy, generator=args.generator)
    vgg_weights = GAN.vgg19.get_weights()
    GAN.vgg19.set_weights(vgg_weights)

    train_results = {}
    for epoch in range(n_epochs):
        print(f'Epoch {epoch + 1:03d}/{n_epochs:03d}')

        start = time()
        train(train_ds, GAN, train_steps, train_results)
        end = time()

        print(f'elapsed time for epoch {epoch}: {end - start}')

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            GAN.save_checkpoint()

    df_train_results = pd.DataFrame.from_dict(train_results)
    df_train_results.to_csv('output.csv')


if __name__ == '__main__':
    main(args)
