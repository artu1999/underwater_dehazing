import tensorflow as tf
import os


def load_paired(tfrecord, batch_size=64, img_type='png'):
    AUTOTUNE = tf.data.AUTOTUNE
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    dataset = tf.data.TFRecordDataset(tfrecord, num_parallel_reads=AUTOTUNE)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(lambda x: load_img(x, img_type=img_type)).shuffle(256).batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def load_img(example, img_type='png'):
    feature_description = {
        "imageA":tf.io.FixedLenFeature([], tf.string),
        "imageB":tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, feature_description)
    if img_type == 'png':
      imageA = tf.io.decode_png(example["imageA"], channels=3)
      imageB = tf.io.decode_png(example["imageB"], channels=3)
    else:
      imageA = tf.io.decode_jpeg(example["imageA"], channels=3)
      imageB = tf.io.decode_jpeg(example["imageB"], channels=3)
    imageA = tf.image.resize(imageA, (256, 256))
    imageB = tf.image.resize(imageB, (256, 256))
    imageA = (imageA/127.5)-1.0
    imageB = (imageB/127.5)-1.0
    return imageA, imageB


def write_new_tf(tfrec_name, datasetA_path, datasetB_path, img_type = 'png'):
    if img_type == 'png':
      filenames = tf.io.gfile.glob(datasetA_path + '*png')
    else:
      filenames = [*tf.io.gfile.glob(datasetA_path + '*JPEG'), *tf.io.gfile.glob(datasetA_path + '*jpg')]
    len(filenames)
    with tf.io.TFRecordWriter(tfrec_name) as writer:
      for path in filenames:
        filename = os.path.split(path)[1]
        if img_type == 'png':
          imageA = tf.io.decode_png(tf.io.read_file(datasetA_path + filename))
          imageB = tf.io.decode_png(tf.io.read_file(datasetB_path + filename))
          example = create_example(imageA, imageB, img_type='png')
        else:
          imageA = tf.io.decode_jpeg(tf.io.read_file(datasetA_path + filename))
          imageB = tf.io.decode_jpeg(tf.io.read_file(datasetB_path + filename))
          example = create_example(imageA, imageB, img_type='jpeg')
        writer.write(example)
    return


def create_example(imageA, imageB, img_type='png'):
    if img_type == 'png':
        feature = {
            "imageA": image_feature(imageA, img_type='png'),
            "imageB": image_feature(imageB, img_type='png')

        }
    else:
        feature = {
            "imageA": image_feature(imageA, img_type='jpeg'),
            "imageB": image_feature(imageB, img_type='jpeg')

        }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def image_feature(value, img_type='png'):
    """Returns a bytes_list from a string / byte."""
    if img_type == 'png':
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.encode_png(value).numpy()])
        )
    else:
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
        )