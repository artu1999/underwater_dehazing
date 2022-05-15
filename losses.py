import tensorflow as tf

def MAE(y_true, y_pred):
    outputs = tf.abs(y_true - y_pred)
    return tf.reduce_mean(outputs, axis=list(range(1, len(y_true.shape))))

def MSE(y_true, y_pred):
    outputs = tf.square(y_true - y_pred)
    return tf.reduce_mean(outputs, axis=list(range(1, len(y_true.shape))))

def reduce_mean(per_sample_loss, batch_size):
  """ return the global mean of per-sample loss """
  return tf.reduce_sum(per_sample_loss) / batch_size