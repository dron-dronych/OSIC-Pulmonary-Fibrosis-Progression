import tensorflow as tf


def laplace_log_likelihood(y_true, y_pred):
    """
    a modified version of the Laplace Log Likelihood
    the metric is designed to reflect both the accuracy and certainty of each prediction
    :param y_true: a tensor label
    :param y_pred: a tuple of tensors (predicted_value, probability)
    :return: a modified Laplace Log Likelihood tensor
    """
    fvc_pred, prob = y_pred[0], y_pred[1]
    sigma = prob
    sigma_clipped = tf.math.maximum(sigma, 70)
    delta = tf.math.minimum(tf.math.abs(y_true - fvc_pred), 1000)
    delta = tf.cast(delta, tf.float32)

    metric = -(tf.math.sqrt(2.0) * delta / sigma_clipped) - tf.math.log(tf.math.sqrt(2.0) * sigma_clipped)

    return metric