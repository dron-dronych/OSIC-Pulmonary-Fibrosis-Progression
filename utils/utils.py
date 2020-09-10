import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


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


def build_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    preds = layers.Dense(1, activation='linear')(x)

    # this adds a confidence (constant = 100) to each prediction
    @tf.function
    def constant(tensor):
        batch_size = tf.shape(tensor)[0]
        constant_ = tf.constant(100)
        return tf.broadcast_to(constant_, shape=(batch_size, 1))

    confidence = keras.layers.Lambda(constant)(inputs)

    model = keras.Model(inputs=[inputs],
                        outputs=[preds, confidence]
                        )

    model.compile(optimizer='adam',
                  loss=keras.losses.MeanAbsoluteError(),
                  metrics=laplace_log_likelihood
                  )

    return model