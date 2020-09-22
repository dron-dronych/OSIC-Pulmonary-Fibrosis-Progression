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


def build_conv_net(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3))(inputs)
    x = layers.Conv2D(3, (3, 3))(x)
    # TODO check https://github.com/dron-dronych/OSIC-Pulmonary-Fibrosis-Progression/issues/3
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    preds = layers.Dense(1, activation='linear')(x)

    model = keras.Model(inputs=[inputs],
                        outputs=[preds]
                        )
    model.compile(optimizer='adam',
                  loss=keras.losses.MeanAbsoluteError(),
                  metrics=laplace_log_likelihood
                  )

    return model


def build_conv_net_vgg16base(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(
        64, (3, 3), activation='relu', padding='same', name='block1_conv1')(
        inputs)
    x = layers.Conv2D(
        64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(
        128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = layers.Conv2D(
        128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = layers.Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = layers.Conv2D(
        256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = layers.Dense(32, activation='relu')(x)

    preds = layers.Dense(1, activation='linear')(x)

    model = keras.Model(inputs=[inputs],
                        outputs=[preds]
                        )
    model.compile(optimizer='adam',
                  loss=keras.losses.MeanAbsoluteError(),
                  metrics=laplace_log_likelihood
                  )

    return model