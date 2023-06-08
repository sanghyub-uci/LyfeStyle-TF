import tensorflow as tf


def generate_model():
    model = tf.keras.models.Sequential([
        # tf.keras.layers.Flatten(),
        tf.keras.layers.Reshape((1,6)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.7),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    return model
