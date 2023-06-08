import tensorflow as tf
import numpy as np
import tensorflowjs as tfjs

from model import generate_model


def fetch_dataset():
    images = np.load('./data/pmdata_clean/data.npy')
    labels = np.load('./data/pmdata_clean/output.npy')
    print(images.shape, labels.shape)
    images = tf.constant(images, dtype=tf.float32) # X is a np.array
    labels = tf.constant(labels, dtype=tf.float32) # y is a np.array
    dataset = tf.data.Dataset.from_tensor_slices(images)

    return images, labels


def train(train_model, x, y):
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.05, decay_steps=10000, decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    # loss_fn = tf.keras.losses.MeanSquaredError()
    loss_fn = tf.keras.losses.Huber(delta=0.5, reduction="auto", name="huber_loss")
    filepath = "./weights/v006/save_{epoch:02d}.h5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filepath, monitor="loss", verbose=1, save_best_only=True, mode="min")
    train_model.compile(optimizer=optimizer,
                        loss=loss_fn,
                        metrics=['mse', 'accuracy'])
    train_model.fit(x, y, batch_size=32, epochs=190, callbacks=[checkpoint])
    return train_model


def save_model(train_model):
    train_model.save('./weights/v006.h5')
    tfjs.converters.save_keras_model(train_model, './weights/v006_js')


def main():
    train_model = generate_model()
    print("fetching")
    x, y = fetch_dataset()
    print("training")
    train_model = train(train_model, x, y)
    save_model(train_model)


if __name__ == "__main__":
    main()
