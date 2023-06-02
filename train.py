import tensorflow as tf
import numpy as np

from model import generate_model


def fetch_dataset():
    images = np.load('./data/pmdata_clean/data.npy')
    labels = np.load('./data/pmdata_clean/output.npy')
    print(images.shape, labels.shape)
    images = tf.constant(images, dtype=tf.float32) # X is a np.array
    labels = tf.constant(labels, dtype=tf.float32) # y is a np.array
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    return dataset


def train(train_model, dataset):
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.05, decay_steps=10000, decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    loss_fn = tf.keras.losses.MeanSquaredError()
    # loss_fn = tf.keras.losses.Huber(delta=0.5, reduction="auto", name="huber_loss")
    train_model.compile(optimizer=optimizer,
                        loss=loss_fn,
                        metrics=['accuracy'])
    train_model.fit(dataset, batch_size=32, epochs=2000)
    return train_model


def save_model(train_model):
    train_model.save('./weights/v001.h5')


def main():
    train_model = generate_model()
    print("fetching")
    dataset = fetch_dataset()
    print("training")
    train_model = train(train_model, dataset)
    save_model(train_model)


if __name__ == "__main__":
    main()
