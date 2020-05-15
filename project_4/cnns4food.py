import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

IMG_WIDTH = 32
IMG_HEIGHT = 32


def load_image(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img = tf.keras.applications.resnet_v2.preprocess_input(img)
    return img


def load_triplets(triplet):
    ids = tf.strings.split(triplet)
    a = load_image(tf.io.read_file('food/' + ids[0] + '.jpg'))
    b = load_image(tf.io.read_file('food/' + ids[1] + '.jpg'))
    c = load_image(tf.io.read_file('food/' + ids[2] + '.jpg'))
    return tf.stack([a, b, c], axis=0), tf.strings.to_number(ids[3])


def create_model(freeze=True):
    resnet_weights_path = 'resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5'
    inputs = tf.keras.Input(shape=(3, IMG_HEIGHT, IMG_WIDTH, 3))
    encoder = tf.keras.applications.ResNet50V2(
        include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        weights=resnet_weights_path)
    encoder.trainable = not freeze
    # encoder = tf.keras.models.Sequential()
    # encoder.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    # encoder.add(tf.keras.layers.MaxPooling2D((2, 2)))
    # encoder.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    # encoder.add(tf.keras.layers.MaxPooling2D((2, 2)))
    # encoder.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    # encoder.add(tf.keras.layers.Flatten())
    # tf.keras.layers.GlobalAveragePooling2D()
    # encoder.add(tf.keras.layers.Dense(128, activation='relu'))
    global_average = tf.keras.layers.GlobalAveragePooling2D()
    image_a_features = global_average(encoder(inputs[:, 0, ...]))
    image_b_features = global_average(encoder(inputs[:, 1, ...]))
    image_c_features = global_average(encoder(inputs[:, 2, ...]))
    distance_ab = tf.math.abs(image_a_features - image_b_features)
    distance_ac = tf.math.abs(image_a_features - image_c_features)
    final_logits = tf.keras.layers.Dense(1)(distance_ac - distance_ab)
    triple_siamese = tf.keras.Model(inputs=inputs, outputs=final_logits)
    triple_siamese.summary()
    return triple_siamese


def make_training_labels():
    samples = 'train_triplets.txt'
    with open(samples, 'r') as file:
        triplets = [line for line in file.readlines()]
    train_samples, val_samples = train_test_split(triplets, test_size=0.2)
    with open('val_samples.txt', 'w') as file:
        for item in val_samples:
            flip = np.random.binomial(size=1, n=1, p=0.5)
            a, b, c = item.rstrip().split()
            sample = ' '.join([a, c, b, '0']) if flip else ' '.join([a, b, c, '1'])
            file.write(sample + '\n')
    with open('train_samples.txt', 'w') as file:
        for item in train_samples:
            flip = np.random.binomial(size=1, n=1, p=0.5)
            a, b, c = item.rstrip().split()
            sample = ' '.join([a, c, b, '0']) if flip else ' '.join([a, b, c, '1'])
            file.write(sample + '\n')
    return len(train_samples)


def show_batch(image_batch, label_batch):
    plt.figure(figsize=(3, 6))
    for i in range(4):
        a, b, c = image_batch[i, ...]
        ax = plt.subplot(4, 3, 3 * i + 1)
        plt.imshow(a)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_ylabel(int(label_batch[i, ...]))
        ax = plt.subplot(4, 3, 3 * i + 2)
        plt.imshow(b)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax = plt.subplot(4, 3, 3 * i + 3)
        plt.imshow(c)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
    plt.tight_layout()
    plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    num_train_samples = make_training_labels()
    train_dataset = tf.data.TextLineDataset(
        'train_samples.txt'
    )
    train_dataset = train_dataset.map(
        load_triplets,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_dataset = tf.data.TextLineDataset(
        'val_samples.txt'
    )
    val_dataset = val_dataset.map(
        load_triplets,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    model = create_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    train_dataset = train_dataset.shuffle(1024).batch(args.batch_size).repeat()
    val_dataset = val_dataset.batch(args.batch_size)
    image, label = next(iter(train_dataset))
    show_batch(image.numpy(), label.numpy())
    history = model.fit(
        train_dataset,
        steps_per_epoch=num_train_samples // args.batch_size,
        epochs=args.epochs,
        validation_data=val_dataset
    )


if __name__ == '__main__':
    main()
