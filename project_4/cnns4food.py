import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

IMG_WIDTH = 96
IMG_HEIGHT = 96


def load_image(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    img = img / 127.5 - 1
    img = tf.image.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    return img


def load_triplets(triplet):
    ids = tf.strings.split(triplet)
    anchor = load_image(tf.io.read_file('food/' + ids[0] + '.jpg'))
    truthy = load_image(tf.io.read_file('food/' + ids[1] + '.jpg'))
    falsy = load_image(tf.io.read_file('food/' + ids[2] + '.jpg'))
    return tf.stack([anchor, truthy, falsy], axis=0), 1


def create_model(freeze=True):
    # mobilenet_weights_path = 'resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5'
    inputs = tf.keras.Input(shape=(3, IMG_HEIGHT, IMG_WIDTH, 3))
    encoder = tf.keras.applications.MobileNetV2(
        include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    encoder.trainable = not freeze
    decoder = tf.keras.Sequential([
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Lambda(
            lambda t: tf.math.l2_normalize(t, axis=1))
    ])
    anchor, truthy, falsy = inputs[:, 0, ...], inputs[:, 1, ...], inputs[:, 2, ...]
    anchor_features = decoder(encoder(anchor))
    truthy_features = decoder(encoder(truthy))
    falsy_features = decoder(encoder(falsy))
    embeddings = tf.stack([anchor_features, truthy_features, falsy_features], axis=-1)
    triple_siamese = tf.keras.Model(inputs=inputs, outputs=embeddings)
    triple_siamese.summary()
    return triple_siamese


def create_inference_model(model):
    embeddings = model.output
    anchor, truthy, falsy = embeddings[..., 0], embeddings[..., 1], embeddings[..., 2]
    distance_truthy = tf.reduce_sum(tf.square(anchor - truthy), 1)
    distance_falsy = tf.reduce_sum(tf.square(anchor - falsy), 1)
    predictions = tf.greater_equal(distance_falsy, distance_truthy)
    return tf.keras.Model(inputs=model.inputs, outputs=predictions)


def make_training_labels():
    samples = 'train_triplets.txt'
    with open(samples, 'r') as file:
        triplets = [line for line in file.readlines()]
    train_samples, val_samples = train_test_split(triplets, test_size=0.2)
    with open('val_samples.txt', 'w') as file:
        for item in val_samples:
            file.write(item)
    with open('train_samples.txt', 'w') as file:
        for item in train_samples:
            file.write(item)
    return len(train_samples)


def make_dataset(dataset_filename):
    dataset = tf.data.TextLineDataset(
        dataset_filename
    )
    dataset = dataset.map(
        load_triplets,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset


def triplet_loss(_, embeddings):
    anchor, truthy, falsy = embeddings[..., 0], embeddings[..., 1], embeddings[..., 2]
    distance_truthy = tf.reduce_sum(tf.square(anchor - truthy), 1)
    distance_falsy = tf.reduce_sum(tf.square(anchor - falsy), 1)
    return tf.reduce_mean(tf.math.softplus(distance_truthy - distance_falsy))


def accuracy(_, embeddings):
    anchor, truthy, falsy = embeddings[..., 0], embeddings[..., 1], embeddings[..., 2]
    distance_truthy = tf.reduce_sum(tf.square(anchor - truthy), 1)
    distance_falsy = tf.reduce_sum(tf.square(anchor - falsy), 1)
    return tf.reduce_mean(
        tf.cast(tf.greater_equal(distance_falsy, distance_truthy), tf.float32))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    num_train_samples = make_training_labels()
    train_dataset = make_dataset('train_samples.txt')
    val_dataset = make_dataset('val_samples.txt')
    model = create_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=triplet_loss,
                  metrics=[accuracy])
    train_dataset = train_dataset.shuffle(1024).batch(args.batch_size).repeat()
    val_dataset = val_dataset.batch(args.batch_size)
    history = model.fit(
        train_dataset,
        steps_per_epoch=num_train_samples // args.batch_size,
        epochs=args.epochs,
        validation_data=val_dataset,
        validation_steps=10
    )
    test_dataset = make_dataset('test_triplets.txt')
    inference_model = create_inference_model(model)
    predictions = inference_model.predict(test_dataset)


if __name__ == '__main__':
    main()
