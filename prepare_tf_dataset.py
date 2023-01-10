"""
Prepare a tensorflow dataset for training.

Usage:
    python prepare_tf_dataset.py \
        --image-dir=/path/to/image/directory \
        --output-dir=/path/to/output/tf/dataset \
        --image-size=64
"""

from pathlib import Path

import fire
import tensorflow as tf


def create_image_dataset(
    image_dir: str, output_dir: str, image_size: int
) -> tf.data.Dataset:
    """Load all images from the image directory and
    create a tensorflow dataset in the output directory."""

    # get all image paths in the image directory
    image_dir = Path(image_dir)
    image_paths = list(image_dir.glob("**/*.png"))
    # map each image path to a string
    image_paths = map(str, image_paths)
    # to list
    image_paths = list(image_paths)
    # create a tensorflow dataset from the image paths
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    # load the images from the image paths
    dataset = dataset.map(tf.io.read_file)
    # decode the images
    dataset = dataset.map(lambda x: tf.image.decode_png(x, channels=3))
    # convert the images to float32
    dataset = dataset.map(lambda x: tf.image.convert_image_dtype(x, tf.float32))
    # resize the images to image_size x image_size
    dataset = dataset.map(lambda x: tf.image.resize(x, [image_size, image_size]))
    # normalize the images to [-1, 1]
    dataset = dataset.map(lambda x: (x - 0.5) * 2)
    # export the dataset to the output directory
    tf.data.Dataset.save(dataset, output_dir)
    # print image directory and output directory and dataset size
    print("Image directory:", image_dir)
    print("Output directory:", output_dir)
    print("Dataset size:", len(image_paths))
    # return the dataset
    return dataset


def main(image_dir: str = "data", output_dir: str = "tfdata", image_size: int = 64):
    """Prepare a tensorflow dataset for training."""

    create_image_dataset(image_dir, output_dir, image_size)


if __name__ == "__main__":
    fire.Fire(main)
