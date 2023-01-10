"""Testcases for prepare_tf_dataset.py."""

import tensorflow as tf

from prepare_tf_dataset import create_image_dataset


def test_create_image_dataset() -> None:
    """Test the create_image_dataset function."""

    image_dir = "assets/images"
    output_dir = "/tmp/tfdata"
    image_size = 64
    dataset = create_image_dataset(image_dir, output_dir, image_size)
    # get the first image from the dataset
    image = next(iter(dataset))
    # check the image shape
    assert image.shape == (image_size, image_size, 3)
    # check the image dtype
    assert image.dtype == tf.float32
    # check the image values
    assert tf.reduce_min(image) >= -1.0
    assert tf.reduce_max(image) <= 1.0
