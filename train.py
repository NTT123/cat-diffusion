"""Train a diffusion model."""
import logging
import math
import pickle

import fire
import jax
import jax.numpy as jnp
import numpy as np
import opax
import pax
import tensorflow as tf
from PIL import Image

from model import GaussianDiffusion, UNet


def make_image_grid(images, padding=2):
    """Place images in a square grid."""
    num_images = images.shape[0]
    size = int(math.sqrt(num_images))
    assert size * size == num_images, "expecting a square grid"
    img = images[0]

    height = img.shape[0] * size + padding * (size + 1)
    width = img.shape[1] * size + padding * (size + 1)
    out = np.zeros((height, width, img.shape[-1]), dtype=img.dtype)
    for i in range(num_images):
        x_coord = i % size
        y_coord = i // size
        xstart = x_coord * (img.shape[0] + padding) + padding
        xend = xstart + img.shape[0]
        ystart = y_coord * (img.shape[1] + padding) + padding
        yend = ystart + img.shape[1]
        out[xstart:xend, ystart:yend, :] = images[i]
    return out


def train(
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    num_training_steps: int = 10_000,
    log_freq: int = 1000,
    image_size: int = 64,
    random_seed: int = 42,
    data_dir: str = "tfdata",
    hidden_dim: int = 64,
    output_dir: str = "output",
):
    """Train a diffusion model."""

    pax.seed_rng_key(random_seed)

    model = UNet(dim=hidden_dim, dim_mults=(1, 2, 4, 8))

    diffusion = GaussianDiffusion(
        model,
        image_size=image_size,
        timesteps=1000,
        loss_type="l1",  # L1 or L2
    )

    # load tensorflow dataset from data directory
    dataset = tf.data.Dataset.load(data_dir)
    # print data directory and dataset size
    logging.info("Data directory: %s", data_dir)
    logging.info("Dataset size: %d", len(dataset))

    dataloader = (
        dataset.repeat()
        .shuffle(batch_size * 100)
        .batch(batch_size)
        .take(num_training_steps)
        .prefetch(tf.data.AUTOTUNE)
    )

    def loss_fn(model, inputs):
        model, loss = pax.purecall(model, inputs)
        return loss, (loss, model)

    update_fn = pax.utils.build_update_fn(loss_fn)
    fast_update_fn = jax.jit(update_fn)

    optimizer = opax.adam(learning_rate)(diffusion.parameters())

    total_loss = 0.0
    for step, batch in dataloader.enumerate(start=1):
        batch = jax.tree_util.tree_map(lambda x: x.numpy(), batch)
        diffusion, optimizer, loss = fast_update_fn(diffusion, optimizer, batch)
        total_loss = total_loss + loss

        if step % log_freq == 0:
            loss = total_loss / log_freq
            total_loss = 0.0
            logging.info("[step %08d]  train loss %.3f", step, loss)

            imgs = jax.device_get(diffusion.eval().sample(16))
            imgs = ((imgs * 0.5 + 0.5) * 255).astype(jnp.uint8)
            imgs = make_image_grid(imgs)
            sample_image = Image.fromarray(imgs)
            sample_image.save(f"{output_dir}/sample_{step:08d}.png")

            # get model state dict
            model_state_dict = jax.device_get(diffusion.state_dict())
            # save model state dict to file
            file_name = f"{output_dir}/model_{step:08d}.pkl"
            with open(file_name, "wb") as model_ckpt_file:
                pickle.dump(model_state_dict, model_ckpt_file)
            logging.info("Model state dict saved to %s.", file_name)


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fire.Fire(train)
