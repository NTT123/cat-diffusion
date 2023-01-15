"""Train a diffusion model."""
import math
import pickle
from datetime import datetime
from pathlib import Path
from typing import Tuple

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


def create_tensorboard_writer(
    output_dir: str, run_name: str
) -> tf.summary.SummaryWriter:
    """Initialize tensorboard writer."""

    # add current time to run name
    run_name = f"{run_name}-{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}"
    # create tensorboard writer
    writer = tf.summary.create_file_writer(
        f"{output_dir}/logs/{run_name}", flush_millis=60_000 * 60
    )
    # return tensorboard writer
    return writer


def save_model(diffusion: GaussianDiffusion, output_dir: str, step: int):
    """Save model state dict to file."""
    # get model state dict
    model_state_dict = jax.device_get(diffusion.state_dict())
    # save model state dict to file
    file_name = f"{output_dir}/model_{step:08d}.pkl"
    with open(file_name, "wb") as model_ckpt_file:
        pickle.dump(model_state_dict, model_ckpt_file)
    print(f"Model state dict saved to {file_name}.")


def create_or_load_model(
    image_size: int, hidden_dim: int, output_dir: str
) -> Tuple[GaussianDiffusion, int]:
    """Create or load model from file."""
    # create unet model
    unet = UNet(dim=hidden_dim, dim_mults=(1, 2, 4, 8))
    # create diffusion model
    diffusion = GaussianDiffusion(
        unet,
        image_size=image_size,
        timesteps=1000,
        loss_type="l1",  # L1 or L2
    )
    last_training_step = 0

    # load model state dict from file if exists
    # list model files in the output directory
    # that matches the pattern model_*.pkl
    model_files = list(Path(output_dir).glob("model_*.pkl"))
    # sort model files by step
    model_files.sort(key=lambda x: int(x.stem.split("_")[-1]))
    # if model files exist, load the last one
    if model_files:
        model_file = model_files[-1]
        last_training_step = int(model_file.stem.split("_")[-1])
        with open(model_file, "rb") as model_ckpt_file:
            model_state_dict = pickle.load(model_ckpt_file)
            # load model state dict
            diffusion = diffusion.load_state_dict(model_state_dict)
            print(f"Model state dict loaded from {model_file}.")

    # return diffusion model and last step
    return diffusion, last_training_step


def train(
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    num_training_steps: int = 10_000,
    log_freq: int = 100,
    image_size: int = 64,
    random_seed: int = 42,
    data_dir: str = "tfdata",
    hidden_dim: int = 64,
    output_dir: str = "output",
    run_name: str = "cat",
):
    """Train a diffusion model."""

    pax.seed_rng_key(random_seed)
    writer = create_tensorboard_writer(output_dir, run_name)

    diffusion, last_training_step = create_or_load_model(
        image_size, hidden_dim, output_dir
    )

    # load tensorflow dataset from data directory
    dataset = tf.data.Dataset.load(data_dir)
    # print data directory and dataset size
    print(f"Data directory: {data_dir}")
    print(f"Dataset size: {len(dataset)}")

    dataloader = (
        dataset.repeat()
        .shuffle(len(dataset))
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
    for step, batch in dataloader.enumerate(start=last_training_step + 1):
        batch = jax.tree_util.tree_map(lambda x: x.numpy(), batch)
        diffusion, optimizer, loss = fast_update_fn(diffusion, optimizer, batch)
        total_loss = total_loss + loss

        if step % log_freq == 0:
            loss = total_loss / log_freq
            total_loss = 0.0
            print(f"[step {step:08d}]  train loss {loss:.3f}")
            # write to tensorboard
            with writer.as_default():
                tf.summary.scalar("Train/loss", loss, step=step)

        if step % 1000 == 0:
            imgs = jax.device_get(diffusion.eval().sample(16))
            imgs = ((imgs * 0.5 + 0.5) * 255).astype(jnp.uint8)
            imgs = make_image_grid(imgs)
            sample_image = Image.fromarray(imgs)
            sample_image.save(f"{output_dir}/sample_{step:08d}.png")
            # add image to tensorboard
            with writer.as_default():
                tf.summary.image("Train/sample", imgs[None], step=step)
                writer.flush()
            save_model(diffusion, output_dir, step)


if __name__ == "__main__":
    fire.Fire(train)
