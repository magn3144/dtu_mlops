"""Adapted from https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb.

A simple implementation of Gaussian MLP Encoder and Decoder trained on MNIST
"""
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from model import Decoder, Encoder, Model
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import hydra
import logging


def loss_function(x, x_hat, mean, log_var, hyp1, hyp2, hyp3):
    """Elbo loss function."""
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    kld = -hyp1 * torch.sum(hyp2 + log_var - mean.pow(hyp3) - log_var.exp())
    return reproduction_loss + kld


@hydra.main(config_name="config.yaml")
def main(cfg):
    log = logging.getLogger(__name__)

    # Load model Hyperparameters
    dataset_path = cfg.hyperparameters.dataset_path
    cuda = cfg.hyperparameters.cuda
    DEVICE = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
    batch_size = cfg.hyperparameters.batch_size
    x_dim = cfg.hyperparameters.x_dim
    hidden_dim = cfg.hyperparameters.hidden_dim
    latent_dim = cfg.hyperparameters.latent_dim
    lr = cfg.hyperparameters.lr
    epochs = cfg.hyperparameters.epochs
    hyp1 = cfg.hyperparameters.hyp1
    hyp2 = cfg.hyperparameters.hyp2
    hyp3 = cfg.hyperparameters.hyp3
    seed = cfg.hyperparameters.seed

    torch.manual_seed(seed)

    # Data loading
    mnist_transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = MNIST(dataset_path, transform=mnist_transform, train=True, download=True)
    test_dataset = MNIST(dataset_path, transform=mnist_transform, train=False, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)

    model = Model(encoder=encoder, decoder=decoder).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=lr)

    # Training
    log.info("Start training VAE...")
    model.train()
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            if batch_idx % 100 == 0:
                log.info(batch_idx)
            x = x.view(batch_size, x_dim)
            x = x.to(DEVICE)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var, hyp1, hyp2, hyp3)

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()
        log.info(f"Epoch {epoch+1} complete!,  Average Loss: {overall_loss / (batch_idx*batch_size)}")
    log.info("Finish!!")

    # save weights
    torch.save(model, f"{os.getcwd()}/trained_model.pt")

    # Generate reconstructions
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader):
            if batch_idx % 100 == 0:
                log.info(batch_idx)
            x = x.view(batch_size, x_dim)
            x = x.to(DEVICE)
            x_hat, _, _ = model(x)
            break

    save_image(x.view(batch_size, 1, 28, 28), "orig_data.png")
    save_image(x_hat.view(batch_size, 1, 28, 28), "reconstructions.png")

    # Generate samples
    with torch.no_grad():
        noise = torch.randn(batch_size, latent_dim).to(DEVICE)
        generated_images = decoder(noise)

    save_image(generated_images.view(batch_size, 1, 28, 28), "generated_sample.png")


if __name__ == "__main__":
    main()