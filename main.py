import torch
import torchsummary
from torch.optim import Adam
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

from dataloader import AnimeDataset, CelebADataset
from original_networks import Generator, Discriminator

import shutil
import os


def yield_infinite(dataloader):
    while True:
        for data in dataloader:
            yield(data)


def gradient_penalty(y, x, device):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).to(device)
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm-1)**2)


def classification_loss(logit, target):
    return torch.nn.functional.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)


def train(generator, discriminator, batch_size=32, num_batches=10000,
          device='cuda', log_interval=25, log_dir="./logs"):
    test_loader = yield_infinite(
            DataLoader(CelebADataset(amount=8), batch_size=8)
            )

    mixed_dataset = ConcatDataset(
            [AnimeDataset(amount=10000), CelebADataset(amount=10000)]
            )

    mixed_loader = yield_infinite(
            DataLoader(mixed_dataset, batch_size=batch_size,
                       shuffle=True, drop_last=True)
            )

    disc_optim = Adam(params=discriminator.parameters(),
                      lr=0.0002, betas=(0.5, 0.999))
    gen_optim = Adam(params=generator.parameters(),
                     lr=0.0002, betas=(0.5, 0.999))

    g_losses = []
    d_losses = []

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir)

    for index in range(num_batches):
        # compute loss on real images
        real_images, classes = next(mixed_loader)
        real_images, classes = real_images.to(device), classes.to(device)
        real_images = (real_images - 0.5) * 2
        predicted_source, predicted_classes = discriminator(real_images)
        d_loss_real = - torch.mean(predicted_source)
        d_loss_class = classification_loss(predicted_classes, classes)

        # compute loss on fake images
        target_classes = 1 - classes
        fake_images = generator(real_images, target_classes)
        predicted_source, _ = discriminator(fake_images.detach())
        d_loss_fake = torch.mean(predicted_source)

        # gradient penalty
        alpha = torch.rand(real_images.size(0), 1, 1, 1).to(device)
        x_hat = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(True)
        out_src, _ = discriminator(x_hat)
        d_loss_gp = gradient_penalty(out_src, x_hat, device) * 10

        # apply gradients
        d_loss = d_loss_real + d_loss_fake + d_loss_class + d_loss_gp
        discriminator.zero_grad()
        generator.zero_grad()
        d_loss.backward()
        disc_optim.step()

        if index > 0 and index % 5 == 0:
            # original to target
            generator.zero_grad()
            fake_images = generator(real_images, target_classes)
            predicted_source, predicted_classes = discriminator(fake_images)
            g_loss_fake = - torch.mean(predicted_source)
            g_loss_class = classification_loss(predicted_classes, target_classes)

            # target to original
            reconstruction = generator(fake_images, classes)
            g_loss_reconstruction = torch.mean(torch.abs(real_images - reconstruction)) * 10

            # apply gradients
            g_loss = g_loss_fake + g_loss_reconstruction + g_loss_class
            discriminator.zero_grad()
            generator.zero_grad()
            g_loss.backward()
            gen_optim.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            writer.add_scalar("d_loss", d_loss, global_step=index)
            writer.add_scalar("g_loss", g_loss, global_step=index)
            writer.flush()

        if index % log_interval == 0:
            print(f"loss gen: {torch.Tensor(g_losses)[-100:].mean()}, loss disc: {torch.Tensor(d_losses)[-100:].mean()}")
            with torch.no_grad():
                test_images = (next(test_loader)[0] - 0.5) * 2
                fake_images = generator(test_images.to(device), torch.ones(len(test_images), 1).to(device))
                reconstruction = generator(fake_images, torch.zeros(len(test_images), 1).to(device))
                collection = torch.cat([test_images, fake_images.detach().cpu(), reconstruction.detach().cpu()])
                collection = (collection * 0.5) + 0.5
                save_image(collection, "fake.png")


if __name__ == "__main__":
    generator = Generator().to('cuda')
    discriminator = Discriminator().to('cuda')

    # discriminator.summary()
    # generator.summary()
    torchsummary.summary(generator, (3, 64, 64))
    torchsummary.summary(discriminator, (3, 64, 64))

    train(generator, discriminator)
