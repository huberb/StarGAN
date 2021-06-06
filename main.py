import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.utils import save_image
from dataloader import AnimeDataset, CelebADataset
from networks import Generator, Discriminator


def yield_infinite(dataloader):
    while True:
        for data in dataloader:
            yield(data)


def train(generator, discriminator, batch_size=64,
          num_batches=10000, device='cuda', log_interval=25):
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

    loss_fn = torch.nn.BCELoss()

    disc_optim = Adam(params=discriminator.parameters(),
                      lr=0.0002, betas=(0.5, 0.999))
    gen_optim = Adam(params=generator.parameters(),
                     lr=0.0002, betas=(0.5, 0.999))

    real_label = torch.ones(batch_size, 1).to(device)
    fake_label = torch.zeros(batch_size, 1).to(device)

    g_losses = []
    d_losses = []

    for index in range(num_batches):
        discriminator.zero_grad()

        # compute loss on real images
        real_images, classes = next(mixed_loader)
        real_images, classes = real_images.to(device), classes.to(device)
        real_images = (real_images - 0.5) * 2
        predicted_source, predicted_classes = discriminator(real_images)
        d_loss_real = loss_fn(predicted_source, real_label)
        d_loss_class = loss_fn(predicted_classes, classes)

        # compute loss on fake images
        target_classes = 1 - classes
        fake_images = generator(real_images, target_classes)
        predicted_source, _ = discriminator(fake_images.detach())
        d_loss_fake = loss_fn(predicted_source, fake_label)

        # apply gradients
        d_loss = d_loss_real + d_loss_fake + d_loss_class
        d_loss.backward()
        disc_optim.step()

        # original to target
        generator.zero_grad()
        fake_images = generator(real_images, target_classes)
        predicted_source, predicted_classes = discriminator(fake_images)
        g_loss_fake = loss_fn(predicted_source, real_label)
        g_loss_class = loss_fn(predicted_classes, target_classes)

        # target to original
        reconstruction = generator(fake_images, classes)
        g_loss_reconstruction = torch.mean(torch.abs(real_images - reconstruction))

        # apply gradients
        g_loss = g_loss_fake + g_loss_reconstruction + g_loss_class
        g_loss.backward()
        gen_optim.step()

        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())

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
    generator = Generator()
    discriminator = Discriminator()

    # discriminator.summary()
    # generator.summary()

    train(generator, discriminator)
