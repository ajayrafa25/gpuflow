"""CIFAR-10 classification with ResNet18 using PyTorch Lightning."""
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18


class CIFAR10ResNet(pl.LightningModule):
    def __init__(self, lr: float = 1e-3, num_classes: int = 10):
        super().__init__()
        self.save_hyperparameters()
        self.model = resnet18(pretrained=False)
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def main():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    full_train = torchvision.datasets.CIFAR10(
        root="/workspace/data", train=True, download=True, transform=transform_train
    )
    val_set = torchvision.datasets.CIFAR10(
        root="/workspace/data", train=False, download=True, transform=transform_val
    )
    train_set, _ = random_split(full_train, [45000, 5000])

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    model = CIFAR10ResNet(lr=1e-3)

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed",
        log_every_n_steps=50,
    )
    trainer.fit(model, train_loader, val_loader)
    print(f"Best val_acc: {trainer.callback_metrics.get('val_acc', 'N/A')}")


if __name__ == "__main__":
    main()
