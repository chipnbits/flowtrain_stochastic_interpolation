""" Practice Datasets from mnist and cifar """

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10, FashionMNIST
from torchvision.transforms import Compose, Normalize, Pad, ToTensor


class Gaussian2d(Dataset):
    """A Dataset wrapper for a 2D Gaussian distribution."""

    def __init__(self, mean, eigenvalues, principal_axis, size, device="cpu"):
        self.mean = mean
        self.eigenvalues = eigenvalues
        self.principal_axis = principal_axis
        self.sigma = self.calculate_sigma()
        self.size = size
        self.device = device

    def calculate_sigma(self):
        eigenvals = torch.sort(self.eigenvalues, descending=True)[0]
        principal_axis = self.principal_axis / torch.norm(self.principal_axis)

        # Make covariance matrix
        D = torch.diag(eigenvals)
        orthogonal_complement = torch.tensor([-principal_axis[1], principal_axis[0]])
        Q = torch.stack([principal_axis, orthogonal_complement])
        cov = Q.T @ D @ Q

        # Get the factorization for sigma
        sigma = torch.linalg.cholesky(cov)
        return sigma

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return (torch.randn(2) @ self.sigma + self.mean).to(self.device)

    def sample(self, n):
        return torch.stack([self.__getitem__(0) for _ in range(n)]).to(self.device)


class GaussianMixed(Dataset):
    """Data set for a mixture of two 2D Gaussians."""

    def __init__(self, size, device="cpu"):
        self.size = size
        self.device = device
        self.gauss0 = Gaussian2d(
            torch.tensor([4.0, 4.0]),
            torch.tensor([0.2, 0.02]),
            torch.tensor([0, 1.0]),
            1,
            self.device,
        )
        self.gauss1 = Gaussian2d(
            torch.tensor([-1.0, 4.0]),
            torch.tensor([0.2, 0.02]),
            torch.tensor([0, 1.0]),
            1,
            self.device,
        )

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        p = torch.rand(1)
        if p < 0.4:
            return self.gauss0[0]
        else:
            return self.gauss1[0]

    def sample(self, n):
        p = torch.rand(n)
        return torch.stack([self.gauss0[0] if p_ < 0.4 else self.gauss1[0] for p_ in p])


class OnlyImagesFashionMNIST(Dataset):
    def __init__(self, root, train=True, download=True, transform=None):
        # Initialize the FashionMNIST dataset
        self.dataset = FashionMNIST(
            root=root, train=train, download=download, transform=transform
        )

    def __len__(self):
        # Return the length of the dataset
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the item, but only return the image
        image, _ = self.dataset[idx]
        return image


def get_fashion_mnist(batch_size, device):
    transform = Compose([ToTensor(), Pad(2), Normalize((0.5,), (0.5,))])

    # Use the custom dataset class that returns only images
    dataset = OnlyImagesFashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        generator=torch.Generator(device=device),
    )
    return data_loader


def get_cifar10(batch_size, device):
    transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10(root="./data", download=True, train=True, transform=transform)
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        generator=torch.Generator(device=device),
    )
    return trainloader
