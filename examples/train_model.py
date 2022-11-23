import eagerx_interbotix

# torch
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

# other
from typing import List
import numpy as np
from PIL import Image
from datetime import datetime
import h5py
import pathlib

NAME = "HER_force_torque_2022-10-13-1836"
STEPS = 1_600_000
MODEL_NAME = f"rl_model_{STEPS}_steps"
ROOT_DIR = pathlib.Path(eagerx_interbotix.__file__).parent.parent.resolve()
LOG_DIR = ROOT_DIR / "logs" / f"{NAME}"
GRAPH_FILE = f"graph.yaml"


class H5Dataset(Dataset):
    def __init__(
        self,
        file_path: str,
        transform: transforms = None,
        target_mean: List[float] = None,
        target_std: List[float] = None,
    ):
        self.file_path = file_path
        self.transform = transform

        self.target_mean = torch.Tensor(target_mean) if target_mean is not None else None
        self.target_std = torch.Tensor(target_std) if target_std is not None else None

        self.f = h5py.File(self.file_path, "r")
        self.targets = torch.tensor(
            np.hstack((self.f["box_pos"][:, :2], self.f["box_yaw"], self.f["goal_pos"][:, :2], self.f["goal_yaw"]))
        )
        self.length = len(self.targets)

    def __len__(self):
        assert self.length is not None
        return self.length

    def __getitem__(self, index: int):
        x = Image.fromarray(np.asarray(self.f["img"][index], dtype="uint8"))
        y = self.targets[index]

        if self.transform is not None:
            x = self.transform(x)

        if self.target_mean is not None and self.target_std is not None:
            y = (y - self.target_mean) / self.target_std

        return x, y.float()

    def close(self):
        self.f.close()


def loop_train(dataloader, model, loss_fn, optimizer):
    model.train()
    train_stats = {}
    losses = []
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.item()
        losses.append(loss)
    loss_mean = np.mean(losses)
    print(f"Train error (MSE): {loss_mean:>8f}")
    train_stats["loss"] = loss_mean

    return train_stats


def loop_test(dataloader, model, loss_fn):
    model.eval()
    test_stats = {}
    test_losses = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss = loss_fn(pred, y).item()
            test_losses.append(test_loss)

    test_loss_mean = np.mean(test_losses)
    print(f"Test error (MSE): {test_loss_mean:>8f}")
    test_stats["loss"] = test_loss_mean

    return test_stats


if __name__ == "__main__":
    # Set parameters
    dataset_size = 20000
    batch_size = 32
    epochs = 10
    split = 0.7
    seed = 1
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    log_dir = str(LOG_DIR / f"{date_time}" / f"{dataset_size}_{batch_size}_{epochs}_{split}_{seed}")
    writer = SummaryWriter(log_dir)
    file_path = str(LOG_DIR / f"dataset_{dataset_size}.hdf5")
    torch.manual_seed(seed)
    np.random.seed(seed)

    f = h5py.File(LOG_DIR / f"dataset_{dataset_size}.hdf5", "r")

    # Get mean and std for input and target for normalization
    image_mean = np.mean(np.reshape(f["img"], (-1, 3)), 0) / 255
    image_std = np.std(np.reshape(f["img"], (-1, 3)), 0) / 255
    target_mean = np.mean(np.hstack((f["box_pos"][:, :2], f["box_yaw"], f["goal_pos"][:, :2], f["goal_yaw"])), 0)
    target_std = np.std(np.hstack((f["box_pos"][:, :2], f["box_yaw"], f["goal_pos"][:, :2], f["goal_yaw"])), 0)

    # Set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Initialize transforms
    input_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=image_mean,
                std=image_std,
            ),
        ]
    )

    # Initialize Dataset and Dataloaders
    dataset = H5Dataset(
        file_path,
        transform=input_transforms,
        target_mean=target_mean,
        target_std=target_std,
    )
    n_train = int(len(dataset) * split)
    n_test = len(dataset) - n_train
    train_data, test_data = random_split(dataset, [n_train, n_test])
    train_dataloader = DataLoader(train_data, batch_size=batch_size, drop_last=True, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, drop_last=True, shuffle=True, pin_memory=True)

    # Load model
    model = torch.hub.load("pytorch/vision:v0.10.0", "shufflenet_v2_x1_0", pretrained=False)
    for param in model.parameters():
        param.requires_grad = True

    # Adjust last layer
    model.fc = nn.Linear(1024, 6)
    model.to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters())

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_stats = loop_train(train_dataloader, model, loss_fn, optimizer)
        test_stats = loop_test(test_dataloader, model, loss_fn)

        for k, v in train_stats.items():
            writer.add_scalar(f"train_{k}", v, epoch + 1)
        for k, v in test_stats.items():
            writer.add_scalar(f"test_{k}", v, epoch + 1)

    print("Done!")

    writer.close()
    dataset.close()

    # Additional information
    MODEL_PATH = f"{log_dir}.tar"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "seed": seed,
            "train_loss": train_stats["loss"],
            "test_loss": test_stats["loss"],
            "image_mean": image_mean,
            "image_std": image_std,
            "target_mean": target_mean,
            "target_std": target_std,
        },
        MODEL_PATH,
    )
