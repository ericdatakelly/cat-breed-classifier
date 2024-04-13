import copy
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


class CatDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        img_dir: str,
        transform=None,
        extension: str = "jpg",
    ):
        """
        Args:
            dataframe (pd.DataFrame): Dataframe containing image file names and labels.
            img_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            extension (str, optional): Image file extension. Defaults to 'jpg'.
        """
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.extension = extension

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_name = Path(self.img_dir, f"{self.dataframe.iloc[idx, 0]}.{self.extension}")
        image = Image.open(img_name).convert("RGB")
        label = self.dataframe.iloc[idx, 3] - 1

        if self.transform:
            image = self.transform(image)

        return image, label


def load_split_data(file_path: str) -> pd.DataFrame:
    """Load data from a given split file.

    Args:
        file_path (str): Path to the split file.

    Returns:
        pd.DataFrame: Loaded data.
    """
    data = pd.read_csv(
        file_path,
        delim_whitespace=True,
        names=["file_name", "class_id", "species", "breed_id", "breed_name"],
    )
    # Filter out only cat images
    cat_data = data[data["species"] == 1]
    return cat_data


def train_model(
    model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer, num_epochs: int
) -> Tuple[nn.Module, List[float]]:
    """Train the model.

    Args:
        model (nn.Module): Model to be trained.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer for updating model parameters.
        num_epochs (int): Number of training epochs.

    Returns:
        Tuple[nn.Module, List[float]]: Trained model and list of loss values.
    """
    best_acc = 0.0
    loss_plot = []
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Backward + optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        loss_plot.append(epoch_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Deep copy the model if it has the best accuracy so far
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, loss_plot


def make_dataloaders(
    train_split_path: str, val_split_path: str, img_dir: str, batch_size: int
) -> Tuple[DataLoader, DataLoader, pd.DataFrame, pd.DataFrame]:
    """Create data loaders for training and validation data.

    Args:
        train_split_path (str): Path to the training split file.
        val_split_path (str): Path to the validation split file.
        img_dir (str): Directory containing the images.
        batch_size (int): Batch size for data loaders.

    Returns:
        Tuple[DataLoader, DataLoader, pd.DataFrame, pd.DataFrame]: Training data loader, validation data loader, training data, validation data.
    """
    # Load training and validation data
    train_df = load_split_data(train_split_path)
    val_df = load_split_data(val_split_path)
    train_df["breed_name"] = train_df["file_name"].apply(
        lambda x: " ".join(x.split("_")[:-1])
    )
    val_df["breed_name"] = val_df["file_name"].apply(
        lambda x: " ".join(x.split("_")[:-1])
    )

    # Define transformations
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.RandomAffine(
                    degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    # Create data loaders
    train_dataset = CatDataset(train_df, img_dir, transform=data_transforms["train"])
    val_dataset = CatDataset(val_df, img_dir, transform=data_transforms["val"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_df, val_df


def prep_model(
    train_df: pd.DataFrame, lr: float
) -> Tuple[nn.Module, nn.Module, optim.Optimizer]:
    """Prepare the model for training.

    Args:
        train_df (pd.DataFrame): Training data.
        lr (float): Learning rate.

    Returns:
        Tuple[nn.Module, nn.Module, optim.Optimizer]: Model, loss function, and optimizer.
    """
    model = models.resnet50(pretrained=True)

    # Freeze all layers in the network
    for param in model.parameters():
        param.requires_grad = False

    # Replace the last fully connected layer
    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(train_df["breed_id"].unique()))

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)

    return model, criterion, optimizer


batch_size = 32
num_epochs = 300
lr = 0.0001

# Paths to the data
train_split_path = "../data/cat_dog_breeds_oxford/annotations/trainval.txt"
val_split_path = "../data/cat_dog_breeds_oxford/annotations/test.txt"
img_dir = "../data/cat_dog_breeds_oxford/images"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, train_df, val_df = make_dataloaders(
    train_split_path, val_split_path, img_dir, batch_size
)

# Assuming train_df includes all possible breeds and their correct names
breed_mapping = train_df.groupby("breed_id")["breed_name"].first().to_dict()

# Prepare model
model, criterion, optimizer = prep_model(train_df, lr)

# Train model
model_ft, loss_plot = train_model(model, criterion, optimizer, num_epochs=num_epochs)

# Path where the model will be saved
model_name = (
    f"cat_breed_classifier_{batch_size}_{num_epochs:03d}_{round(loss_plot[-1], 3):.3f}"
)
model_save_path = Path(f"{model_name}.pth").resolve()

# Save the model
torch.save(model_ft.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Show loss plot
plt.plot(loss_plot)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig(f"{model_name}_loss.png")
