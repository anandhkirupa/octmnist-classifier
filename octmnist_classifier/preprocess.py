import numpy as np
import torch
import medmnist
from medmnist import OCTMNIST, INFO
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_octmnist(flatten=False, data_flag='octmnist'):
    info = INFO[data_flag]
    train = OCTMNIST(split='train', download=True)
    test = OCTMNIST(split='test', download=True)
    val = OCTMNIST(split='val', download=True)

    # Combine all splits
    images = np.concatenate([train.imgs, test.imgs, val.imgs], axis=0)
    labels = np.concatenate([train.labels, test.labels, val.labels], axis=0).squeeze()

    if flatten:
        images = images.reshape(images.shape[0], -1)
    else:
        images = images[:, np.newaxis, :, :]  # Add channel dimension

    images = images.astype(np.float32) / 255.0
    return images, labels

def oversample_smote(X, y, plot=False):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    if plot:
        _plot_distribution(y_resampled, "SMOTE Oversampling")
    return X_resampled, y_resampled


def oversample_smote_tomek(X, y, plot=False):
    smt = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smt.fit_resample(X, y)
    if plot:
        _plot_distribution(y_resampled, "SMOTE + Tomek Oversampling")
    return X_resampled, y_resampled


def undersample_majority(X, y, plot=False):
    undersampler = RandomUnderSampler(sampling_strategy='majority')
    X_resampled, y_resampled = undersampler.fit_resample(X, y)
    if plot:
        _plot_distribution(y_resampled, "UnderSampling Majority Class")
    return X_resampled, y_resampled

def balance_data(X, y, strategy='smote_tomek', plot=False):
    X_flat = X.reshape(len(X), -1)

    if strategy == 'smote':
        X_res, y_res = oversample_smote(X_flat, y, plot=plot)
    elif strategy == 'smote_tomek':
        X_res, y_res = oversample_smote_tomek(X_flat, y, plot=plot)
    elif strategy == 'undersample':
        X_res, y_res = undersample_majority(X_flat, y, plot=plot)
    else:
        raise ValueError(f"Unsupported sampling strategy: {strategy}")

    return X_res.reshape(-1, 1, 28, 28), y_res

def prepare_dataloaders(X, y, test_size=0.2, val_size=0.1, batch_size=64):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(test_size + val_size), random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size/(test_size + val_size), random_state=42)

    def to_loader(X, y, shuffle):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

    return (
        to_loader(X_train, y_train, shuffle=True),
        to_loader(X_val, y_val, shuffle=False),
        to_loader(X_test, y_test, shuffle=False)
    )

def generate_kfold_loaders(X, y, batch_size=64, k_folds=5):
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True
        )

        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True
        )

        yield fold, train_loader, val_loader

def _plot_distribution(y, title):
    unique_labels, counts = np.unique(y, return_counts=True)
    sns.barplot(x=unique_labels, y=counts)
    plt.xlabel("Class Labels")
    plt.ylabel("Number of Samples")
    plt.title(title)
    plt.tight_layout()
    plt.show()


__all__ = [
    "load_octmnist",
    "oversample_smote",
    "oversample_smote_tomek",
    "undersample_majority",
    "balance_data",
    "prepare_dataloaders",
    "generate_kfold_loaders"
]