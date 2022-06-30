import numpy as np
import pandas as pd 
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, ToTensor, Compose
import torchvision.transforms as T


from src.utils.utils import get_data_path

class PadChestDataset(Dataset):
    """
    PyTorch Dataset for the PadChest dataset.
    """

    def __init__(self, df:pd.DataFrame, size=(128,128), max_value=255):
        """
        Initializes the dataset.
        """

        self.df = df
        self.data_path = get_data_path()

        varnames = list(self.df.columns)
        self.file_varnames = ["ImageDir", "ImageID"]
        self.conditions_varnames = [v for v in varnames if v.startswith("conditions_")]
        self.findings_varnames = [v for v in varnames if v.startswith("findings_")]
        self.localizations_varnames = [
            v for v in varnames if v.startswith("localizations_")
        ]
        self.varnames = (
            self.file_varnames + self.findings_varnames + self.localizations_varnames + self.conditions_varnames
        )

        # torchvision transform to divide by max value and resize if necessary
        self.image_transform = T.Compose(
            [
                Resize(size),
                ToTensor(),
                lambda x: x / max_value,
            ]
        )

        # print(f"Number of findings: {len(self.findings_varnames)}")
        # print(f"Number of localizations: {len(self.localizations_varnames)}")
        # print(f"Number of conditions: {len(self.conditions_varnames)}")

        self.df = self.df[self.varnames]

    def __len__(self):
        """Returns length of the dataset."""
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns the item at the given index. The item is a tuple of the form
        (image, findings, localizations), where findings and localizations are
        multi-hot encoded.
        """

        row = self.df.iloc[idx]
        image_path = (
            self.data_path
            / "processed"
            / "padchest"
            / "images"
            / "224"
            / row["ImageID"]
        )

        image_PIL = Image.open(image_path)

        #print(f"max image value {np.max(image_PIL)}")
        image = self.image_transform(image_PIL)

        datapoint = (
            image,
            np.asarray(row[self.conditions_varnames], dtype=bool),
            np.asarray(row[self.findings_varnames], dtype=bool),
            np.asarray(row[self.localizations_varnames], dtype=bool),
        )

        return datapoint


class ISICDataset(Dataset):
    """
    PyTorch Dataset for the ISIC dataset.
    """

    def __init__(self, df:pd.DataFrame, size=(128,128), max_value=255):
        """
        Initializes the dataset.
        """

        self.df = df
        self.data_path = get_data_path()

        varnames = list(self.df.columns)
        self.file_varnames = ["Image"]
        self.varnames = [
            v for v in varnames if v in ['MEL', 'NV', 'BCC', 'AK', 
                                        'BKL', 'DF', 'VASC', 'SCC', 'UNK']
                                        ]

        print(f"file varnames {self.file_varnames}, varnames {self.varnames}")

        # torchvision transform to divide by max value and resize if necessary
        self.image_transform = T.Compose(
            [
                Resize(size),
                ToTensor(),
                lambda x: x / max_value,
            ]
        )

        self.df = self.df[self.varnames]


    def __len__(self):
        """Returns length of the dataset."""
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns the item at the given index. The item is a tuple of the form
        (image, conditions), where findings and localizations are
        multi-hot encoded.
        """

        row = self.df.iloc[idx]
        image_path = (
            self.data_path
            / "processed"
            / "ICIC"
            / "images"
            / row["Image"]
        )

        image_PIL = Image.open(image_path)

        #print(f"max image value {np.max(image_PIL)}")
        image = self.image_transform(image_PIL)

        datapoint = (
            image,
            np.asarray(row[self.varnames], dtype=bool),
        )

        return datapoint