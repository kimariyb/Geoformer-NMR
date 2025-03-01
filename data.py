import os
import torch
import numpy as np

from dgllife.utils import RandomSplitter
from torch.utils.data import DataLoader
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning import LightningDataModule
from tqdm import tqdm
from utils.loader import CarbonDatasetBuilder, ChemicalShiftDataset
from utils.collator import GeoformerDataCollator

class DataModule(LightningDataModule):
    def __init__(self, hparams):
        super(DataModule, self).__init__()
        self.hparams.update(hparams.__dict__) if hasattr(
            hparams, "__dict__"
        ) else self.hparams.update(hparams)
        self._mean, self._std = self.hparams["mean"], self.hparams["std"]
        self._saved_dataloaders = dict()
        self.dataset = None

    def prepare_dataset(self):
        dataset_root = os.path.join(self.hparams["dataset_root"], self.hparams["dataset"])

        if self.hparams["dataset"] == "carbon":
            self.dataset = CarbonDatasetBuilder(root=dataset_root).build()  # type: ChemicalShiftDataset
        elif self.hparams["dataset"] == "hydrogen":
            raise NotImplementedError("Hydrogen dataset is not implemented yet.")

        self.train_dataset, self.val_dataset, self.test_dataset = RandomSplitter.train_val_test_split(
            dataset=self.dataset,
            frac_train=self.hparams["train_size"], 
            frac_val=self.hparams["val_size"], 
            frac_test=self.hparams["test_size"], 
            random_state=self.hparams["seed"]
        )

        print(
            f"train {len(self.train_dataset)}, val {len(self.val_dataset)}, test {len(self.test_dataset)}"
        )
    
        self._standardize()

        print(
            f"****** Standardized dataset with mean {self._mean:.4f} and std {self._std:.4f} ******"
        )

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, "train")

    def val_dataloader(self):
        return self._get_dataloader(self.val_dataset, "val")

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, "test")

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    def _get_dataloader(self, dataset, stage, store_dataloader=True):
        store_dataloader = store_dataloader and not self.hparams["reload"]
        if stage in self._saved_dataloaders and store_dataloader:
            return self._saved_dataloaders[stage]

        if stage == "train":
            batch_size = self.hparams["batch_size"]
            shuffle = True
        elif stage in ["val", "test"]:
            batch_size = self.hparams["inference_batch_size"]
            shuffle = False

        collator = GeoformerDataCollator(max_nodes=self.hparams["max_nodes"])

        dl = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.hparams["num_workers"],
            pin_memory=True,
            drop_last=False,
            collate_fn=collator,
        )

        if store_dataloader:
            self._saved_dataloaders[stage] = dl

        return dl

    @rank_zero_only
    def _standardize(self):
        train_data = tqdm(
            self._get_dataloader(
                self.train_dataset, "val", store_dataloader=False
            ),
            desc="computing mean and std",
        )

        ys = torch.cat([data[-2][data[-1]] for data in train_data])

        self._mean = ys.mean(dim=0)
        self._std = ys.std(dim=0)
    
