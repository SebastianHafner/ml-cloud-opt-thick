import torch
from pathlib import Path
from abc import abstractmethod
import numpy as np
from omegaconf import DictConfig


class AbstractDataset(torch.utils.data.Dataset):

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.dataset_path = Path(cfg.paths.base_path_data)

        # Unpacking cfg
        self.property_column_mapping = cfg.dataloader.property_column_mapping
        self.input_keys = cfg.dataloader.inputs
        self.skip_band1 = cfg.dataloader.skip_band1
        self.skip_band10 = cfg.dataloader.skip_band10

        self.regressor = cfg.dataloader.regressor
        self.normalize_cot = cfg.dataloader.normalize_cot

        self.uniform_dist_no_cloud_thin_cloud_reg_cloud = cfg.dataloader.uniform_dist_no_cloud_thin_cloud_reg_cloud
        self.threshold_thickness_is_thin_cloud = cfg.dataloader.threshold_thickness_is_thin_cloud
        self.threshold_thickness_is_cloud = cfg.dataloader.threshold_thickness_is_cloud

    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


class Dataset(AbstractDataset):

    def __init__(self, cfg: DictConfig, run_type: str, no_augmentations: bool = False):
        super().__init__(cfg)

        self.run_type = run_type
        self.no_augmentations = no_augmentations

        self.input_noise = cfg.dataloader.input_noise_train if self.run_type == 'train' else cfg.dataloader.input_noise_val

        # Read data
        self.dataset = np.load(self.dataset_path / f'{self.run_type}set_smhi.npy')

        # Separate input and regression variable
        # Collect indices for selected inputs
        self.dataset_indices = [i for inp in self.input_keys for i in self.property_column_mapping[inp]]
        # If spectral bands are in inputs, check whether any bands (1 and 10) are skipped
        if 'spec_bands' in self.input_keys and (self.skip_band1 or self.skip_band10):
            self.dataset_indices = [i for i in self.dataset_indices if not (i == 1 and self.skip_band1) or
                                    (i == 10 and self.skip_band10)]

        self.inputs = self.dataset[:, self.dataset_indices]
        self.input_dim = self.inputs.shape[1]

        self.gts = np.squeeze(self.dataset[:, self.property_column_mapping[self.regressor]])
        self.gts_binary = np.squeeze(self.dataset[:, self.property_column_mapping['type']]) > 0

        # Create copies of original entities (some modifications, e.g. adding
        # noise, is done after each epoch, and the starting point per modification
        # should always be the original data)
        self.inputs_orig = self.inputs.copy()
        self.gts_orig = self.gts.copy()
        self.gts_binary_orig = self.gts_binary.copy()

        self.means_input, self.stds_input = np.zeros(self.dataset.shape[1]), np.ones(self.dataset.shape[1])
        self.means_input[self.property_column_mapping['spec_bands']] = self.cfg.dataloader.spec_bands_stats.means
        self.stds_input[self.property_column_mapping['spec_bands']] = self.cfg.dataloader.spec_bands_stats.stds
        self.means_input = self.means_input[self.dataset_indices]
        self.stds_input = self.stds_input[self.dataset_indices]

        # Subsampling dataset to balance distribution of no-cloud, thin-cloud, opaque-cloud
        if self.uniform_dist_no_cloud_thin_cloud_reg_cloud and not self.no_augmentations:
            all_0_idxs = np.squeeze(np.nonzero(self.gts_orig < self.threshold_thickness_is_thin_cloud))
            all_thin_idxs = np.squeeze(np.nonzero(np.logical_and(self.gts >= self.threshold_thickness_is_thin_cloud,
                                                                 self.gts < self.threshold_thickness_is_cloud)))
            all_reg_idxs = np.squeeze(np.nonzero(self.gts_orig >= self.threshold_thickness_is_cloud))
            nbr_thin = len(all_thin_idxs)

            curr_idxs = np.concatenate(
                [np.random.choice(all_0_idxs, nbr_thin), np.random.choice(all_reg_idxs, nbr_thin), all_thin_idxs])
            self.inputs = self.inputs_orig[curr_idxs, :].copy()
            self.gts = self.gts_orig[curr_idxs]
            self.gts_binary = self.gts_binary_orig[curr_idxs]

        self.length = self.inputs.shape[0]

    def __getitem__(self, index):

        x = self.inputs[index]

        # Add noise disturbances to data (if enabled)
        white_noise = np.random.randn(x.shape[0]) * self.means_input * self.input_noise
        x += white_noise

        x = (x - self.means_input) / self.stds_input

        y = self.gts[index]
        if self.normalize_cot:
            self.y /= self.cfg.dataloader.cot_max
        y_binary = self.gts_binary[index]

        item = {
            'x': torch.tensor(x).float(),
            'y': torch.tensor([y]).float(),
            'y_binary': torch.tensor(y_binary).float(),
        }

        return item

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples.'