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

        self.property_column_mapping = {
            'spec_bands': [i for i in range(1 + cfg.dataloader.skip_band1, 14)],
            'angles': [14, 15, 16],
            'thick': [17],
            'type': [18],
            'prof_id': [19],
            'gas_vapour': [20, 21],
            'surf_prof': [22]
        }

        # Unpacking cfg
        self.regressor = cfg.dataloader.regressor
        self.input_keys = cfg.dataloader.inputs
        self.skip_band1 = cfg.dataloader.skip_band1
        self.skip_band10 = cfg.dataloader.skip_band10
        self.threshold_thickness_is_thin_cloud = cfg.dataloader.threshold_thickness_is_thin_cloud
        self.threshold_thickness_is_cloud = cfg.dataloader.threshold_thickness_is_cloud
        self.uniform_dist_no_cloud_thin_cloud_reg_cloud = cfg.dataloader.uniform_dist_no_cloud_thin_cloud_reg_cloud
        self.input_noise = cfg.dataloader.input_noise_train if self.run_type == 'train' else cfg.dataloader.input_noise_val

        # Specify model input and output dimensions
        self.input_dim = np.sum([len(self.property_column_mapping[inp]) for inp in self.input_keys]) - self.skip_band10

        # Read data
        self.dataset = np.load(self.dataset_path / f'{self.run_type}set_smhi.npy')
        nbr_examples = self.dataset.shape[0]

        # Separate input and regression variable
        self.inputs = np.concatenate([self.dataset[:, self.property_column_mapping[inp]] for inp in self.input_keys], axis=1)
        self.gts = np.squeeze(self.dataset[:, self.property_column_mapping[self.regressor]])
        self.gts_binary = np.squeeze(self.dataset[:, self.property_column_mapping['type']]) > 0
        if self.skip_band10:
            if self.skip_band1:
                self.inputs = self.inputs[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11]]
            else:
                self.inputs = self.inputs[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]]

        # TODO: Also move to get item
        # Normalize regressor data
        self.gts /= self.cfg.dataloader.cot_max

        # Create copies of original entities (some modifications, e.g. adding
        # noise, is done after each epoch, and the starting point per modification
        # should always be the original data)
        self.inputs_orig = self.inputs.copy()
        self.gts_orig = self.gts.copy()
        self.gts_binary_orig = self.gts_binary.copy()

        self.means_input = np.mean(self.inputs_orig, axis=0)
        self.stds_input = np.std(self.inputs_orig, axis=0)

        # Below we ensure equal distribution of no-cloud, thin-cloud, opaque-cloud
        if self.uniform_dist_no_cloud_thin_cloud_reg_cloud:
            all_0_idxs = np.squeeze(np.nonzero(self.gts_orig < self.threshold_thickness_is_thin_cloud))
            all_thin_idxs = np.squeeze(np.nonzero(np.logical_and(self.gts >= self.threshold_thickness_is_thin_cloud,
                                                                 self.gts < self.threshold_thickness_is_cloud)))
            all_reg_idxs = np.squeeze(np.nonzero(self.gts_orig >= self.threshold_thickness_is_cloud))
            nbr_thin = len(all_thin_idxs)
            nbr_examples = nbr_thin * 3

            perm = np.random.permutation(nbr_examples)
            curr_idxs = np.concatenate(
                [np.random.choice(all_0_idxs, nbr_thin), np.random.choice(all_reg_idxs, nbr_thin), all_thin_idxs])
            self.inputs = self.inputs_orig[curr_idxs, :].copy()
            self.inputs = self.inputs[perm, :]
            self.gts = self.gts_orig[curr_idxs]
            self.gts = self.gts[perm]
            self.gts_binary = self.gts_binary_orig[curr_idxs]
            self.gts_binary = self.gts_binary[perm]

        self.length = nbr_examples

    def __getitem__(self, index):

        x = self.inputs[index]

        # Add noise disturbances to data (if enabled)
        white_noise = np.random.randn(x.shape[0]) * self.means_input * self.input_noise
        x += white_noise
        # TODO: means and stds should always come from the train dataset
        x = (x - self.means_input) / self.stds_input

        y = self.gts[index]
        y_binary = self.gts_binary[index]

        item = {
            'x': torch.tensor(x).float(),
            'y': torch.tensor(y).float(),
            'y_binary': torch.tensor(y_binary).float(),
        }

        return item

    def __len__(self):
        return self.length

    def __str__(self):
        return f'Dataset with {self.length} samples.'