import torch
import torch.nn as nn
from torch.utils import data as torch_data
from torch import Tensor
from torcheval.metrics import MeanSquaredError, R2Score
import numpy as np
import wandb
from tqdm import tqdm
from utils import datasets
from omegaconf import DictConfig
from typing import Tuple


class Measurer(object):
    def __init__(self, normalized: bool, max_value: float):
        self.normalized = normalized
        self.max_value = max_value
        self.preds, self.labels = [], []
        self.metrics = {
            'mse': MeanSquaredError(),
            'r2': R2Score(),
        }

    def add_sample(self, pred: Tensor, label: Tensor):
        pred = pred.float().detach().cpu().flatten()
        label = label.float().detach().cpu().flatten()
        if self.normalized:
            pred, label = pred * self.max_value, label * self.max_value

        self.metrics['mse'].update(pred, label)
        self.metrics['r2'].update(pred, label)
        self.preds.extend(list(pred.numpy()))
        self.labels.extend(list(label.numpy()))

    def reset(self):
        self.labels, self.preds = [], []
        self.metrics['mse'].reset()
        self.metrics['r2'].reset()

    def mse(self) -> float:
        return self.metrics['mse'].compute().item()

    def r2(self) -> float:
        return self.metrics['r2'].compute().item()


def model_evaluation(model: nn.Module, cfg: DictConfig, run_type: str, epoch: float,
                     max_samples: int = None) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    measurer = Measurer(normalized=cfg.dataloader.normalize_cot, max_value=cfg.dataloader.cot_max)
    dataset = datasets.Dataset(cfg, run_type, no_augmentations=True)
    dataloader_kwargs = {
        'batch_size': cfg.trainer.batch_size,
        'num_workers': 0 if cfg.debug else cfg.dataloader.num_workers,
        'shuffle': True,  # Shuffle due to max samples
        'pin_memory': True,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    max_samples = len(dataset) * cfg.trainer.batch_size if max_samples is None else max_samples
    counter = 0


    for step, batch in enumerate(tqdm(dataloader)):
        x, y = batch['x'].to(device), batch['y'].to(device)
        with torch.no_grad():
            y_hat = model(x)

        measurer.add_sample(y_hat, y)
        counter += x.size(0)
        if counter >= max_samples:
            break

    # Log to WandB
    mse = measurer.mse()
    r2 = measurer.r2()
    wandb.log({
        f'{run_type} mse': mse,
        f'{run_type} r2': r2,
        f'{run_type} cot avg label': np.mean(measurer.labels),
        f'{run_type} cot avg pred': np.mean(measurer.preds),
        'step': step,
        'epoch': epoch,
    })

    return mse


# # Track training and validation statistics
# if it % cfg.log_freq == 0 and USE_SC:
#     # Reset val_ctr if necessary
#     if (val_ctr + 1) * cfg.trainer.batch_size >= nbr_examples_val:
#         val_ctr = 0
#     # Compute a prediction and get the loss
#     curr_gts = gts_val[val_ctr * cfg.trainer.batch_size: (val_ctr + 1) * cfg.trainer.batch_size]
#     curr_gts_binary = gts_val_binary[val_ctr * cfg.trainer.batch_size: (val_ctr + 1) * cfg.trainer.batch_size]
#     preds_val = 0
#     for model in models:
#         model.eval()
#         curr_pred = model(
#             inputs_val[val_ctr * cfg.trainer.batch_size: (val_ctr + 1) * cfg.trainer.batch_size, :]) / len(models)
#         preds_val += curr_pred
#         model.train()
#     loss_val = criterion(preds_val[:, 0], curr_gts)
#     loss_val.cpu().detach().numpy()
#     sc.s(sc_loss_string_base + '_val').collect(loss_to_sc)
#
#     val_ctr += 1
#     sc.print()
#     sc.save()