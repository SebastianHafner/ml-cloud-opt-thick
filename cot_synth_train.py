import random
import datetime
import timeit
import numpy as np

import torch
import torch.nn as nn
from torch.utils import data as torch_data

from utils import datasets, evaluation
from models import model_manager
import wandb

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path='config', config_name='config')
def run_training(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg)) # Print the entire config

    # Make training deterministic
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running {cfg.model.name} on device {device}')

    # Initialize logging
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    wandb.init(
        name=f'{cfg.model.name}_{timestamp}',
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        project='ml-cloud-opt-thick',
        mode='online' if not cfg.debug else 'disabled',
    )

    dataset = datasets.Dataset(cfg, run_type='train')
    print(dataset)

    model = model_manager.build_model(cfg, dataset.input_dim, 1)
    model.to(device)

    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.trainer.lr,
        weight_decay=cfg.trainer.weight_decay,
        betas=(cfg.trainer.beta1, 0.999)
    )

    dataloader_kwargs = {
        'batch_size': cfg.trainer.batch_size,
        'num_workers': 0 if cfg.debug else cfg.dataloader.num_workers,
        'shuffle': cfg.dataloader.shuffle,
        'drop_last': True,
        'pin_memory': True,
    }
    dataloader = torch_data.DataLoader(dataset, **dataloader_kwargs)

    # unpacking cfg
    epochs = cfg.trainer.epochs
    steps_per_epoch = len(dataloader)

    # tracking variables
    global_step = epoch_float = 0

    # early stopping
    best_val_value = None
    trigger_times = 0
    stop_training = False
    best_val_value = evaluation.model_evaluation(model, cfg, 'val', epoch_float, cfg.trainer.max_eval_samples)

    for epoch in range(1, epochs + 1):
        print(f'Starting epoch {epoch}/{epochs}.')

        start = timeit.default_timer()
        loss_set = []

        for i, batch in enumerate(dataloader):
            model.train()
            optimizer.zero_grad()

            x, y = batch['x'].to(device), batch['y'].to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            loss_set.append(loss.item())
            global_step += 1
            epoch_float = global_step / steps_per_epoch

            if global_step % cfg.log_freq == 0:
                print(f'Logging step {global_step} (epoch {epoch_float:.2f}).')
                # logging
                time = timeit.default_timer() - start
                wandb.log({
                    'loss': np.mean(loss_set),
                    'time': time,
                    'step': global_step,
                    'epoch': epoch_float,
                })
                start = timeit.default_timer()
                loss_set = []

        assert (epoch == epoch_float)
        print(f'epoch float {epoch_float} (step {global_step}) - epoch {epoch}')

        _ = evaluation.model_evaluation(model, cfg, 'train', epoch_float, cfg.trainer.max_eval_samples)
        val_value = evaluation.model_evaluation(model, cfg, 'val', epoch_float, cfg.trainer.max_eval_samples)

        print(f'val value {val_value:.3f} (best val value {best_val_value:.3f})')
        if best_val_value is not None and val_value >= best_val_value:
            trigger_times += 1
            if trigger_times >= cfg.trainer.patience:
                stop_training = True
        else:
            best_val_value = val_value
            wandb.log({
                'best val': best_val_value,
                'step': global_step,
                'epoch': epoch_float,
            })
            print(f'saving network ({best_val_value:.3f})', flush=True)
            model_manager.save_model(model, optimizer, epoch, cfg)
            trigger_times = 0

        if stop_training:
            break

    model = model_manager.load_model(cfg, device)
    _ = evaluation.model_evaluation(model, cfg, 'test', epoch_float)


if __name__ == '__main__':
    run_training()