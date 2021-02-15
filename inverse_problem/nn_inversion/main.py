from .dataset import SpectrumDataset
import torch
from tqdm import tqdm
from torch import nn
import os
from pathlib import Path
from torch.utils.data import DataLoader
from inverse_problem.nn_inversion.models import HyperParams, TopNet
from inverse_problem.nn_inversion import models
from inverse_problem.nn_inversion import transforms


class Model:
    """Model class for fitting data
    """

    def __init__(self, hps: HyperParams):
        self.hps = hps
        self.criterion = nn.MSELoss()
        self.on_gpu = torch.cuda.is_available()
        self.net = getattr(models, hps.net)(hps)
        self.optimizer = self._init_optimizer()
        self.transform = getattr(transforms, hps.transform_type)()
        if self.on_gpu:
            self.net.cuda()

    def _init_optimizer(self):
        return torch.optim.Adam(self.net.parameters(),
                                lr=self.hps.lr, weight_decay=self.hps.weight_decay)

    def fit_step(self, sample_batch):
        # todo only work for simple model, need to refactor later
        x = sample_batch['X']
        y = sample_batch['Y'][:, self.hps.predict_ind]
        self.optimizer.zero_grad()
        outputs = self.net(x)
        loss = self.criterion(outputs, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def eval_step(self, sample_batch):
        self.net.eval()
        x = sample_batch['X']
        y = sample_batch['Y'][:, self.hps.predict_ind]
        with torch.no_grad():
            outputs = self.net(x)
            loss = self.criterion(outputs, y)
        return loss.item()

    def make_loader(self, filename: Path = None) -> DataLoader:
        if filename is None:
            project_path = Path(__file__).resolve().parents[2]
            filename = os.path.join(project_path, 'data/parameters_base.fits')
        transformed_dataset = SpectrumDataset(filename, source=self.hps.source,
                                              transform=self.transform)
        return DataLoader(transformed_dataset, batch_size=self.hps.batch_size, shuffle=True)


    def train(self, model_path: Path = None):
        train_loader = self.make_loader()
        best_valid_loss = float('inf')
        history = []
        log_template = "\nEpoch {ep:03d} train_loss: {t_loss:0.4f} \
         val_loss {v_loss:0.4f}"
        train_loss = float('inf')
        val_loss = float('inf')
        with tqdm(desc="epoch", total=self.hps.n_epochs) as pbar_outer:
            for epoch in range(self.hps.n_epochs):
                for i, sample in enumerate(train_loader):
                    if self.hps.per_epoch == i:
                        break
                    train_loss = self.fit_step(sample)
                    val_loss = self.eval_step(sample)
                    history.append((train_loss, val_loss))

                pbar_outer.update(1)
                tqdm.write(log_template.format(ep=epoch + 1, t_loss=train_loss,
                                               v_loss=val_loss))
        return history

