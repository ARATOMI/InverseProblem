import numpy as np
import pandas as pd
import torch
from inverse_problem.milne_edington.me import read_full_spectra, HinodeME, BatchHinodeME
from inverse_problem.nn_inversion import transforms



class HinodeDataset(Dataset):

    def __init__(self, 
                 mode, 
                 data_path, 
                 train_split: float = 0.8, 
                 scaler = None,
                ):
        
        full_dataset = fits.open(data_path)[0].data

        if mode == 'train':
            self._params_dataset = full_dataset[:int(train_split * full_dataset.shape[0])]
        elif mode == 'val':
            self._params_dataset = full_dataset[int(train_split * full_dataset.shape[0]):]
        elif mode == 'smol':
            self._params_dataset = full_dataset[:100000]

        self.scaler = scaler
        
        

    def __len__(self):
        return self._params_dataset.shape[0]
        

    def __getitem__(self, index):

        params = self._params_dataset[index]
        spectrum = HinodeME(params).compute_spectrum().reshape(-1)

        if self.scaler:
            params = self.scaler.scale_params(params)
            spectrum = self.scaler.scale_spectrum(spectrum)
       
        params = torch.tensor(params.astype(np.float32), dtype=torch.float32) #torch.from_numpy(params.astype(np.float32))
        spectrum = torch.tensor(spectrum.astype(np.float32), dtype=torch.float32) #torch.from_numpy(spectrum.reshape(-1).astype(np.float32))

        
            
        return spectrum, params
        
        
        
        
        
        
        
