import numpy as np
import pandas as pd
import torch
from inverse_problem.milne_edington.me import read_full_spectra, HinodeME, BatchHinodeME



def prepare_data(data_path, size_limit=None, batch_size=10000):

    params = fits.open(DATA_PATH)[0].data
    lines = None

    if size_limit is None:
      size_limit = params.shape[0]

    for i in range((size_limit - 1)//batch_size + 1):
      print(f'Computing: {batch_size*i} - {min(batch_size*(i+1), params.shape[0])}')
      modelBatchME = BatchHinodeME(params[batch_size*i:min(batch_size*(i+1), size_limit)])
      if lines is None:
        lines = modelBatchME.compute_spectrum()
      else:
        lines = np.concatenate((lines, modelBatchME.compute_spectrum()), axis=0)

    lines = np.reshape(lines, (lines.shape[0], lines.shape[1]*lines.shape[2])).astype(np.float32)
    params = params[:size_limit].astype(np.float32)

    return lines, params
  
  
  
  def calculate_metrics(true_values, predicted_values, save_path = f'/metrics/model_metrics.csv'):

    names = ['Field_Strength',
             'Field_Inclination',
             'Field_Azimuth',
             'Doppler_Width',
             'Damping',
             'Line_Strength',
             'Original_Continuum_Intensity',
             'Source_Function_Gradient',
             'Doppler_Shift2',
             'Stray_Light_Fill_Factor',
             'Stray_light_Shift']

    params_scale = np.max(true_values, axis=0) - np.min(true_values, axis=0)
    metrics_df = compute_metrics(np.expand_dims(true_values, axis=0), np.expand_dims(predicted_values, axis=0), names, save_path = f'/content/metrics_{current_epoch}.csv')
    metrics_df['mse'] = metrics_df['mse'] / (params_scale**2)
    metrics_df['mae'] = metrics_df['mae'] / (params_scale)

    return metrics_df
  
  
  
  def compare_metrics(base, distilled):

    metrics = []
    metrics.append(np.sqrt(np.mean(distilled['r2'] / base['r2'])))
    metrics.append(np.sqrt(np.mean(base['mse'] / distilled['mse'])))
    metrics.append(np.mean(base['mae'] / distilled['mae']))    
    
    return (sum(metrics) / len(metrics))
