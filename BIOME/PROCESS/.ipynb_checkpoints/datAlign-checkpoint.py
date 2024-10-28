#######################################################################################################################
## -- necessary libraries -- ##########################################################################################
#######################################################################################################################
import numpy as np
from ..FILE import *
import scipy.signal as sg
from scipy.stats import spearmanr

#######################################################################################################################
## -- aligning finapres labels with corresponding waveforms -- ########################################################
#######################################################################################################################
def findOutlierIndices(data_src, data_tgt, threshold = 3):
  z_scores_src = (data_src - np.mean(data_src)) / np.std(data_src)
  outlier_mask_src = np.abs(z_scores_src) > threshold
  z_scores_tgt = (data_tgt - np.mean(data_tgt)) / np.std(data_tgt)
  outlier_mask_tgt = np.abs(z_scores_tgt) > threshold
  combined_mask = outlier_mask_src | outlier_mask_tgt
  indices_to_remove = np.where(combined_mask)[0]
  return indices_to_remove

def datAlign(re_tim, re_tgt, ms_tim, lower = None, upper = None):
  
  ## -- IBI calculation -- ##
  bioz_IBI, true_IBI = np.diff(ms_tim), np.diff(re_tim)
  true_TGT, true_IDX = re_tgt[:-1], np.arange(len(true_IBI))
  upper_limit = len(true_TGT) - len(bioz_IBI) - 1
  lower_limit = 0
  true_IDX = [true_IDX[i] for i in range(lower_limit, upper_limit) if (true_TGT[i:i + len(bioz_IBI)] != 500).all()]
  
  ## -- IBI allignment -- ##
  min_error, e_rmse, p_corr, s_corr = np.Inf, np.Inf, 0, 0
  IBI_src, IBI_tgt = true_IBI[:len(bioz_IBI)], bioz_IBI
  for i in range(len(true_IDX)):
    
    if (lower != None) and (lower != None):
      if not ((lower <= re_tim[i]) and (re_tim[i] <= upper)):
        continue
    
    new_error = np.mean(np.abs(np.subtract(true_IBI[true_IDX[i]:true_IDX[i] + len(bioz_IBI)], bioz_IBI[:len(bioz_IBI)])))
    if new_error < min_error:
      min_error = new_error
      offset = true_IDX[i]

      IBI_tgt, IBI_src = bioz_IBI, true_IBI[true_IDX[i]:true_IDX[i] + len(bioz_IBI)]
      bioz_tmp_IBI, true_tmp_IBI = bioz_IBI, true_IBI[true_IDX[i]:true_IDX[i] + len(bioz_IBI)]
      mask = findOutlierIndices(bioz_tmp_IBI, true_tmp_IBI, 2)
      true_tmp_IBI = np.delete(true_tmp_IBI, mask)
      bioz_tmp_IBI = np.delete(bioz_tmp_IBI, mask)

      e_rmse = np.mean(np.abs(np.subtract(bioz_tmp_IBI, true_tmp_IBI)))
      p_corr = np.corrcoef(bioz_tmp_IBI, true_tmp_IBI)[0, 1]
      s_corr, _ = spearmanr(bioz_tmp_IBI, true_tmp_IBI)

  return offset, (e_rmse, p_corr, s_corr)
