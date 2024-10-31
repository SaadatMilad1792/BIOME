#######################################################################################################################
## -- necessary libraries -- ##########################################################################################
#######################################################################################################################
import os
import numpy as np
import pandas as pd
from ..MODELS import *

#######################################################################################################################
## -- data extraction module -- #######################################################################################
#######################################################################################################################
def dataExtr(dataFrame):
  
  ch1DelList, ch1AbsList, ch1TimList, ch1IcgList, ch1PhiList = [], [], [], [], []
  for subject, subjectDataFrame in dataFrame.groupby("subject"):

    ch1del = np.array(list(subjectDataFrame["ch1"].apply(lambda x: x["del"])))
    ch1abs = np.array(list(subjectDataFrame["ch1"].apply(lambda x: x["abs"])))
    ch1tim = np.array(list(subjectDataFrame["ch1"].apply(lambda x: x["tim"] - x["tim"][0])))
    ch1icg = np.array(list(subjectDataFrame["ch1"].apply(lambda x: x["icg"])))
    ch1phi = np.array(list(subjectDataFrame["ch1"].apply(lambda x: x["phi"])))

    ch1del = (ch1del - ch1del.mean()) / ch1del.std()
    ch1abs = (ch1abs - ch1abs.mean()) / ch1abs.std()
    ch1tim = (ch1tim - ch1tim.mean()) / ch1tim.std()
    ch1icg = (ch1icg - ch1icg.mean()) / ch1icg.std()
    ch1phi = (ch1phi - ch1phi.mean()) / ch1phi.std()

    ch1DelList.append(ch1del)
    ch1AbsList.append(ch1abs)
    ch1TimList.append(ch1tim)
    ch1IcgList.append(ch1icg)
    ch1PhiList.append(ch1phi)

  ch1del = np.concatenate(ch1DelList, axis = 0)
  ch1abs = np.concatenate(ch1AbsList, axis = 0)
  ch1tim = np.concatenate(ch1TimList, axis = 0)
  ch1icg = np.concatenate(ch1IcgList, axis = 0)
  ch1phi = np.concatenate(ch1PhiList, axis = 0)

  obj = np.stack([ch1del, ch1abs, ch1tim, ch1icg, ch1phi], axis = 1)

  lbldia = np.array(list(dataFrame["fin"].apply(lambda x: x["re_dia"]))).reshape(-1, 1)
  lblsys = np.array(list(dataFrame["fin"].apply(lambda x: x["re_sys"]))).reshape(-1, 1)
  lbl = np.concatenate([lbldia - np.mean(lbldia), lblsys - np.mean(lblsys)], axis = 1)

  return obj, lbl