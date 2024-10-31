#######################################################################################################################
## -- necessary libraries -- ##########################################################################################
#######################################################################################################################
import os
import numpy as np
import pandas as pd
from ..MODELS import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

#######################################################################################################################
## -- physiology based training -- ####################################################################################
#######################################################################################################################
def phyAware(args, dataFrame = None):
  mappingArg, genericArg = args["mapping"]["phyAware"], args["generic"]
  inpDir, inpFolder, inpPkl = genericArg["outDir"], genericArg["outFolder"], genericArg["outPkl"]
  subjectList, phyAwareActive = mappingArg["subjectList"], mappingArg["phyAwareActive"]
  savePytorch, saveReportLogs = mappingArg["savePytorch"], mappingArg["saveReportLogs"]
  valSetRatio, phyAwareDevice = mappingArg["valSetRatio"], mappingArg["phyAwareDevice"]
  maxCpuCount, parallelActive = mappingArg["maxCpuCount"], mappingArg["parallelActive"]
  maxCpuCount = None if not parallelActive else maxCpuCount
  
  if not phyAwareActive:
    print("Module 'phyAware' has been marked as deactive; Activate from the arguments file.")
    return False
  
  if dataFrame == None:
    dataFrame = pd.read_pickle(os.path.join(inpDir, inpFolder, f"{inpPkl}.pkl.gz"), compression = "gzip")

  dataFrame = qFilters(args, dataFrame)
  sortedSubjectList = sorted(dataFrame["subject"].unique())
  phaseBundle = [["ph02", "ph03", "ph04"], ["ph05", "ph06"], ["ph07", "ph08"]] # 
  
  for subject in sortedSubjectList:
    if subject not in subjectList and subjectList != "All":
      continue
      
    trValSet, teSub = [s for s in sortedSubjectList if s != subject], [subject]
    trSub, vaSub = valSplit(trValSet, valSetRatio)

    trDf = dataFrame[dataFrame["subject"].isin(trSub)]
    vaDf = dataFrame[dataFrame["subject"].isin(vaSub)]
    teDf = dataFrame[dataFrame["subject"].isin(teSub)]
    
    for ph in phaseBundle:
      
      trDfPh = trDf[trDf["phase"].isin(ph)]
      vaDfPh = vaDf[vaDf["phase"].isin(ph)]
      teDfPh = teDf[teDf["phase"].isin(ph)]
      
      print(("Physiology: " + str(ph) + " | "), 
            ("trSize: " + str(len(trDfPh)) + " | "), 
            ("vaSize: " + str(len(vaDfPh)) + " | "), 
            ("teSize: " + str(len(teDfPh)) + " | "), 
            ("Device: " + str(phyAwareDevice)), flush = True)
      
      if len(trDfPh) and len(vaDfPh) and len(teDfPh):
        
        # try:
        print(f"subject {subject}, phase bundle {ph}, started.", flush = True)
        trInp, trLbl = dataExtr(trDfPh)
        vaInp, vaLbl = dataExtr(vaDfPh)
        teInp, teLbl = dataExtr(teDfPh)

        tr_X = torch.tensor(trInp, dtype = torch.float32, requires_grad = True).to(phyAwareDevice)
        tr_y = torch.tensor(trLbl, dtype = torch.float32, requires_grad = True).to(phyAwareDevice)

        va_X = torch.tensor(vaInp, dtype = torch.float32, requires_grad = False).to(phyAwareDevice)
        va_y = torch.tensor(vaLbl, dtype = torch.float32, requires_grad = False).to(phyAwareDevice)

        te_X = torch.tensor(teInp, dtype = torch.float32, requires_grad = False).to(phyAwareDevice)
        te_y = torch.tensor(teLbl, dtype = torch.float32, requires_grad = False).to(phyAwareDevice)

        trDataset = TensorDataset(tr_X, tr_y)
        vaDataset = TensorDataset(va_X, va_y)
        teDataset = TensorDataset(te_X, te_y)

        trLoader = DataLoader(trDataset, batch_size = 512, shuffle = True)  # num_workers = maxCpuCount, pin_memory = True
        vaLoader = DataLoader(vaDataset, batch_size = 512, shuffle = True)  # num_workers = maxCpuCount, pin_memory = True
        teLoader = DataLoader(teDataset, batch_size = 512, shuffle = False) # num_workers = maxCpuCount, pin_memory = True
        
        model = ComplexCNN(args)
        model.count_parameters()
        model.train_model(subject, ph, genericArg["initialSeed"], trLoader, vaLoader, teLoader, num_epochs=500, learning_rate=1e-4, device = phyAwareDevice)

        print()
        print()
          
        # except Exception as e:
          # print(f"subject {subject}, phase bundle {ph}, failed tr block with error: {e}", flush = True)
      
      else:
        print(f"subject {subject}, phase bundle {ph}, does not exist.", flush = True)