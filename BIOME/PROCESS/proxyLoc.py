#######################################################################################################################
## -- necessary libraries -- ##########################################################################################
#######################################################################################################################
import copy
import numpy as np
from ..FILE import *
from ..PROCESS import *

import matplotlib.pyplot as plt
#######################################################################################################################
## -- detects the plausible location of the maneuver on the waveform -- ###############################################
#######################################################################################################################
def proxyLoc(args, directory):
  
  trialDataPackage, log = loadData(args, directory)
  finData, bioData = trialDataPackage["finData"], trialDataPackage["bioData"]
  reFinTim, reFinSbp, reFinDbp = finData["re_tim"], finData["re_sys"], finData["re_dia"]
  fiFinTim, fiFinSbp, fiFinDbp = finData["fi_tim"], finData["fi_sys"], finData["fi_dia"]
  phList = list(bioData.keys())
  
  rePreserve = [copy.copy(reFinTim), copy.copy(reFinSbp), copy.copy(reFinDbp)]
  fiPreserve = [copy.copy(fiFinTim), copy.copy(fiFinSbp), copy.copy(fiFinDbp)]
  
  calClusterCount, calClusters = calClust(reFinSbp, mode = "cal")
  sigClusterCount, sigClusters = calClust(reFinSbp, mode = "sig")
  
  maneuverBundles = []
  maneuverBundles.append([x for x in phList if any(y in x for y in ["02", "03", "04"])])
  maneuverBundles.append([x for x in phList if any(y in x for y in ["05", "06"])])
  maneuverBundles.append([x for x in phList if any(y in x for y in ["07", "08"])])
  
  if calClusterCount == 3:
      
    sigClusters = sigClusters[-3:]
    for sig in range(len(sigClusters) - 1, -1, -1):

      timeSpan = []
      for ph in range(len(maneuverBundles[sig])):
        time = bioData[maneuverBundles[sig][ph]]["data"]["channel1"]["tim"]
        timeSpan.append(time[-1][-1] - time[0][0])
      
      lowLim = 0
      for ph in range(len(maneuverBundles[sig])):
        msTim = bioData[maneuverBundles[sig][ph]]["data"]["biozCharacteristicPoints"]["pinPoint"]["maxSlope"]["tim"]
        try:
          lower = max(reFinTim[sigClusters[sig]][0] + np.sum(timeSpan[:ph]), lowLim)
          upper = reFinTim[sigClusters[sig]][-1] - np.sum(timeSpan[ph:])
          offset, metric = datAlign(reFinTim[sigClusters[sig]], reFinSbp[sigClusters[sig]], msTim, lower, upper)
          offset = offset + sigClusters[sig][0]
          lowLim = reFinTim[offset + len(msTim) - 1] - 0.1 * (reFinTim[offset + len(msTim) - 1] - reFinTim[offset])
          trialDataPackage["bioData"][maneuverBundles[sig][ph]]["prep"] = "CS"
        
        except:
          offset, metric = datAlign(reFinTim, reFinSbp, msTim)
          trialDataPackage["bioData"][maneuverBundles[sig][ph]]["prep"] = "CU"
        
        reTimLabel, fiTimLabel = reFinTim[offset:(offset + len(msTim) - 1)], fiFinTim[offset:(offset + len(msTim) - 1)]
        reSbpLabel, fiSbpLabel = reFinSbp[offset:(offset + len(msTim) - 1)], fiFinSbp[offset:(offset + len(msTim) - 1)]
        reDbpLabel, fiDbpLabel = reFinDbp[offset:(offset + len(msTim) - 1)], fiFinDbp[offset:(offset + len(msTim) - 1)]
        
        trialDataPackage["bioData"][maneuverBundles[sig][ph]]["calQ"] = calClusterCount
        trialDataPackage["bioData"][maneuverBundles[sig][ph]]["wfqm"] = metric
        trialDataPackage["bioData"][maneuverBundles[sig][ph]]["fina"] = {
          "fpAdcVal": finData["fp_adc_val"],
          "fpAdcTim": finData["fp_adc_tim"],
          "reTim": reTimLabel, "fiTim": fiTimLabel,
          "reSbp": reSbpLabel, "fiSbp": fiSbpLabel,
          "reDbp": reDbpLabel, "fiDbp": fiDbpLabel,
        }
             
  else:
    
    timeSpan = []
    for ph in range(len(phList)):
      time = bioData[phList[ph]]["data"]["channel1"]["tim"]
      timeSpan.append(time[-1][-1] - time[0][0])
        
    for ph in range(len(phList)):
      msTim = bioData[phList[ph]]["data"]["biozCharacteristicPoints"]["pinPoint"]["maxSlope"]["tim"]
      try:
        lower = reFinTim[0] + np.sum(timeSpan[:ph])
        upper = reFinTim[-1] - np.sum(timeSpan[ph:])
        offset, metric = datAlign(reFinTim, reFinSbp, msTim, lower, upper)
        trialDataPackage["bioData"][phList[ph]]["prep"] = "US"
      except:
        offset, metric = datAlign(reFinTim, reFinSbp, msTim)
        trialDataPackage["bioData"][phList[ph]]["prep"] = "UU"
        
      reTimLabel, fiTimLabel = reFinTim[offset:(offset + len(msTim) - 1)], fiFinTim[offset:(offset + len(msTim) - 1)]
      reSbpLabel, fiSbpLabel = reFinSbp[offset:(offset + len(msTim) - 1)], fiFinSbp[offset:(offset + len(msTim) - 1)]
      reDbpLabel, fiDbpLabel = reFinDbp[offset:(offset + len(msTim) - 1)], fiFinDbp[offset:(offset + len(msTim) - 1)]

      trialDataPackage["bioData"][phList[ph]]["calQ"] = calClusterCount
      trialDataPackage["bioData"][phList[ph]]["wfqm"] = metric
      trialDataPackage["bioData"][phList[ph]]["fina"] = {
        "fpAdcVal": finData["fp_adc_val"],
        "fpAdcTim": finData["fp_adc_tim"],
        "reTim": reTimLabel, "fiTim": fiTimLabel,
        "reSbp": reSbpLabel, "fiSbp": fiSbpLabel,
        "reDbp": reDbpLabel, "fiDbp": fiDbpLabel,
      }
      
  return trialDataPackage["bioData"], [rePreserve, fiPreserve], log
  