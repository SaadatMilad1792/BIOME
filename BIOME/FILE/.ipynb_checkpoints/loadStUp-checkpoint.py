#######################################################################################################################
## -- necessary libraries -- ##########################################################################################
#######################################################################################################################
import os
import sys
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..FILE import *
from ..PROCESS import *
from ..ANALYTIC import *

#######################################################################################################################
## -- parallelized internal function call -- ##########################################################################
#######################################################################################################################
def parallel(args, trialDir, trial, demographic, subject, date, shift, genericArg):
  inpDirectory = os.path.join(trialDir, trial)
  trialDataPackage, finapresRaw, log = proxyLoc(args, inpDirectory)
  reFinTim, reFinSbp, reFinDbp = finapresRaw[0][0], finapresRaw[0][1], finapresRaw[0][2]
  rowKeepList = []
  
  for phaseId, phase in enumerate(trialDataPackage):
    data = trialDataPackage[phase]["data"]
    fina = trialDataPackage[phase]["fina"]
    desc = trialDataPackage[phase]["desc"]
    calQ = trialDataPackage[phase]["calQ"]
    wfqm = trialDataPackage[phase]["wfqm"]
    prep = trialDataPackage[phase]["prep"]
    
    demo = demographic[demographic["pID"].str.upper() == subject.upper()]
    numCycle = data["biozCharacteristicPoints"]["numCycle"]
    
    for cyc in range(numCycle):
      shift = fina["reTim"][0] - data["finaWave"]["fp_adc_tim"][0][0]
      rowKeepList.append(pd.DataFrame([{
        "project_name": genericArg["projectName"],
        "subject": subject,
        "date": date,
        "trial": trial,
        "phase": f"ph{phase[-2:]}",
        "phase_description": desc,
        "calQ": calQ,
        "wfqm": wfqm,
        "prep": prep,
        "age": list(demo["age"])[0],
        "gender": list(demo["gender"])[0],
        "height (cm)": list(demo["height_cm"])[0],
        "weight (kg)": list(demo["weight_kg"])[0],
        "ch1": {
          "abs": data["channel1"]["abs"][cyc],
          "phi": data["channel1"]["phi"][cyc],
          "del": data["channel1"]["del"][cyc],
          "icg": data["channel1"]["icg"][cyc],
          "tim": data["channel1"]["tim"][cyc] + shift,
        },
        "ch2": {
          "abs": data["channel2"]["abs"][cyc],
          "phi": data["channel2"]["phi"][cyc],
          "del": data["channel2"]["del"][cyc],
          "icg": data["channel2"]["icg"][cyc],
          "tim": data["channel2"]["tim"][cyc] + shift,
        },
        "ppg": {
          "mag": data["ppgData"]["mag"][cyc],
          "tim": data["ppgData"]["tim"][cyc] + shift,
        },
        "acc": {
          "xax": data["accmeter"]["xax"][cyc],
          "yax": data["accmeter"]["yax"][cyc],
          "zax": data["accmeter"]["zax"][cyc],
          "mag": data["accmeter"]["mag"][cyc],
          "tim": data["accmeter"]["tim"][cyc] + shift,
        },
        "fin": {
          "fp_adc_val": data["finaWave"]["fp_adc_val"][cyc],
          "fp_adc_tim": data["finaWave"]["fp_adc_tim"][cyc],
          "fp_tim": reFinTim,
          "fp_sys": reFinSbp,
          "fp_dia": reFinDbp,
          "re_tim": fina["reTim"][cyc],
          "re_sys": fina["reSbp"][cyc],
          "re_dia": fina["reDbp"][cyc],
          "fi_tim": fina["fiTim"][cyc],
          "fi_sys": fina["fiSbp"][cyc],
          "fi_dia": fina["fiDbp"][cyc],
        },
        "characteristic_points": {
          "max_crest": {
            "val": data["biozCharacteristicPoints"]["pinPoint"]["maxCrest"]["val"][cyc],
            "tim": data["biozCharacteristicPoints"]["pinPoint"]["maxCrest"]["tim"][cyc] + shift,
          },
          "max_slope": {
            "val": data["biozCharacteristicPoints"]["pinPoint"]["maxSlope"]["val"][cyc],
            "tim": data["biozCharacteristicPoints"]["pinPoint"]["maxSlope"]["tim"][cyc] + shift,
          },
          "min_crest": {
            "val": data["biozCharacteristicPoints"]["pinPoint"]["minCrest"]["val"][cyc],
            "tim": data["biozCharacteristicPoints"]["pinPoint"]["minCrest"]["tim"][cyc] + shift,
          },
        },
      }]))
  return pd.concat(rowKeepList)


#######################################################################################################################
## -- loads trials and dates for a subject -- #########################################################################
#######################################################################################################################
def loadStUp(args):
  rowKeepList = []
  processArg, genericArg = args["process"]["loadStUp"], args["generic"]
  inpDir, inpFolder = genericArg["inpDir"], genericArg["inpFolder"]
  outDir, outFolder = genericArg["outDir"], genericArg["outFolder"]
  outPkl = genericArg["outPkl"]
  directory = os.path.join(inpDir, inpFolder)
  subjectList, loadStUpActive = processArg["subjectList"], processArg["loadStUpActive"]
  savePickles, savePltOverall = processArg["savePickles"], processArg["savePltOverall"]
  maxCpuCount, parallelActive = processArg["maxCpuCount"], processArg["parallelActive"]
  maxCpuCount = None if not parallelActive else maxCpuCount

  if not loadStUpActive:
    print("Module 'loadStUp' has been marked as deactive; Activate from the arguments file.")
    return False

  demographic = pd.read_pickle(os.path.join(directory, "../", "demographics.pkl"), compression="gzip")
  sortedSubjectList = sorted(dirSweep(directory))
  
  with ProcessPoolExecutor(max_workers = maxCpuCount) as executor:
    futures = []
    for subject in sortedSubjectList:
      if subject not in subjectList and subjectList != "All":
        continue

      print(subject)
      dateDir = os.path.join(directory, subject)
      dateList = dirSweep(dateDir)
      
      for date in dateList:
        trialDir = os.path.join(dateDir, date, "BioZ_Data_Processed")
        trialList = dirSweep(trialDir)
        
        for trial in trialList:
          futures.append(executor.submit(parallel, args, trialDir, trial, demographic, subject, date, 0, genericArg))

    for future in as_completed(futures):
      result = future.result()
      rowKeepList.append(result)
      print(f"Completed Processing [SUBJECT: {result['subject'].unique()}, DATE: {result['date'].unique()}, TRIAL: {result['trial'].unique()}]")

  rowKeepList = pd.concat(rowKeepList).reset_index(drop = True)
  rowKeepList = rowKeepList.sort_values(by = ['subject', 'trial']).reset_index(drop = True)

  if savePltOverall:
    analytic(args, rowKeepList)

  if savePickles:
    rowKeepList.to_pickle(os.path.join(outDir, outFolder, f"{outPkl}.pkl.gz"), compression = "gzip")
  
  return rowKeepList