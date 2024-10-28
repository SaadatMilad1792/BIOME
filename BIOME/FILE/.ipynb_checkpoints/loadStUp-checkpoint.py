#######################################################################################################################
## -- necessary libraries -- ##########################################################################################
#######################################################################################################################
import pandas as pd
from ..FILE import *
from ..PROCESS import *
from ..ANALYTIC import *

import sys
#######################################################################################################################
## -- loads trials and dates for a subject -- #########################################################################
#######################################################################################################################
def loadStUp(args):
  
  rowKeepList = []
  subArgsObj, genericArg = args["process"]["loadStUp"], args["generic"]
  inpDir, inpFolder = genericArg["inpDir"], genericArg["inpFolder"]
  outDir, outFolder = genericArg["outDir"], genericArg["outFolder"]
  outPkl = genericArg["outPkl"]
  directory = os.path.join(inpDir, inpFolder)
  KeepSubList = subArgsObj["subjectList"]
  subjectList = sorted(dirSweep(directory))
  demographic = pd.read_pickle(directory + "/../" + "demographics.pkl", compression = "gzip")
  for subId, subject in enumerate(subjectList):
    
    if (subject not in KeepSubList) and (KeepSubList != "All"):
      continue
    
    print(subject)
    dateDir = directory + "/" + subject
    dateList = dirSweep(dateDir)
    for dateId, date in enumerate(dateList):
      
      trialDir = dateDir + "/" + date + "/" + "BioZ_Data_Processed"
      trialList = dirSweep(trialDir)
      for trialId, trial in enumerate(trialList):
        
        inpDirectory = trialDir + "/" + trial
        trialDataPackage, finapresRaw, log = proxyLoc(args, inpDirectory)
        reFinTim, reFinSbp, reFinDbp = finapresRaw[0][0], finapresRaw[0][1], finapresRaw[0][2]
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
          
            shift = fina["reTim"][0] - fina["fpAdcTim"][0][0] # fina["reTim"][cyc] - fina["fpAdcTim"][cyc][0]
            rowKeepList.append(pd.DataFrame([{
              "project_name": genericArg["projectName"],
              "subject": subject,
              "date": date,
              "calQ": calQ,
              "wfqm": wfqm,
              "prep": prep,
              "trial": trial,
              "phase": f"ph{phase[-2:]}",
              "age": list(demo["age"])[0],
              "gender": list(demo["gender"])[0],
              "height (cm)": list(demo["height_cm"])[0],
              "weight (kg)": list(demo["weight_kg"])[0],
              "phase_description": desc,
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
              "biozCharacteristicPoints": {
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
            
  rowKeepList = pd.concat(rowKeepList).reset_index(drop = True)
  analytic(args, rowKeepList)
  rowKeepList.to_pickle(os.path.join(outDir, outFolder, f"{outPkl}.pkl.gz"), compression = "gzip")
  return rowKeepList