#######################################################################################################################
## -- necessary libraries -- ##########################################################################################
#######################################################################################################################
import re
from ..FILE import *

#######################################################################################################################
## -- loads data for an entire trial as an object of different phases -- ##############################################
#######################################################################################################################
def loadData(args, directory):

  bioFileName = sorted([item for item in dirSweep(directory) if not "aph" in item])
  accFileName = sorted([item for item in dirSweep(directory) if "aph" in item])
  bioDataDir = [f"{directory}/{item}" for item in bioFileName]
  accDataDir = [f"{directory}/{item}" for item in accFileName]
  subArgsObj = args["generic"]
  
  trialDataPackage, log, forward = {}, [], 0
  for bioFileId, bioFile in enumerate(bioFileName):
    
    matchingId = False
    for accFileId, accFile in enumerate(accFileName):
      
      bioId = re.search(r'\d+', bioFileName[bioFileId].split('_', 1)[0]).group()
      accId = re.search(r'\d+', accFileName[accFileId].split('_', 1)[0]).group()
      matchingId, phaseId = (int(bioId) == int(accId)), int(bioId)

      if matchingId == True:
        bioData = fLoadMat(bioDataDir[bioFileId])
        accData = fLoadCSV(accDataDir[accFileId])
        dataObjectPackage, finapres = parseDat(args, bioData, accData)
        trialDataPackage.update({
          f"phaseId{phaseId:02}": {
            "phId": phaseId,
            "desc": subArgsObj["phaseDescription"][f"{phaseId}"],
            "data": dataObjectPackage,
            "fina": None,
            "calQ": None,
            "wfqm": None,
            "prep": None,
          }
        })
        log.append(f"Success: bioFile {bioFileName[bioFileId]}, has a matching accFile")
        break
    
    if matchingId == False:
      log.append(f"Warning: bioFile {bioFileName[bioFileId]}, does not have a matching accFile")
  
  trialDataPackage = {
    "bioData": trialDataPackage,
    "finData": finapres,
  }
  return trialDataPackage, log