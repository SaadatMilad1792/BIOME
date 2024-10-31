#######################################################################################################################
## -- necessary libraries -- ##########################################################################################
#######################################################################################################################
import pandas as pd

#######################################################################################################################
## -- quality filter sweep -- #########################################################################################
#######################################################################################################################
def qFilters(args, dataFrame):
  
  mappingArg, genericArg = args["mapping"]["qFilters"], args["generic"]
  pThreshold, sThreshold = mappingArg["pThreshold"], mappingArg["sThreshold"]
  rThreshold, finCalStat = mappingArg["rThreshold"], mappingArg["finCalStat"]
  filterStUp = mappingArg["filterStUp"]
  
  newDataFrame = dataFrame[ (dataFrame['wfqm'].apply(lambda x: x[0] < rThreshold))
                          & (dataFrame['wfqm'].apply(lambda x: x[1] > pThreshold))
                          & (dataFrame['wfqm'].apply(lambda x: x[2] > sThreshold))
                          & (dataFrame["trial"].str.contains(filterStUp))
                          & (dataFrame['prep'] == finCalStat)]
  
  return newDataFrame