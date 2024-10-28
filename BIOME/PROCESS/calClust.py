#######################################################################################################################
## -- necessary libraries -- ##########################################################################################
#######################################################################################################################
import sys

#######################################################################################################################
## -- finds the clusters of calibrations to pinpoint maneuvers -- #####################################################
#######################################################################################################################
def calClust(data, calValue = 500, mode = "cal"):
  calClusters, calCluster = [], []
  
  for i, value in enumerate(data):
    if (value >= calValue if mode == "cal" else value < calValue if mode == "sig" else sys.exit("Modes: ['cal', 'sig']")):
      calCluster.append(i)
    elif calCluster:
      calClusters.append(calCluster)
      calCluster = []

  if calCluster:
    calClusters.append(calCluster)

  return len(calClusters), calClusters