#######################################################################################################################
## -- necessary libraries -- ##########################################################################################
#######################################################################################################################
import sys
import numpy as np
from ..FILE import *
import scipy.signal as sg

from bokeh.layouts import column
from bokeh.io import output_notebook
from bokeh.plotting import figure, output_notebook, show

#######################################################################################################################
## -- parses data from mat file, creates new object packages for contents -- ##########################################
#######################################################################################################################
def segments(time, bioz):

  ## -- generate robust max slope points (second degree local minima) -- ##
  gradientSg = np.gradient(bioz, time)
  LocalMin = sg.argrelextrema(gradientSg, np.less)[0]
  minIndex = sg.argrelextrema(gradientSg[LocalMin], np.less)
  locSlope = LocalMin[minIndex]
  
  ## -- sys and dia max and min points calculation based on max slope -- ##
  maxCrest, maxSlope, minCrest, stableData = [], [], [], 0
  local_max, LocalMin = sg.argrelextrema(bioz, np.greater)[0], sg.argrelextrema(bioz, np.less)[0]
  for i in range(1, len(locSlope) - 1):
    C1 = len(local_max[(local_max > locSlope[i - 1]) & (local_max < locSlope[i])]) > 0
    C2 = len(LocalMin[(LocalMin > locSlope[i]) & (LocalMin < locSlope[i + 1])]) > 0
    if C1 and C2:
      maxCrest.append(np.max(local_max[(local_max > locSlope[i - 1]) & (local_max < locSlope[i])]))
      minCrest.append(np.min(LocalMin[(LocalMin > locSlope[i]) & (LocalMin < locSlope[i + 1])]))
      maxSlope.append(locSlope[i])
      # print(maxCrest[-1], minCrest[-1], maxSlope[-1], stableData)
      
    ## -- 10% of the trial size as stability margin -- ##
    elif i < int(0.1 * len(locSlope)):
      stableData = stableData + 1
      maxCrest, maxSlope, minCrest = [], [], []
      
  ## -- max slope outlier removal and sanity check (lower edge) -- ##
  IBI_df1, mask = np.diff(maxSlope), []
  df1_avg, df1_std = np.mean(IBI_df1), np.std(IBI_df1)
  
  consecutive = False
  for i in range(1, len(maxSlope) - 1):

    df21 = maxSlope[i + 0] - maxSlope[i - 1]
    df32 = maxSlope[i + 1] - maxSlope[i - 0]
    df31 = maxSlope[i + 1] - maxSlope[i - 1]
    if (df21 < df1_avg - df1_std) and (df32 < df1_avg - df1_std) and (consecutive == False):
      consecutive = True
      mask.append(i)
    elif (df21 < df1_avg - df1_std) and (df31 < df1_avg + df1_std):
      mask.append(i)
    else:
      consecutive = False
      
  maxCrest, maxSlope, minCrest = np.delete(maxCrest, mask), np.delete(maxSlope, mask), np.delete(minCrest, mask)
  
  ## -- max slope outlier removal and sanity check (higher edge) -- ##
  IBI_df1 = np.diff(maxSlope)
  df1_avg, df1_std = np.mean(IBI_df1), np.std(IBI_df1)
  maxs, maxc, minc, mask = [], [], [], []
  
  for i in range(1, len(maxSlope)):
      
    df21 = maxSlope[i] - maxSlope[i - 1]
    if (df21 > 2 * (df1_avg - df1_std)):
      if i > 1:
        maxc.append(int((maxCrest[i] + maxCrest[i - 1]) / 2))
        maxs.append(int((maxSlope[i] + maxSlope[i - 1]) / 2))
        minc.append(int((minCrest[i] + minCrest[i - 1]) / 2))
      else:
        mask.append(i - 1)

  maxCrest = np.sort(np.concatenate((np.delete(maxCrest, mask), maxc))).astype(int)
  maxSlope = np.sort(np.concatenate((np.delete(maxSlope, mask), maxs))).astype(int)
  minCrest = np.sort(np.concatenate((np.delete(minCrest, mask), minc))).astype(int)
  biozCharacteristicPoints = {
    "maxCrest": maxCrest,
    "maxSlope": maxSlope,
    "minCrest": minCrest,
    "numCycle": len(maxSlope) - 1,
    "pinPoint": {
      "maxCrest": {"val": bioz[maxCrest], "tim": time[maxCrest]},
      "maxSlope": {"val": bioz[maxSlope], "tim": time[maxSlope]},
      "minCrest": {"val": bioz[minCrest], "tim": time[minCrest]},
    }
  }
  
  # print(len(maxCrest), len(maxSlope), len(minCrest))
    
  # output_notebook()
  # plot = figure(title = "bioz and gradientSg", x_axis_label = 'time', y_axis_label = 'Value', plot_width = 800, plot_height = 400)
  # plot.scatter(time[maxCrest], (bioz[maxCrest] - np.min(bioz)) / (np.max(bioz) - np.min(bioz)), color = "orange", legend_label = "maxCrest")
  # plot.scatter(time[maxSlope], (bioz[maxSlope] - np.min(bioz)) / (np.max(bioz) - np.min(bioz)), color = "red", legend_label = "maxSlope")
  # plot.scatter(time[minCrest], (bioz[minCrest] - np.min(bioz)) / (np.max(bioz) - np.min(bioz)), color = "green", legend_label = "minCrest")
  # plot.line(time, (bioz - np.min(bioz)) / (np.max(bioz) - np.min(bioz)), line_width = 2, legend_label = "bioz")
  # # plot.line(time, (gradientSg - np.min(gradientSg)) / (np.max(gradientSg) - np.min(gradientSg)), line_width = 2, line_color = "orange", legend_label = "gradientSg")
  # plot.legend.location = "top_left"
  # show(column(plot))
  # sys.exit()
  
  return biozCharacteristicPoints