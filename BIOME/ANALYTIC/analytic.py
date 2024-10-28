#######################################################################################################################
## -- 1. libraries and essential packages -- ##########################################################################
#######################################################################################################################

## -- 1.1: essential python libraries -- ##
import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..PROCESS import *

#######################################################################################################################
## -- 2. data frame quality visual module -- ##########################################################################
#######################################################################################################################
def analytic(args, df):
  uniqueSub = list(df["subject"].unique())
  for sub in uniqueSub:
    dataQualVis(args, df[df["subject"] == sub], sub)
  
  return "Task Completed!"

#######################################################################################################################
## -- 3. data frame quality visual sub module -- ######################################################################
#######################################################################################################################
def dataQualVis(args, df, subject):
  
  PD = args["generic"]["phaseDescription"]
  trial, trial_count = df["trial"].unique(), len(df["trial"].unique())
  fig, ax = plt.subplots(6, trial_count, figsize = (8 * trial_count, 36), dpi = 200, facecolor = "white")
  
  if trial_count == 1:
    ax = np.expand_dims(ax, axis=1)
      
  ## -- 2.1: accelerometer magnitude plots -- ##
  for i, t in enumerate(trial):
    phases = df[df["trial"] == t]["phase"].unique()
    phases = sorted(phases, key = lambda x: int(x[2:]))
    doOnce = True
    for p in phases:
      CPD = PD[f"{p[3:]}"]
      g_tim = np.concatenate(np.array(df.loc[(df["phase"] == p) & (df["trial"] == t), "acc"].apply(lambda x: x["tim"])))
      g_mag = np.concatenate(np.array(df.loc[(df["phase"] == p) & (df["trial"] == t), "acc"].apply(lambda x: x["mag"])))
      
      if doOnce:
        doOnce = False
        fin_tim = np.array(df.loc[(df["phase"] == p) & (df["trial"] == t), "fin"].apply(lambda x: x["fp_tim"]))[0]
        fin_dbp = np.array(df.loc[(df["phase"] == p) & (df["trial"] == t), "fin"].apply(lambda x: x["fp_dia"]))[0]
        calCount, calIntervals = calClust(fin_dbp)
        for calSpan in calIntervals:
          calTimes = fin_tim[calSpan]
          ax[0, i].axvspan(calTimes[0], calTimes[-1], color = "gray", alpha = 0.3)
          ax[0, i].axvline(x = calTimes[0], color = "black", linestyle = "--", linewidth = 1.5)
          ax[0, i].axvline(x = calTimes[-1], color = "black", linestyle = "--", linewidth = 1.5)
      
      ax[0, i].plot(g_tim, g_mag, label = f"{p}, {CPD}")
      Quality = df[df["trial"] == t]["prep"].unique()
      ax[0, i].set_title(f"Trial: {t}, Quality: {Quality}", fontsize = 20)
      ax[0, i].set_xlabel("Time (s)", fontsize = 12)
      ax[0, i].set_ylabel("Accelerometer Magnitude", fontsize = 12)
      ax[0, i].legend(fontsize = 4, loc="lower right")
      ax[0, i].grid(True)
      
  ## -- 2.2: IBI plots -- ##
  for i, t in enumerate(trial):
    phases = df[df["trial"] == t]["phase"].unique()
    phases = sorted(phases, key = lambda x: int(x[2:]))
    doOnce = True
    for p in phases:
      CPD = PD[f"{p[3:]}"]
      g_tim = np.concatenate(np.array(df.loc[(df["phase"] == p) & (df["trial"] == t), "ch1"].apply(lambda x: [x["tim"][-1]])))
      g_mag = np.concatenate(np.array(df.loc[(df["phase"] == p) & (df["trial"] == t), "ch1"].apply(lambda x: [(x["tim"])[:len(np.trim_zeros(x["del"], 'b'))][-1] - (x["tim"])[:len(np.trim_zeros(x["del"], 'b'))][0]])))
      
      if doOnce:
        doOnce = False
        fin_tim = np.array(df.loc[(df["phase"] == p) & (df["trial"] == t), "fin"].apply(lambda x: x["fp_tim"]))[0]
        fin_dbp = np.array(df.loc[(df["phase"] == p) & (df["trial"] == t), "fin"].apply(lambda x: x["fp_dia"]))[0]
        calCount, calIntervals = calClust(fin_dbp)
        for calSpan in calIntervals:
          calTimes = fin_tim[calSpan]
          ax[1, i].axvspan(calTimes[0], calTimes[-1], color = "gray", alpha = 0.3)
          ax[1, i].axvline(x = calTimes[0], color = "black", linestyle = "--", linewidth = 1.5)
          ax[1, i].axvline(x = calTimes[-1], color = "black", linestyle = "--", linewidth = 1.5)
      
      ax[1, i].plot(g_tim, g_mag, label = f"{p}, {CPD}")
      Quality = df[df["trial"] == t]["prep"].unique()
      ax[1, i].set_title(f"Trial: {t}, Quality: {Quality}", fontsize = 20)
      ax[1, i].set_xlabel("Time (s)", fontsize = 12)
      ax[1, i].set_ylabel("IBI BioZ", fontsize = 12)
      ax[1, i].legend(fontsize = 4, loc="lower right")
      ax[1, i].grid(True)
      
  ## -- 2.3: SBP plots -- ##
  for i, t in enumerate(trial):
    phases = df[df["trial"] == t]["phase"].unique()
    phases = sorted(phases, key = lambda x: int(x[2:]))
    doOnce = True
    for p in phases:
      CPD = PD[f"{p[3:]}"]
      g_tim = np.array(df.loc[(df["phase"] == p) & (df["trial"] == t), "fin"].apply(lambda x: x["re_tim"]))
      g_mag = np.array(df.loc[(df["phase"] == p) & (df["trial"] == t), "fin"].apply(lambda x: x["re_sys"]))
      
      if doOnce:
        doOnce = False
        fin_tim = np.array(df.loc[(df["phase"] == p) & (df["trial"] == t), "fin"].apply(lambda x: x["fp_tim"]))[0]
        fin_dbp = np.array(df.loc[(df["phase"] == p) & (df["trial"] == t), "fin"].apply(lambda x: x["fp_dia"]))[0]
        calCount, calIntervals = calClust(fin_dbp)
        for calSpan in calIntervals:
          calTimes = fin_tim[calSpan]
          ax[2, i].axvspan(calTimes[0], calTimes[-1], color = "gray", alpha = 0.3)
          ax[2, i].axvline(x = calTimes[0], color = "black", linestyle = "--", linewidth = 1.5)
          ax[2, i].axvline(x = calTimes[-1], color = "black", linestyle = "--", linewidth = 1.5)
      
      ax[2, i].plot(g_tim, g_mag, label = f"{p} SBP, {CPD}")
      Quality = df[df["trial"] == t]["prep"].unique()
      ax[2, i].set_title(f"Trial: {t}, Quality: {Quality}", fontsize = 20)
      ax[2, i].set_xlabel("Time (s)", fontsize = 12)
      ax[2, i].set_ylabel("Finapres SBP (mmHg)", fontsize = 12)
      # ax[2, i].set_ylim(top = 300)
      ax[2, i].legend(fontsize = 4, loc="lower right")
      ax[2, i].grid(True)
      
  ## -- 2.4: DBP plots -- ##
  for i, t in enumerate(trial):
    phases = df[df["trial"] == t]["phase"].unique()
    phases = sorted(phases, key = lambda x: int(x[2:]))
    doOnce = True
    for p in phases:
      CPD = PD[f"{p[3:]}"]
      g_tim = np.array(df.loc[(df["phase"] == p) & (df["trial"] == t), "fin"].apply(lambda x: x["re_tim"]))
      g_mag = np.array(df.loc[(df["phase"] == p) & (df["trial"] == t), "fin"].apply(lambda x: x["re_dia"]))
      
      if doOnce:
        doOnce = False
        fin_tim = np.array(df.loc[(df["phase"] == p) & (df["trial"] == t), "fin"].apply(lambda x: x["fp_tim"]))[0]
        fin_dbp = np.array(df.loc[(df["phase"] == p) & (df["trial"] == t), "fin"].apply(lambda x: x["fp_dia"]))[0]
        calCount, calIntervals = calClust(fin_dbp)
        for calSpan in calIntervals:
          calTimes = fin_tim[calSpan]
          ax[3, i].axvspan(calTimes[0], calTimes[-1], color = "gray", alpha = 0.3)
          ax[3, i].axvline(x = calTimes[0], color = "black", linestyle = "--", linewidth = 1.5)
          ax[3, i].axvline(x = calTimes[-1], color = "black", linestyle = "--", linewidth = 1.5)

      ax[3, i].plot(g_tim, g_mag, label = f"{p} DBP, {CPD}")
      Quality = df[df["trial"] == t]["prep"].unique()
      ax[3, i].set_title(f"Trial: {t}, Quality: {Quality}", fontsize = 20)
      ax[3, i].set_xlabel("Time (s)", fontsize = 12)
      ax[3, i].set_ylabel("Finapres DBP (mmHg)", fontsize = 12)
      # ax[3, i].set_ylim(top = 300)
      ax[3, i].legend(fontsize = 4, loc="lower right")
      ax[3, i].grid(True)
      
  ## -- 2.5: BioZ absolute plots -- ##
  for i, t in enumerate(trial):
    phases = df[df["trial"] == t]["phase"].unique()
    phases = sorted(phases, key = lambda x: int(x[2:]))
    doOnce = True
    for p in phases:
      CPD = PD[f"{p[3:]}"]
      g_tim = np.concatenate(np.array(df.loc[(df["phase"] == p) & (df["trial"] == t), "ch1"].apply(lambda x: (x["tim"])[:len(np.trim_zeros(x["del"], "b")) - 1])))
      g_mag = np.concatenate(np.array(df.loc[(df["phase"] == p) & (df["trial"] == t), "ch1"].apply(lambda x: np.trim_zeros(x["abs"], "b")[:-1])))
      
      if doOnce:
        doOnce = False
        fin_tim = np.array(df.loc[(df["phase"] == p) & (df["trial"] == t), "fin"].apply(lambda x: x["fp_tim"]))[0]
        fin_dbp = np.array(df.loc[(df["phase"] == p) & (df["trial"] == t), "fin"].apply(lambda x: x["fp_dia"]))[0]
        calCount, calIntervals = calClust(fin_dbp)
        for calSpan in calIntervals:
          calTimes = fin_tim[calSpan]
          ax[4, i].axvspan(calTimes[0], calTimes[-1], color = "gray", alpha = 0.3)
          ax[4, i].axvline(x = calTimes[0], color = "black", linestyle = "--", linewidth = 1.5)
          ax[4, i].axvline(x = calTimes[-1], color = "black", linestyle = "--", linewidth = 1.5)
      
      ax[4, i].plot(g_tim, g_mag, label = f"{p}, {CPD}")
      Quality = df[df["trial"] == t]["prep"].unique()
      ax[4, i].set_title(f"Trial: {t}, Quality: {Quality}", fontsize = 20)
      ax[4, i].set_xlabel("Time (s)", fontsize = 12)
      ax[4, i].set_ylabel("Absolute BioZ (Ohms)", fontsize = 12)
      ax[4, i].legend(fontsize = 4, loc="lower right")
      ax[4, i].grid(True)
      
  ## -- 2.6: BioZ delta plots -- ##
  for i, t in enumerate(trial):
    phases = df[df["trial"] == t]["phase"].unique()
    phases = sorted(phases, key = lambda x: int(x[2:]))
    doOnce = True
    for p in phases:
      CPD = PD[f"{p[3:]}"]
      g_tim = np.concatenate(np.array(df.loc[(df["phase"] == p) & (df["trial"] == t), "ch1"].apply(lambda x: (x["tim"])[:len(np.trim_zeros(x["del"], "b"))])))
      g_mag = np.concatenate(np.array(df.loc[(df["phase"] == p) & (df["trial"] == t), "ch1"].apply(lambda x: np.trim_zeros(x["del"], "b"))))
      
      if doOnce:
        doOnce = False
        fin_tim = np.array(df.loc[(df["phase"] == p) & (df["trial"] == t), "fin"].apply(lambda x: x["fp_tim"]))[0]
        fin_dbp = np.array(df.loc[(df["phase"] == p) & (df["trial"] == t), "fin"].apply(lambda x: x["fp_dia"]))[0]
        calCount, calIntervals = calClust(fin_dbp)
        for calSpan in calIntervals:
          calTimes = fin_tim[calSpan]
          ax[5, i].axvspan(calTimes[0], calTimes[-1], color = "gray", alpha = 0.3)
          ax[5, i].axvline(x = calTimes[0], color = "black", linestyle = "--", linewidth = 1.5)
          ax[5, i].axvline(x = calTimes[-1], color = "black", linestyle = "--", linewidth = 1.5)
      
      ax[5, i].plot(g_tim, g_mag, label = f"{p}, {CPD}")
      Quality = df[df["trial"] == t]["prep"].unique()
      ax[5, i].set_title(f"Trial: {t}, Quality: {Quality}", fontsize = 20)
      ax[5, i].set_xlabel("Time (s)", fontsize = 12)
      ax[5, i].set_ylabel("Delta BioZ (Ohms)", fontsize = 12)
      ax[5, i].legend(fontsize = 4, loc="lower right")
      ax[5, i].grid(True)

  plt.tight_layout()
  os.makedirs("./Analytics", exist_ok = True)
  plt.savefig(f"./Analytics/SUBJECT_{subject}_OVERALL.png")
  plt.close()
  # plt.show()