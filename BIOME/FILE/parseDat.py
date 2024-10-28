#######################################################################################################################
## -- necessary libraries -- ##########################################################################################
#######################################################################################################################
import numpy as np

#######################################################################################################################
## -- parses data from mat file, creates new object packages for contents -- ##########################################
#######################################################################################################################
def parseDat(args, bioData, accData):

  ## -- argument handling -- ##
  subArgsObj = args["process"]["parseDat"]
  resampleActive, resampleRate = subArgsObj["resampleActive"], subArgsObj["resampleRate"]
  
  ## -- channel 1 data handling -- ##
  ch1_abs = np.array((bioData["data"])[(bioData["data"])["bioz"]["z"][0, 0]][()]).reshape(-1)
  ch1_phi = np.array((bioData["data"])[(bioData["data"])["bioz"]["p"][0, 0]][()]).reshape(-1)
  ch1_tim = np.array((bioData["data"])[(bioData["data"])["bioz"]["t"][0, 0]][()]).reshape(-1)
  ch1_del = np.array((bioData["data"])[(bioData["data"])["bioz"]["dz"][0, 0]][()]).reshape(-1)
  ch1_tdr = np.array((bioData["data"])[(bioData["data"])["bioz"]["t_dr"][0, 0]][()]).reshape(-1)
  
  ## -- channel 2 data handling -- ##
  ch2_abs = np.array((bioData["data"])[(bioData["data"])["bioz"]["z"][1, 0]][()]).reshape(-1)
  ch2_phi = np.array((bioData["data"])[(bioData["data"])["bioz"]["p"][1, 0]][()]).reshape(-1)
  ch2_tim = np.array((bioData["data"])[(bioData["data"])["bioz"]["t"][1, 0]][()]).reshape(-1)
  ch2_del = np.array((bioData["data"])[(bioData["data"])["bioz"]["dz"][1, 0]][()]).reshape(-1)
  ch2_tdr = np.array((bioData["data"])[(bioData["data"])["bioz"]["t_dr"][1, 0]][()]).reshape(-1)
  
  ## -- ppg data handling -- ##
  ppg_mag = np.array((bioData["data"])["ppg"]["dv"][()]).reshape(-1)
  ppg_tim = np.array((bioData["data"])["ppg"]["t_dv"][()]).reshape(-1)

  ## -- finapres 're' and 'fi' files -- ##
  fre_tim = np.array((bioData["data"])["fp"]["reSYS"]["time"][()]).reshape(-1)
  fre_sys = np.array((bioData["data"])["fp"]["reSYS"]["values"][()]).reshape(-1)
  fre_dia = np.array((bioData["data"])["fp"]["reDIA"]["values"][()]).reshape(-1)
  ffi_tim = np.array((bioData["data"])["fp"]["fiSYS"]["time"][()]).reshape(-1)
  ffi_sys = np.array((bioData["data"])["fp"]["fiSYS"]["values"][()]).reshape(-1)
  ffi_dia = np.array((bioData["data"])["fp"]["fiDIA"]["values"][()]).reshape(-1)
  
  ## -- accelerometer data handling -- ##
  acc_tim = accData[" Time (s)"]
  acc_xax = accData[" X-axis"]
  acc_yax = accData[" Y-axis"]
  acc_zax = accData[" Z-axis"]
  acc_mag = accData[" Magnitude "]
  
  ## -- segmentation performed on channel 1 and extract characteristic points [IMPORTANT PREPROCESSING PROCEDURE] -- ##
  from ..PROCESS import segments, reSample
  biozCharacteristicPoints = segments(ch1_tdr, ch1_del)
  maxSlope = biozCharacteristicPoints["maxSlope"]
  maxCrest = biozCharacteristicPoints["maxCrest"]
  minCrest = biozCharacteristicPoints["minCrest"]
  numCycle = biozCharacteristicPoints["numCycle"]
  
  ## -- channel 1 segmentation -- ##
  ch1_tim = [reSample(ch1_tim[maxCrest[i] : maxCrest[i + 1]], resampleRate, resampleActive) for i in range(numCycle)]
  ch1_phi = [reSample(ch1_phi[maxCrest[i] : maxCrest[i + 1]], resampleRate, resampleActive) for i in range(numCycle)]
  ch1_abs = [reSample(ch1_abs[maxCrest[i] : maxCrest[i + 1]], resampleRate, resampleActive) for i in range(numCycle)]
  ch1_del = [reSample(ch1_del[maxCrest[i] : maxCrest[i + 1]], resampleRate, resampleActive) for i in range(numCycle)]
  ch1_icg = [reSample(np.gradient(ch1_del[i], ch1_tim[i]), resampleRate, resampleActive) for i in range(numCycle)]
  
  ## -- channel 2 segmentation -- ##
  ch2_tim = [reSample(ch2_tim[maxCrest[i] : maxCrest[i + 1]], resampleRate, resampleActive) for i in range(numCycle)]
  ch2_phi = [reSample(ch2_phi[maxCrest[i] : maxCrest[i + 1]], resampleRate, resampleActive) for i in range(numCycle)]
  ch2_abs = [reSample(ch2_abs[maxCrest[i] : maxCrest[i + 1]], resampleRate, resampleActive) for i in range(numCycle)]
  ch2_del = [reSample(ch2_del[maxCrest[i] : maxCrest[i + 1]], resampleRate, resampleActive) for i in range(numCycle)]
  ch2_icg = [reSample(np.gradient(ch2_del[i], ch2_tim[i]), resampleRate, resampleActive) for i in range(numCycle)]
    
  ## -- finapres adc values -- ##
  fp_abp_adc = np.array((bioData["data"])["fp"]["abp"]["adc"][()]).reshape(-1)
  fp_abp_tim = np.array((bioData["data"])["fp"]["abp"]["t"][()]).reshape(-1)
  
  ## -- ppg segmentation -- ##
  ppg_mag = [reSample(ppg_mag[maxCrest[i] : maxCrest[i + 1]], resampleRate, resampleActive) for i in range(numCycle)]
  ppg_tim = [reSample(ppg_tim[maxCrest[i] : maxCrest[i + 1]], resampleRate, resampleActive) for i in range(numCycle)]
  
  ## -- adc segmentation -- ##
  fp_abp_adc = [reSample(fp_abp_adc[(ch1_tim[i][0] <= fp_abp_tim) & (fp_abp_tim <= ch1_tim[i][-1])], resampleRate, resampleActive) for i in range(numCycle)]
  fp_abp_tim = [reSample(fp_abp_tim[(ch1_tim[i][0] <= fp_abp_tim) & (fp_abp_tim <= ch1_tim[i][-1])], resampleRate, resampleActive) for i in range(numCycle)]
  
  ## -- accelerometer segmentation -- ##
  acc_xax = np.array([reSample(acc_xax[(ch1_tim[i][0] <= acc_tim) & (acc_tim <= ch1_tim[i][-1])], resampleRate, resampleActive) for i in range(numCycle)])
  acc_yax = np.array([reSample(acc_yax[(ch1_tim[i][0] <= acc_tim) & (acc_tim <= ch1_tim[i][-1])], resampleRate, resampleActive) for i in range(numCycle)])
  acc_zax = np.array([reSample(acc_zax[(ch1_tim[i][0] <= acc_tim) & (acc_tim <= ch1_tim[i][-1])], resampleRate, resampleActive) for i in range(numCycle)])
  acc_mag = np.array([reSample(acc_mag[(ch1_tim[i][0] <= acc_tim) & (acc_tim <= ch1_tim[i][-1])], resampleRate, resampleActive) for i in range(numCycle)])
  acc_tim = np.array([reSample(acc_tim[(ch1_tim[i][0] <= acc_tim) & (acc_tim <= ch1_tim[i][-1])], resampleRate, resampleActive) for i in range(numCycle)])
  
  ## -- creating data dictionaries (channel 1) -- ##
  channel1 = {
    "abs": ch1_abs,
    "phi": ch1_phi,
    "del": ch1_del,
    "icg": ch1_icg,
    "tim": ch1_tim,
  }
  
  ## -- creating data dictionaries (channel 2) -- ##
  channel2 = {
    "abs": ch2_abs,
    "phi": ch2_phi,
    "del": ch2_del,
    "icg": ch2_icg,
    "tim": ch2_tim,
  }
  
  ## -- creating data dictionaries (ppg) -- ##
  ppgData = {
    "mag": ppg_mag,
    "tim": ppg_tim,
  }
  
  ## -- creating data dictionaries (accelerometer) -- ##
  accelerometer = { 
    "tim": acc_tim,
    "xax": acc_xax,
    "yax": acc_yax,
    "zax": acc_zax,
    "mag": acc_mag,
  }
  
  ## -- creating data dictionaries (finapres) -- ##
  finapres = {
    "fp_adc_val": fp_abp_adc,
    "fp_adc_tim": fp_abp_tim,
    "re_tim": fre_tim,
    "re_sys": fre_sys,
    "re_dia": fre_dia,
    "fi_tim": ffi_tim,
    "fi_sys": ffi_sys,
    "fi_dia": ffi_dia,
  }
  
  ## -- creating data object package (data) -- ##
  dataObjectPackage = {
    "ppgData": ppgData,
    "channel1": channel1,
    "channel2": channel2,
    "accmeter": accelerometer,
    "biozCharacteristicPoints": biozCharacteristicPoints,
  }
  
  return dataObjectPackage, finapres