#######################################################################################################################
## -- necessary libraries -- ##########################################################################################
#######################################################################################################################
import yaml

#######################################################################################################################
## -- parameter (args) loader -- ######################################################################################
#######################################################################################################################
def fLoadArg(directory):
  with open(directory, "r") as file:
    return yaml.load(file, Loader = yaml.FullLoader)