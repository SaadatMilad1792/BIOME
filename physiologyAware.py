## -- import BIOME -- ##
import BIOME

## -- load and assign arguments -- ##
args = BIOME.fLoadArg("./args.yaml")


## -- BIOME -> loadStUp(args) -- ##
# args["process"]["loadStUp"]["subjectList"] = ["P034"]
args["process"]["loadStUp"]["loadStUpActive"] = False
args["process"]["loadStUp"]["savePickles"] = False
args["process"]["loadStUp"]["savePltOverall"] = False
gbdf = BIOME.loadStUp(args)

## -- BIOME -> phyAware(args) -- ##
# args["mapping"]["phyAware"]["subjectList"] = ["P034"]
args["mapping"]["phyAware"]["phyAwareActive"] = True
args["mapping"]["phyAware"]["savePytorch"] = False
args["mapping"]["phyAware"]["saveReportLogs"] = False
BIOME.phyAware(args)