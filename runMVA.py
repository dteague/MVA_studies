#!/usr/bin/env python
import os
import argparse
import numpy as np
from collections import OrderedDict

from dylan_utils import FileInfo
import dylan_utils.configs as config
from Utilities.MvaMaker import XGBoostMaker
from Utilities.MVAPlotter import MVAPlotter

##################
# User Variables #
##################

# Variables used in Training
usevar = {
    "NJets": "num('Jets/Jet_pt')",
    "NBJets": "num('BJets/Jet_pt')",
    "HT": "var('Event_variables/Event_HT')",
    "MET": "var('Event_MET/MET_pt')",
    "centrality": "var('Event_variables/Event_centrality')",
    "sphericity": "var('Event_variables/Event_sphericity')",
    "j1Pt": "nth('Jets/Jet_pt', 0)",
    "j2Pt": "nth('Jets/Jet_pt', 1)",
    "j3Pt": "nth('Jets/Jet_pt', 2)",
    "j4Pt": "nth('Jets/Jet_pt', 3)",
    "j5Pt": "nth('Jets/Jet_pt', 4)",
    "j6Pt": "nth('Jets/Jet_pt', 5)",
    "j7Pt": "nth('Jets/Jet_pt', 6)",
    "j8Pt": "nth('Jets/Jet_pt', 7)",
}

# Input Rootfile
INPUT_TREE = "inputTrees_new.root"

# Sampels and the groups they are a part of
groups = [["Signal", ["ttt"]],
          ["FourTop", ["4top2016", ]],
          ["Background", ["ttw", "ttz", "tth", "ttXY", "vvv", "vv",
                          "xg", "other"]]]



def get_com_args():
    parser = config.get_generic_args()
    parser.add_argument('-t', '--train', action="store_true",
                        help="Run the training")
    parser.add_argument("-i", "--indir", type=str, required=True,
                        help="Input root file (output of makeHistFile.py)")
    return parser.parse_args()


def train(args, groupDict):
    mvaRunner = XGBoostMaker(usevar)
    for groupName, samples in groupDict.items():
        mvaRunner.add_group(groupName, samples, args.indir)
    fitModel = mvaRunner.train()
    fitModel.save_model("{}/model.bin".format(args.outdir))
    mvaRunner.output(args.outdir)

    
def get_group_dict(groups):
    groupDict = OrderedDict()
    for groupName, samples in groups:
        new_samples = list()
        for samp in samples:
            if samp in file_info.group2MemberMap:
                new_samples += file_info.group2MemberMap[samp]
            else:
                new_samples.append(samp)
        groupDict[groupName] = new_samples
    return groupDict

        
if __name__ == "__main__":
    args = get_com_args()
    file_info = FileInfo(**vars(args))
    groupDict = get_group_dict(groups)
    groupMembers = [item for sublist in groupDict.values() for item in sublist]
    lumi = args.lumi*1000
    if not os.path.isdir(args.outdir):
        os.mkdir(args.outdir)
        os.mkdir("{}/test".format(args.outdir))
        os.mkdir("{}/train".format(args.outdir))

    if args.train:
        train(args, groupDict)

    NBINS_2D = 50
    stob2d = np.linspace(0.0, 1.0, NBINS_2D + 1)
    stobBins = np.linspace(0.0, 1, 50)
    
    output = MVAPlotter(args.outdir, groupDict.keys(), groupMembers, lumi)
    # output.set_show(args.show)
    output.plot_all_shapes("NJets", np.linspace(0, 15, 16), "allGroups")
    output.plot_all_shapes("MET", np.linspace(0, 500, 50), "allGroups")
    output.plot_all_shapes("HT", np.linspace(0, 1500, 50), "allGroups")
    output.plot_fom("Signal", ["Background"], "BDT.Signal", stobBins, "")
    maxSBVal = output.plot_fom_2d("Signal", "BDT.Background", "BDT.FourTop", stob2d, stob2d)

    output.plot_func_2d("Signal", "BDT.Background", "BDT.FourTop",
                    stob2d, stob2d, "Signal", lines=maxSBVal[1:])
    output.plot_func_2d("Background", "BDT.Background", "BDT.FourTop",
                    stob2d, stob2d, "Background", lines=maxSBVal[1:])
    output.plot_func_2d("FourTop", "BDT.Background", "BDT.FourTop",
                    stob2d, stob2d, "FourTop", lines=maxSBVal[1:])
# ###############
# # Make Plots  #
# ###############
#
#
#

#

#
# gSet = output.get_sample()

# output.write_out("preSelection_BDT.2020.07.14", INPUT_TREE)
#
# output.make_roc("Signal", ["FourTop", "Background"], "Signal", "SignalvsAll")
# output.print_info("BDT.Signal", groupMembers)

# print("FourTop: ", output.approx_likelihood("Signal", ["Background", "FourTop"],
#                                             "BDT.FourTop", stobBins))
# print("Background: ", output.approx_likelihood("Signal", ["Background", "FourTop"],
#                                                "BDT.Background", stobBins))





# # output.apply_cut("BDT.FourTop>{}".format(maxSBVal[2]))
# # output.apply_cut("BDT.Background>{}".format(maxSBVal[1]))

# # output.write_out("postSelection_BDT.2020.06.03_SignalSingle")
