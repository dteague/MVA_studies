"""
.. module:: MVAPlotter
   :synopsis: Takes MvaMaker output and creates graphs
.. moduleauthor:: Dylan Teague
"""

import sys
import math
import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import xgboost as xgb
# import shap


class MVAPlotter(object):
    def __init__(self, workdir, groups, lumi=140000, is_train=False):
        self.groups = list(groups)
        self.do_show = False
        self.save_dir = workdir
        self.lumi=lumi
        colors = ['#f15854', '#faa43a', '#60bd68', '#5da5da']
        #  ['#CC0000', '#99FF00', '#FFCC00', '#3333FF']
        #  ['#462066', '#FFB85F', '#FF7A5A', '#00AAA0']

        self.color_dict = dict()
        for i, group in enumerate(self.groups):
            self.color_dict[group] = colors[i]
        self.color_dict["all"] = colors[len(self.groups)]

        if is_train:
            self.work_set = pd.read_pickle("{}/trainTree.pkl.gz".format(workdir))
            other = pd.read_pickle("{}/testTree.pkl.gz".format(workdir))
        else:
            self.work_set = pd.read_pickle("{}/testTree.pkl.gz".format(workdir))
            other = pd.read_pickle("{}/trainTree.pkl.gz".format(workdir))

        self.train_test_ratio = 1 + 1.0*len(other)/len(self.work_set)
        self.work_set.insert(1, "finalWeight", self.train_test_ratio * lumi * self.work_set["newWeight"])

        # was this multilcass or multiple binaries (maybe outdated)
        multirun = all(elem in self.work_set for elem in self.groups)

        for group in self.groups:
            if group not in self.work_set:
                continue
            if multirun and group != "Signal":
                self.work_set.insert(1, "BDT.{}".format(group),
                                     1-self.work_set[group])
            else:
                self.work_set.insert(1, "BDT.{}".format(group),
                                     self.work_set[group])
            self.work_set = self.work_set.drop(columns=group)

    def __len__(self):
        return len(self.groups)

    def set_show(self, do_show):
        """Set do_show to a value

        :param do_show: If Matplotlib show show plots
        :type do_show: bool
        """
        self.do_show = do_show

    def add_variable(self, name, arr):
        """ Add variable to MVAPlotter object
        (nice for plotting or moving to root based code)
        """
        if len(arr) != len(self.work_set):
            print("bad!")
            sys.exit()
        self.work_set.insert(1, name, arr)

    def apply_cut_max_bdt(self, sample, is_max=False):
        """
        Cut on file by with sample has largest BDT value
        """
        bdt = list()
        for group in self.groups:
            if group == "Signal":
                bdt.append(self.work_set["BDT.{}".format(group)])
            else:
                bdt.append(1-self.work_set["BDT.{}".format(group)])

        max_bdt = np.argmax(bdt, axis=0)
        if is_max:
            self.work_set = self.work_set[max_bdt == self.groups.index(sample)]
        else:
            self.work_set = self.work_set[max_bdt != self.groups.index(sample)]

    def apply_cut(self, cut):
        """
        Cut DataFrame using Root Style cut string
        """
        if cut.find("<") != -1:
            tmp = cut.split("<")
            self.work_set = self.work_set[self.work_set[tmp[0]] < float(tmp[1])]
        elif cut.find(">") != -1:
            tmp = cut.split(">")
            self.work_set = self.work_set[self.work_set[tmp[0]] > float(tmp[1])]
        elif cut.find("==") != -1:
            tmp = cut.split("==")
            self.work_set = self.work_set[self.work_set[tmp[0]] == float(tmp[1])]
        else:
            print("Problem!")
            sys.exit()

    def get_sample(self, groups=None):
        """
        Get DataFrame but requiring group to be one of the included in 'groups'
        """
        # Not list, just single instance
        if isinstance(groups, str):
            return self.work_set[self.work_set["classID"] == self.groups.index(groups)]
        elif not isinstance(groups, (list, np.ndarray)):
            groups = self.groups

        # if is a list
        final_set = pd.DataFrame(columns=self.work_set.columns)
        for group in groups:
            tmp_set = self.work_set[self.work_set["classID"] == self.groups.index(group)]
            final_set = pd.concat((final_set, tmp_set))
        return final_set

    def get_variable(self, var, groups=None):
        """
        Get numpy array of variable (under group contraint if wanted)
        """
        return self.get_sample(groups)[var]

    def get_hist(self, var, bins, groups=None):
        """
        Get numpy histogram of a variable
        """
        final_set = self.get_sample(groups)
        return np.histogram(final_set[var], bins=bins,
                            weights=final_set["finalWeight"])[0]

    def get_hist_err2(self, var, bins, groups=None):
        """
        Get numpy histogram of variable squared (poisson err if sqrt)
        """
        final_set = self.get_sample(groups)
        return np.histogram(final_set[var], bins=bins,
                            weights=final_set["finalWeight"]*final_set["finalWeight"])[0]

    def get_hist_2d(self, groups, var1, var2, bins):
        """
        Get numpy 2d histogram for 2 variables
        """
        final_set = self.get_sample(groups)
        return np.histogram2d(x=final_set[var1], y=final_set[var2], bins=bins,
                              weights=final_set["finalWeight"])[0]

    def get_hist_err2_2d(self, groups, var1, var2, bins):
        """
        Get numpy 2d histogram for 2 variables squared
        """
        final_set = self.get_sample(groups)
        return np.histogram2d(x=final_set[var1], y=final_set[var2], bins=bins,
                              weights=final_set["finalWeight"]*final_set["finalWeight"])[0]

    def plot_func(self, sig, bkg, var, bins, extra_name="", scale=True):
        """
        plot arbitrary variable
        """
        sig_hist = self.get_hist(var, bins, sig)
        bkg_hist = self.get_hist(var, bins, bkg)
        scale = findScale(max(sig_hist), max(bkg_hist)) if scale else 1.
        bkg_name = "all" if len(bkg) > 1 else bkg[0]
        sig_name = sig if scale == 1 else "{} x {}".format(sig, scale)
        if extra_name:
            extra_name = "_{}".format(extra_name)

        # Make plot
        fig, ax = plt.subplots()
        ax.hist(x=bins[:-1], weights=sig_hist*scale, bins=bins, label=sig_name,
                histtype="step", linewidth=1.5, color=self.color_dict[sig])
        ax.hist(x=bins[:-1], weights=bkg_hist, bins=bins, label=bkg_name,
                histtype="step", linewidth=1.5, color=self.color_dict[bkg_name])
        ax.legend()
        ax.set_xlabel(var)
        ax.set_ylabel("Events/bin")
        ax.set_title("Lumi = {} ifb".format(self.lumi/1000.))
        fig.tight_layout()
        plt.savefig("%s/%s%s.png" % (self.save_dir, var, extra_name))
        if self.do_show:
            plt.show()
        plt.close()

    def plot_func_2d(self, samples, var1, var2, bins1, bins2, name, lines=None):
        """plot 2 arbitrary variable (no differnt groups!)"""
        grp = self.get_sample(samples)
        fig, ax = plt.subplots()
        hist2d = ax.hist2d(grp[var1], grp[var2], [bins1, bins2],
                           weights=grp["finalWeight"], cmap=plt.cm.jet)
        if lines is not None:
            ax.plot([lines[0], lines[0]], [bins2[0],bins2[-1]], 'r-')
            ax.plot([bins1[0],bins1[-1]],  [lines[1], lines[1]], 'r-')

        fig.colorbar(hist2d[-1])
        ax.set_xlabel(var1)
        ax.set_ylabel(var2)
        ax.set_title("Lumi = {} ifb".format(self.lumi/1000.))
        fig.tight_layout()
        plt.savefig("%s/2D_%s.png" % (self.save_dir, name))
        if self.do_show:
            plt.show()
        plt.close()

    def get_fom(self, sig, bkg, var, bins, sb_denom=True, reverse=True):
        """Return FoM histogram"""
        drt = 1 if not reverse else -1
        sig_hist = self.get_hist(var, bins, sig)[::drt]
        bkg_hist = self.get_hist(var, bins, bkg)[::drt]
        n_sig = [np.sum(sig_hist[i:]) for i in range(len(bins))]
        if sb_denom:
            n_tot = [np.sum(bkg_hist[i:])+np.sum(sig_hist[i:])
                     for i in range(len(bins))]
        else:
            n_tot = [np.sum(bkg_hist[i:])+np.sum(sig_hist[i:])
                     for i in range(len(bins))]
        return [s/math.sqrt(t) if t > 0 else 0
                for s, t in zip(n_sig, n_tot)][::drt]

    def plot_fom(self, sig, bkg, var, bins, extra_name="", sb_denom=True,
                 reverse=False):
        """plot figure of merit"""
        if extra_name:
            extra_name = "_{}".format(extra_name)
        fom = self.get_fom(sig, bkg, var, bins, sb_denom, reverse)
        fom_maxbin = bins[fom.index(max(fom))]

        fig, ax = plt.subplots()

        plot = ax.plot(bins, fom, label="$S/\sqrt{B}=%.3f$\n cut=%.2f"%(max(fom), fom_maxbin))
        ax.plot(np.linspace(bins[0], bins[-1], 5), [max(fom)]*5,
                linestyle=':', color=plot[-1].get_color())
        ax.set_xlabel("BDT value", horizontalalignment='right', x=1.0)
        ax.set_ylabel("A.U.", horizontalalignment='right', y=1.0)

        ax2 = ax.twinx()
        sig_hist = self.get_hist(var, bins, sig)
        bkg_hist = self.get_hist(var, bins, bkg)
        bkg_name = "all" if len(bkg) > 1 else bkg[0]
        ax2.hist(x=bins[:-1], weights=sig_hist, bins=bins, histtype="step",
                 linewidth=1.5, color=self.color_dict[sig], label=sig,
                 density=True)
        ax2.hist(x=bins[:-1], weights=bkg_hist, bins=bins, histtype="step",
                 linewidth=1.5, color=self.color_dict[bkg_name], density=True,
                 label=bkg_name)
        # ax2.set_ylim(top=1.2*max(max(sig_hist)/sum(sig_hist),
        #                          max(bkg_hist)/sum(bkg_hist)))

        if reverse:
            ax.set_title("Reversed Cumulative Direction")
        ax.legend()
        ax2.legend()
        fig.tight_layout()
        plt.savefig("%s/StoB_%s.png" % (self.save_dir, extra_name))
        if self.do_show:
            plt.show()
        plt.close()

    def plot_fom_2d(self, sig, var1, var2, bins1, bins2, extra_name=""):
        """
        plot figure of merit but using 2d scan
        Returns info on max bin and max StoB
        """
        if extra_name:
            extra_name = "_{}".format(extra_name)
        grp_id = self.groups.index(sig)

        zvals = []
        xbins = list(bins1[:-1])*(len(bins2)-1)
        ybins = np.array([[i]*(len(bins1)-1) for i in bins2[:-1]]).flatten()
        max_fom = (0, -1, -1)
        for valy in bins2[:-1]:
            df_cut2 = self.work_set[self.work_set[var2] >= valy]
            for valx in bins1[:-1]:
                final = df_cut2[df_cut2[var1] >= valx]
                s = np.sum(final[final["classID"] == grp_id]["finalWeight"])
                b = np.sum(final[final["classID"] != grp_id]["finalWeight"])
                fom = s/math.sqrt(s+b) if b+s > 0 else 0
                if fom > max_fom[0]:
                    max_fom = (fom, valx, valy)
                zvals.append(fom)

        plt.hist2d(xbins, ybins, [bins1, bins2], weights=zvals,
                   cmap=plt.cm.jet)
        plt.plot([max_fom[1], max_fom[1]], [0, 1], 'r-')
        plt.plot([0, 1], [max_fom[2], max_fom[2]], 'r-')
        plt.colorbar()
        plt.xlabel(var1)
        plt.ylabel(var2)
        plt.savefig("{}/{}{}.png".format(self.save_dir, "stob2D", extra_name))
        plt.close()

        return max_fom

    def approx_likelihood(self, sig, bkg, var, bins):
        """
        Get Basic max Likelihood significance
        """
        sig_hist = self.get_hist(var, bins, sig)
        bkg_hist = self.get_hist(var, bins, bkg)
        term1, term2 = 0, 0
        for sig_val, bkg_val in zip(sig_hist, bkg_hist):
            if bkg_val <= 0 or sig_val <= 0:
                continue
            term1 += (sig_val+bkg_val)*math.log(1+sig_val/bkg_val)
            term2 += sig_val
        return math.sqrt(2*(term1 - term2))

    def make_roc(self, sig, bkg, var, extra_name=""):
        """
        Make and Save ROC curve for variable
        """
        final_set = pd.concat((self.get_sample(sig), self.get_sample(bkg)))
        pred = final_set["BDT.{}".format(var)].array
        if extra_name:
            extra_name = "_{}".format(extra_name)

        if var != sig:
            truth = [0 if i == self.groups.index(sig) else 1
                     for i in final_set["classID"].array]
        else:
            truth = [1 if i == self.groups.index(sig) else 0
                     for i in final_set["classID"].array]
        fpr, tpr, _ = roc_curve(truth, pred)
        auc = roc_auc_score(truth, pred)

        # plot
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label="AUC = {:.3f}".format(auc))
        ax.plot(np.linspace(0, 1, 5), np.linspace(0, 1, 5), linestyle=':')
        ax.legend()
        ax.set_xlabel("False Positive Rate", horizontalalignment='right', x=1.0)
        ax.set_ylabel("True Positive Rate", horizontalalignment='right', y=1.0)
        fig.tight_layout()
        plt.savefig("{}/roc_curve.BDT.{}{}.png"
                    .format(self.save_dir, var, extra_name))
        if self.do_show:
            plt.show()
        plt.close()

    def write_out(self, filename, input_tree, rm_groups=["Unnamed: 0", "GroupName", "classID", "finalWeight", "newWeight"], chan="SS"):
        """
        Write DataFrame out to ROOT file compatible with VVPlotter
        """
        import ROOT
        sumweight_file = ROOT.TFile(input_tree)
        sumweights = dict()
        for key in sumweight_file.GetListOfKeys():
            if "sumweight" not in key.GetName():
                continue
            sample = key.GetName()[10:]
            sumweights[sample] = key.ReadObj().GetBinContent(1)
        sumweight_file.Close()

        group_names = np.unique(self.work_set["GroupName"])
        var_info = dict()
        for var_name in list(set(self.work_set.columns) - set(rm_groups)):
            var = self.work_set[var_name]
            var_info[var_name] = [min(var), max(var),
                                  np.array_equal(var, var.astype(int))]
        outfile = ROOT.TFile("{}/{}".format(self.save_dir, filename), "RECREATE")
        for group in group_names:
            work_dir = outfile.mkdir(group)
            work_dir.cd()
            work_set = self.work_set[self.work_set["GroupName"] == group]
            work_weight = work_set["weight"]
            for var_name, var_details in var_info.items():
                bot = math.floor(var_details[0])
                n_digits = math.floor(math.log10(var_details[1]))
                top = round(var_details[1]+0.5*pow(10, n_digits))

                if var_details[2]:
                    hist = ROOT.TH1F("{}_{}".format(var_name, chan),
                                     "{}_{}".format(var_name, chan),
                                     int(top-bot), bot, top)
                else:
                    hist = ROOT.TH1F("{}_{}".format(var_name, chan),
                                     "{}_{}".format(var_name, chan),
                                     1024, bot, top)

                for vals, weight in zip(work_set[var_name], work_weight):
                    if "Pt" in var_name and vals < 5:
                        continue
                    hist.Fill(vals, weight)
                hist.Write()
            sumweight_hist = ROOT.TH1D("sumweights", "sumweights", 10, 0, 1)
            sumweight_hist.SetBinContent(1, sumweights[group]/(self.train_test_ratio))
            sumweight_hist.Write()
            outfile.cd()
        outfile.Write()

    def print_info(self, var, subgroups):
        """
        print out basic statistics information for all groups for a variable
        """
        from scipy.stats import kurtosis
        info = list()
        for grp_name in subgroups:
            final_set = self.work_set[self.work_set["GroupName"] == grp_name]
            variable = final_set[var]
            info.append([np.mean(variable), np.std(variable),
                         kurtosis(variable), np.sum(final_set["finalWeight"]),
                         len(final_set), grp_name])

        info = sorted(info, reverse=True)
        for arr in info:
            print("|{:10} | {:.2f}| {} | {:.2f}+-{:.2f} | {:0.2f} |"
                  .format(arr[5], arr[3], arr[4], arr[0], arr[1], arr[2]))
        print("-"*50)

    def plot_all_shapes(self, var, bins, extra_name=""):
        """
        Plot all groups (normalized to 1) to compare shapes
        """
        if extra_name:
            extra_name = "_{}".format(extra_name)

        fig, ax = plt.subplots()
        for group in self.groups:
            ax.hist(x=bins[:-1], weights=self.get_hist(var, bins, group),
                    bins=bins, label=group, histtype="step", linewidth=1.5,
                    density=True)
        ax.legend()
        ax.set_xlabel(var)
        ax.set_ylabel("A.U.")
        ax.set_title("Lumi = {} ifb".format(self.lumi))
        fig.tight_layout()
        plt.savefig("{}/{}{}.png".format(self.save_dir, var, extra_name))
        plt.close()

    # def setSHAP(self, useVar):
    #     self.model = xgb.Booster({'nthread': 4})  # init model
    #     self.model.load_model('{}/model.bin'.format(self.save_dir))  # load data
    #     explainer = shap.TreeExplainer(self.model)
    #     X_vals = self.work_set[useVar].sample(n=10000)

    #     shap_values = explainer.shap_values(X_vals)
    #     comb_vals = np.sum(shap_values, axis=0)
    #     top_inds = np.argsort(-np.sum(np.abs(comb_vals), 0))
    #     for f in  ["weight", "total_gain", "total_cover"]:
    #         scoreList = np.array(sorted([[useVar[int(key[1:])], val] for key, val in self.model.get_score(importance_type= f).items()], key=lambda scores: scores[1])).T
    #         nameSort = scoreList[0]
    #         valSort = scoreList[1].astype(float)
    #         index = np.arange(len(nameSort))
    #         fig, ax = plt.subplots()
    #         ax.barh(index, valSort)
    #         ax.set_yticks(index)
    #         ax.set_yticklabels(nameSort, rotation=15)
    #         ax.set_title(f)
    #         fig.set_size_inches(8, 8)
    #         fig.tight_layout()
    #         plt.savefig("{}/{}_list.png".format(self.save_dir, f))
    #         plt.close()

    #     # # make SHAP plots of the three most important features
    #     # for i in range(3):
    #     #     shap.dependence_plot(top_inds[i], comb_vals, X_vals)

    #     #shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])

    #     f = plt.figure()
    #     shap.summary_plot(shap_values, X_vals, plot_type="bar")
    #     f.savefig("{}/SHAP_list.png".format(self.save_dir))
    #     plt.close()


def findScale(s, b):
    scale = 1
    prevS = 1
    while b//(scale*s) != 0:
        prevS = scale
        if int(math.log10(scale)) == math.log10(scale):
            scale *= 5
        else:
            scale *= 2
    return prevS
