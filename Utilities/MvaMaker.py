"""
.. module:: XGBoostMaker
   :synopsis: Takes in ROOT file to run a BDT training over it using XGBoost
.. moduleauthor:: Dylan Teague
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from dylan_utils import VarGetter
import awkward1 as ak
# from dylan_utils import FileInfo

from sklearn.model_selection import train_test_split

class XGBoostMaker:
    """Wrapper for XGBoost training. Takes an uproot input, a list of
    groups to do a multiclass training, as well as a cut string if
    needed and trains the data. After it is done, the results can be
    outputed to be piped into MVAPlotter

    Args:
      split_ratio(float): Ratio of test events for train test splitting
      group_names(list): List of the names of the different groups
      pred_train(dict): Dictionary of group name to BDT associated with it for train set
      pred_test(dict): Dictionary of group name to BDT associated with it for test set
      train_set(pandas.DataFrame): DataFrame of the training events
      test_set(pandas.DataFrame): DataFrame of the testing events
      cuts(list): List of ROOT style cuts to apply
      param(dict): Variables used in the training

    """
    def __init__(self, use_vars, cuts=""):
        """Constructor method
        """
        self.split_ratio = 0.5
        self.group_names = ["Signal"]
        self.pred_train = dict()
        self.pred_test = dict()

        self.use_vars = use_vars
        self._include_vars = list(use_vars.keys())
        self._drop_vars = ["classID", "groupName", "finalWeight", "scale_factor"]
        self._all_vars = self._include_vars + self._drop_vars 
        self.train_set = pd.DataFrame(columns=self._all_vars)
        self.test_set = pd.DataFrame(columns=self._all_vars)

        for key, func in self.use_vars.items():
            dtype = "int" if "num" in func else 'float'
            self.train_set[key] = self.train_set[key].astype(dtype)
            self.test_set[key] = self.test_set[key].astype(dtype)

        
        self.cuts = cuts.split("&&")
        # XGBoost training
        self.param = {"eta": 0.09,  "nthread": 3, 'reg_alpha': 0.0,
                      'min_child_weight': 1e-6, 'n_estimators': 200,
                      'reg_lambda': 0.05,  'subsample': 1, 'base_score': 0.5,
                      'colsample_bylevel': 1, 'max_depth': 5, 'learning_rate': 0.1,
                      'colsample_bytree': 1, 'gamma': 0, 'max_delta_step': 0,}
        # 'silent': 1, 'scale_pos_weight': 1,

    def add_group(self, group_name, sample_names, indir):
        """**Add Information about a group to class**

        This grabs all the variable information about each sample,
        does some preliminary weighting and splits the data into the
        test and train set (based on `self.split_ratio`)

        Args:
          group_name(string): Name of group being added
          sample_names(list): List of samples in the group
          infile(uproot.ROOTDirectory): Uproot file with data to be added
        """
        class_id = 0
        if group_name != "Signal":
            self.group_names.append(group_name)
            class_id = len(self.group_names)-1

        # Get scale for group
        #
        # Scales each component of group by (# raw Events)/(# scaled Events)
        # This is done so each effective xsec is used as a ratio of the group
        # and the number of raw Events is so the average weight is 1 (what xgb wants)

        totalSW, totalEv = 0, 0
        test_list, train_list = [], []
        df_dict = dict()
        for varname in self._include_vars:
            df_dict[varname] = ak.Array([])
            
        for name in sample_names:
            try:
                arr = VarGetter("{}/{}_cut.parquet".format(indir, name))
            except:
                continue
            df_dict = dict()
            for varname, func in self.use_vars.items():
                df_dict[varname] = ak.to_numpy(eval("arr.{}".format(func)))
            df_dict["scale_factor"] = ak.to_numpy(arr.scale)
            totalSW += ak.sum(arr.scale)
            totalEv += len(arr.scale)

            df = pd.DataFrame.from_dict(df_dict)
            df["classID"] = class_id
            df["groupName"] = name
            train, test = train_test_split(df, test_size=self.split_ratio,
                                           random_state=12345)
            test_list.append(test)
            train_list.append(train)
            print("Add Tree {} of type {} with {} event"
                  .format(name, group_name, len(train)))

        scale = 1.*totalEv/totalSW
        for test, train in zip(test_list, train_list):
            train.insert(0, "finalWeight", scale*np.abs(train["scale_factor"]))
            test.insert(0, "finalWeight", scale*np.abs(test["scale_factor"]))
            self.train_set = pd.concat([train.reset_index(drop=True),
                                        self.train_set], sort=True)
            self.test_set = pd.concat([test.reset_index(drop=True),
                                       self.test_set], sort=True)

    def train(self):
        """**Train for multiclass BDT**

        Does final weighting of the data (normalize all groups total
        weight to the same value), train the BDT, and fill the
        predictions ie the BDT values for each group.

        Returns:
          xgboost.XGBClassifer: XGBoost model that was just trained

        """
        x_train = self.train_set.drop(self._drop_vars, axis=1)
        w_train = self.train_set["finalWeight"].copy()
        y_train = self.train_set["classID"]
        
        x_test = self.test_set.drop(self._drop_vars, axis=1)
        y_test = self.test_set["classID"]

        group_tot = y_train.value_counts()
        for i in np.unique(y_train):
            w_train[self.train_set["classID"] == i] *= min(group_tot)/group_tot[i]

        self.param['objective'] = 'multi:softprob'
        self.param['eval_metric'] = "mlogloss"
        self.param['num_class'] = len(np.unique(y_train))

        fit_model = xgb.XGBClassifier(**self.param)
        fit_model.fit(x_train, y_train, w_train,
                      eval_set=[(x_train, y_train), (x_test, y_test)],
                      early_stopping_rounds=100, verbose=50)

        for i, grp in enumerate(self.group_names):
            self.pred_test[grp] = fit_model.predict_proba(x_test).T[i]
            self.pred_train[grp] = fit_model.predict_proba(x_train).T[i]

        return fit_model

    def output(self, outdir):
        """Wrapper for write out commands

        Args:
          outname: Directory where files will be written

        """
        self._write_pandas("{}/test".format(outdir), self.test_set,
                           self.pred_test)
        self._write_pandas("{}/train".format(outdir), self.train_set,
                           self.pred_train)

    # Private Functions

    def _cut_frame(self, frame):
        """**Reduce frame using root style cut string**

        Args:
          frame(pandas.DataFrame): DataFrame to cut on

        """
        for cut in self.cuts:
            if cut.find("<") != -1:
                tmp = cut.split("<")
                frame = frame[frame[tmp[0]] < float(tmp[1])]
            elif cut.find(">") != -1:
                tmp = cut.split(">")
                frame = frame[frame[tmp[0]] > float(tmp[1])]
            elif cut.find("==") != -1:
                tmp = cut.split("==")
                frame = frame[frame[tmp[0]] == float(tmp[1])]
        return frame

    def _write_pandas(self, outdir, workSet, prediction):
        """**Write out pandas file as a compressed pickle file

        Args:
          outfile(string): Name of file to write
          workSet(pandas.DataFrame): DataFrame of variables to write out
          prediction(pandas.DataFrame): DataFrame of BDT predictions

        """
        set_difference = set(workSet.columns) - set(self._all_vars)
        workSet = workSet.drop(list(set_difference), axis=1)
        for key, arr in prediction.items():
            workSet.insert(0, key, arr)
            
        for group in np.unique(workSet.groupName):
            outname = "{}/{}.parquet".format(outdir, group)
            workSet[workSet.groupName == group].to_parquet(outname, compression="gzip")

        # workSet.to_parquet(outname, compression="gzip")
