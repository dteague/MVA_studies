"""
.. module:: XGBoostMaker
   :synopsis: Takes in ROOT file to run a BDT training over it using XGBoost
.. moduleauthor:: Dylan Teague
"""
import numpy as np
import uproot
import pandas as pd
import xgboost as xgb

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
    def __init__(self, use_vars, spec_vars, cuts):
        """Constructor method
        """
        self.split_ratio = 0.5
        self.group_names = ["Signal"]
        self.pred_train = dict()
        self.pred_test = dict()

        self._include_vars = use_vars + spec_vars
        self._drop_vars = spec_vars + ["classID", "GroupName", "finalWeight"]
        self._all_vars = self._include_vars + ["classID", "GroupName"]
        self.train_set = pd.DataFrame(columns=self._all_vars)
        self.test_set = pd.DataFrame(columns=self._all_vars)

        self.cuts = cuts.split("&&")
        # XGBoost training
        self.param = {"eta": 0.09, 'silent': 1, "nthread": 3, 'reg_alpha': 0.0,
                      'min_child_weight': 1e-6, 'n_estimators': 200,
                      'reg_lambda': 0.05, 'scale_pos_weight': 1, 'subsample': 1,
                      'base_score': 0.5, 'colsample_bylevel': 1, 'max_depth': 5,
                      'colsample_bytree': 1, 'gamma': 0, 'learning_rate': 0.1,
                      'max_delta_step': 0, }

    def add_group(self, group_name, sample_names, infile):
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
        for name in sample_names:
            df = infile[name].pandas.df(self._include_vars)
            df = self._cut_frame(df)
            totalSW += np.sum(np.abs(df["newWeight"]))
            totalEv += len(df)
            scale = 1.*totalEv/totalSW

        for name in sample_names:
            df = infile[name].pandas.df(self._include_vars)
            df = self._cut_frame(df)
            df["GroupName"] = name
            df["classID"] = class_id
            df.insert(0, "finalWeight", np.abs(df["newWeight"])*scale)

            train, test = train_test_split(df, test_size=self.split_ratio,
                                           random_state=12345)
            print("Add Tree {} of type {} with {} event"
                  .format(name, group_name, len(train)))
            self.train_set = pd.concat([train.reset_index(drop=True),
                                        self.train_set], sort=True)
            self.test_set = pd.concat([test.reset_index(drop=True),
                                       self.test_set], sort=True)

    def train_single(self, groupName):
        """**Train for multiclass BDT**

        Does final weighting of the data (normalize all groups total
        weight to the same value), train the BDT only against the
        group specified, and fill the predictions ie the BDT values
        for each group.

        Args:
          groupName(string): Group to train against Signal

        Returns:
          xgboost.XGBClassifer: XGBoost model that was just trained

        """
        # need to generalize (if I'm keeping this at all...)
        rmIdx = 3-self.group_names.index(groupName)
        workTrain = self.train_set[self.train_set["classID"] != rmIdx]

        x_train = workTrain.drop(self._drop_vars, axis=1)
        w_train = workTrain["finalWeight"].copy()
        y_train = [1 if cID==0 else 0 for cID in workTrain["classID"]]
        x_test = self.test_set.drop(self._drop_vars, axis=1)
        y_test = [1 if cID==0 else 0 for cID in self.test_set["classID"]]

        group_tot = y_train.value_counts()
        for i in np.unique(y_train):
            w_train[self.train_set["classID"] == i] *= min(group_tot)/group_tot[i]

        self.param['objective'] = "binary:logistic"  # 'multi:softprob'
        self.param['eval_metric'] = 'logloss'  # "mlogloss"
        self.param['num_class'] = 2
        num_rounds = 150

        dtrain = xgb.DMatrix(x_train, label=y_train, weight=w_train)
        dtrainAll = xgb.DMatrix(self.train_set.drop(self._drop_vars, axis=1))
        dtest = xgb.DMatrix(x_test, label=y_test,
                            weight=self.test_set["finalWeight"])
        evallist = [(dtrain,'train'), (dtest, 'test')]
        fit_model = xgb.train(self.param, dtrain, num_rounds, evallist,
                              verbose_eval=50)

        self.pred_test[groupName] = fit_model.predict(dtest)
        self.pred_train[groupName] = fit_model.predict(dtrainAll)

        return fit_model

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

    def output(self, outname):
        """Wrapper for write out commands

        Args:
          outname: Directory where files will be written

        """
        self._write_pandas("{}/testTree.pkl.gz".format(outname),
                               self.test_set, self.pred_test)
        self._write_pandas("{}/trainTree.pkl.gz".format(outname),
                               self.train_set, self.pred_train)

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

    def _write_pandas(self, outname, workSet, prediction):
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
            workSet.to_pickle(outname, compression="gzip")
