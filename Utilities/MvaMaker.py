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
    def __init__(self, use_vars, spec_vars, cuts):
        self.split_ratio = 0.5
        self.group_names = ["Signal"]
        self.groups_total = list()
        self.pred_train = dict()
        self.pred_test = dict()

        self.include_vars = use_vars + spec_vars
        self.drop_vars = spec_vars + ["classID", "GroupName", "finalWeight"]
        self.all_vars = self.include_vars + ["classID", "GroupName"]
        self.train_set = pd.DataFrame(columns=self.all_vars)
        self.test_set = pd.DataFrame(columns=self.all_vars)

        self.cuts = cuts.split("&&")
        # XGBoost training
        self.param = {"eta": 0.09, 'silent': 1, "nthread": 3, 'reg_alpha': 0.0,
                      'min_child_weight': 1e-6, 'n_estimators': 200,
                      'reg_lambda': 0.05, 'scale_pos_weight': 1, 'subsample': 1,
                      'base_score': 0.5, 'colsample_bylevel': 1, 'max_depth': 5,
                      'colsample_bytree': 1, 'gamma': 0, 'learning_rate': 0.1,
                      'max_delta_step': 0, }

    def add_group(self, inNames, outName, infile):
        class_id = 0
        if outName != "Signal":
            self.group_names.append(outName)
            class_id = len(self.group_names)-1

        # Get scale for group
        #
        # Scales each component of group by (# raw Events)/(# scaled Events)
        # This is done so each effective xsec is used as a ratio of the group
        # and the number of raw Events is so the average weight is 1 (what xgb wants)
        totalSW, totalEv = 0, 0
        for name in inNames:
            df = infile[name].pandas.df(self.include_vars)
            df = self.cutFrame(df)
            totalSW += np.sum(np.abs(df["newWeight"]))
            totalEv += len(df)
            scale = 1.*totalEv/totalSW

        group_total = 0.
        for name in inNames:
            df = infile[name].pandas.df(self.include_vars)
            df = self.cutFrame(df)
            df["GroupName"] = name
            df["classID"] = class_id
            df.insert(0, "finalWeight", np.abs(df["newWeight"])*scale)

            train, test = train_test_split(df, test_size=self.split_ratio,
                                           random_state=12345)
            group_total += len(train)
            print("Add Tree {} of type {} with {} event"
                  .format(name, outName, len(train)))
            self.train_set = pd.concat([train.reset_index(drop=True),
                                        self.train_set], sort=True)
            self.test_set = pd.concat([test.reset_index(drop=True),
                                       self.test_set], sort=True)

        self.groups_total.append(group_total)

    def cutFrame(self, frame):
        """Reduce frame using root style cut string

        :param frame:
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

    def train_single(self, groupName):
        rmIdx = 3-self.group_names.index(groupName)
        workTrain = self.train_set[self.train_set["classID"] != rmIdx]

        x_train = workTrain.drop(self.drop_vars, axis=1)
        w_train = workTrain["finalWeight"].copy()
        y_train = [1 if cID==0 else 0 for cID in workTrain["classID"]]
        x_test = self.test_set.drop(self.drop_vars, axis=1)
        y_test = [1 if cID==0 else 0 for cID in self.test_set["classID"]]

        for i, groupTot in enumerate(self.groups_total):
            w_train[workTrain["classID"] == i] *= min(self.groups_total)/groupTot

        self.param['objective'] = "binary:logistic"  # 'multi:softprob'
        self.param['eval_metric'] = 'logloss'  # "mlogloss"
        self.param['num_class'] = 2
        num_rounds = 150

        dtrain = xgb.DMatrix(x_train, label=y_train, weight=w_train)
        dtrainAll = xgb.DMatrix(self.train_set.drop(self.drop_vars, axis=1))
        dtest = xgb.DMatrix(x_test, label=y_test,
                            weight=self.test_set["finalWeight"])
        evallist = [(dtrain,'train'), (dtest, 'test')]
        fit_model = xgb.train(self.param, dtrain, num_rounds, evallist,
                              verbose_eval=50)

        self.pred_test[groupName] = fit_model.predict(dtest)
        self.pred_train[groupName] = fit_model.predict(dtrainAll)

        return fit_model

    def train(self):
        """
        """
        workTrain = self.train_set
        workNames = self.group_names

        x_train = workTrain.drop(self.drop_vars, axis=1)
        w_train = workTrain["finalWeight"].copy()
        y_train = workTrain["classID"]

        x_test = self.test_set.drop(self.drop_vars, axis=1)
        y_test = self.test_set["classID"]

        for i, groupTot in enumerate(self.groups_total):
            w_train[workTrain["classID"] == i] *= min(self.groups_total)/groupTot

        self.param['objective'] = 'multi:softprob'
        self.param['eval_metric'] = "mlogloss"
        self.param['num_class'] = len(np.unique(y_train))

        fit_model = xgb.XGBClassifier(**self.param)
        fit_model.fit(x_train, y_train, w_train,
                      eval_set=[(x_train, y_train), (x_test, y_test)],
                      early_stopping_rounds=100, verbose=50)

        for i, grp in enumerate(workNames):
            self.pred_test[grp] = fit_model.predict_proba(x_test).T[i]
            self.pred_train[grp] = fit_model.predict_proba(x_train).T[i]

        return fit_model

    def output(self, outname):
        """ Wrapper for write out commands

        :param outname:
        """
        with uproot.recreate("{}/BDT.root".format(outname)) as outfile:
            self.writeOutRoot(outfile, "TestTree", self.test_set,
                              self.pred_test)
            self.writeOutRoot(outfile, "TrainTree", self.train_set,
                              self.pred_train)
            self.writeOutPandas("{}/testTree.pkl.gz".format(outname),
                                self.test_set, self.pred_test)
            self.writeOutPandas("{}/trainTree.pkl.gz".format(outname),
                                self.train_set, self.pred_train)

    def writeOutPandas(self, outname, workSet, prediction):
        """Write out pandas file as a pickle file that is compressed

        :param outname:
        :param workSet:
        :param prediction:
        """
        set_difference = set(workSet.columns) - set(self.all_vars)
        workSet = workSet.drop(list(set_difference), axis=1)
        for key, arr in prediction.items():
            workSet.insert(0, key, arr)
            workSet.to_pickle(outname, compression="gzip")

    def writeOutRoot(self, outfile, treeName, workSet, prediction):
        """Write out as a rootfile

        :param outfile:
        :param treeName:
        :param workSet:
        :param prediction:
        """
        out_dict = {name: workSet[name] for name in self.all_vars}
        out_dict.update(prediction)
        del out_dict["GroupName"]
        outfile[treeName] = uproot.newtree({name: "float32"
                                            for name in out_dict.keys()})
        outfile[treeName].extend(out_dict)
