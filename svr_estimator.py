
import time

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import NuSVR


def loguniform(low=0, high=1, size=None):
    return np.exp(np.random.uniform(np.log(low), np.log(high), size))


class SVR_Estimator(object):

    def __init__(self, metric_all_arch_list, test_acc_all_archs_list,
                 n_hypers=1000, n_train=200, all_curve=False, model_name='svr'):

        self.n_hypers = n_hypers
        self.all_curve = all_curve
        self.n_train = n_train
        self.model_name = model_name

        self.VC = np.vstack(metric_all_arch_list[0])
        if len(metric_all_arch_list) > 1:
            self.HP = np.vstack(metric_all_arch_list[1])
            self.AP = np.vstack(metric_all_arch_list[2])
            self.includeAPHP = True
        else:
            self.includeAPHP = False

        self.DVC = np.diff(self.VC, n=1, axis=1)
        self.DDVC = np.diff(self.DVC, n=1, axis=1)
        self.max_epoch = self.VC.shape[1]
        self.test_acc_seed_all_arch = test_acc_all_archs_list

    def learn_hyper(self, epoch, seed=0):

        n_epoch = int(epoch)
        VC_sub = self.VC[:,:n_epoch]
        DVC_sub = self.DVC[:,:n_epoch]
        DDVC_sub = self.DDVC[:,:n_epoch]
        mVC_sub = np.mean(VC_sub, axis=1)[:,None]
        stdVC_sub = np.std(VC_sub, axis=1)[:,None]
        mDVC_sub = np.mean(DVC_sub, axis=1)[:,None]
        stdDVC_sub = np.std(DVC_sub, axis=1)[:,None]
        mDDVC_sub = np.mean(DDVC_sub, axis=1)[:,None]
        stdDDVC_sub = np.std(DDVC_sub, axis=1)[:,None]

        if self.all_curve:
            TS = np.hstack([VC_sub, DVC_sub, DDVC_sub, mVC_sub, stdVC_sub])
        else:
            TS = np.hstack([mVC_sub, stdVC_sub, mDVC_sub, stdDVC_sub, mDDVC_sub, stdDDVC_sub])

        if self.includeAPHP:
            X = np.hstack([self.AP, self.HP, TS])
        else:
            X = TS

        y_val_acc = self.VC[:,-1]
        y_test_acc = np.array(self.test_acc_seed_all_arch)
        y = np.vstack([y_val_acc, y_test_acc]).T

        # ========== split into train/test data sets ==========
        split = (X.shape[0] - self.n_train)/X.shape[0]
        X_train, X_test, y_both_train, y_both_test = train_test_split(
            X, y, test_size=split, random_state=seed)
        y_train = y_both_train[:,0]  # all final validation acc
        y_test  = y_both_test[:,1]   # all final test acc

        np.random.seed(seed)
        # ========== specify model parameters ==========
        if self.model_name == 'svr':
            C = loguniform(1e-5, 10,  self.n_hypers)
            nu = np.random.uniform(0, 1, self.n_hypers)
            gamma = loguniform(1e-5, 10,  self.n_hypers)
            hyper = np.vstack([C, nu, gamma]).T

        print(f'start CV on {self.model_name}')
        mean_score_list=[]
        t_start = time.time()
        for i in range(self.n_hypers):
            # ========== define model ==========
            if self.model_name == 'svr':
                model = NuSVR(C=hyper[i, 0], nu=hyper[i, 1], gamma=hyper[i, 2], kernel='rbf')

            # ========== perform cross validation to learn the best hyper value ==========
            scores = cross_val_score(model, X_train, y_train, cv=3)
            mean_scores = np.mean(scores)
            mean_score_list.append(mean_scores)
        t_end = time.time()

        best_hyper_idx = np.argmax(mean_score_list)
        best_hyper = hyper[best_hyper_idx]
        max_score = np.max(mean_score_list)
        time_taken = t_end - t_start
        print(f'{self.model_name} on {seed} n_train={self.n_train}: '
              f'best_hyper={best_hyper}, score={max_score}, time={time_taken}')

        self.epoch = epoch
        self.best_hyper = best_hyper
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

        return best_hyper, time_taken

    def extrapolate(self):

        if self.model_name == 'svr':
            best_model = NuSVR(C=self.best_hyper[0], nu=self.best_hyper[1], gamma=self.best_hyper[2], kernel='rbf')

        # ========== train and fit model ==========
        best_model.fit(self.X_train, self.y_train)
        y_pred = best_model.predict(self.X_test)
        self.y_pred = y_pred
        rank_corr, p = stats.spearmanr(self.y_test, y_pred)

        print(f'{self.model_name}: n_train={self.n_train} e={self.epoch}: rank_corr={rank_corr}')

        return rank_corr


    def predict(self, X_test):

        if self.model_name == 'svr':
            best_model = NuSVR(C=self.best_hyper[0], nu=self.best_hyper[1], gamma=self.best_hyper[2], kernel='rbf')

        # ========== train and fit model ==========
        best_model.fit(self.X_train, self.y_train)
        y_pred = best_model.predict(X_test)

        return y_pred