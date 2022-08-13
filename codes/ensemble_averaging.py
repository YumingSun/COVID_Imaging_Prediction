# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 16:22:39 2022

@author: sunym
"""
from sksurv.ensemble import RandomSurvivalForest
from sksurv.svm import FastKernelSurvivalSVM
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis
import numpy as np
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)

class Ensemble:
    def __init__(self,svmParam={},rsfParam={},gbParam={}):
        self.svmParam = svmParam
        self.rsfParam = rsfParam
        self.gbParam = gbParam
        
        
    def get_params(self,deep=True):
        return {"svmParam" : self.svmParam,
                "rsfParam": self.rsfParam,
                "gbParam" : self.gbParam}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
            
    def fit(self,data_x,data_y):
        self.cox_est = CoxPHSurvivalAnalysis(n_iter = 200 )
        self.svm_est = FastKernelSurvivalSVM(**self.svmParam)
        self.rsf_est = RandomSurvivalForest(**self.rsfParam)
        self.gb_est = GradientBoostingSurvivalAnalysis(**self.gbParam)
        
        self.cox_est.fit(data_x,data_y)
        self.svm_est.fit(data_x,data_y)
        self.rsf_est.fit(data_x,data_y)
        self.gb_est.fit(data_x,data_y)
        return self
    
    def predict(self, data_x):
        coxScore = self.cox_est.predict(data_x)
        coxScoreScale = (coxScore - coxScore.min())/(coxScore.max() - coxScore.min())
        coxScoreScale = np.expand_dims(coxScoreScale,axis = 1)
        
        svmScore = self.svm_est.predict(data_x)
        svmScoreScale = (svmScore - svmScore.min())/(svmScore.max() - svmScore.min())
        svmScoreScale = np.expand_dims(svmScoreScale,axis = 1)
        
        rsfScore = self.rsf_est.predict(data_x)
        rsfScoreScale = (rsfScore - rsfScore.min())/(rsfScore.max() - rsfScore.min())
        rsfScoreScale = np.expand_dims(rsfScoreScale,axis = 1)
        
        gbScore = self.gb_est.predict(data_x)
        gbScoreScale = (gbScore - gbScore.min())/(gbScore.max() - gbScore.min())
        gbScoreScale = np.expand_dims(gbScoreScale,axis = 1)
        
        scoreAll = np.concatenate([coxScoreScale,
                                   svmScoreScale,
                                   rsfScoreScale,
                                   gbScoreScale],axis = 1)
        scoreEnsemble = np.mean(scoreAll,axis = 1)
        
        return scoreEnsemble
    
    def score(self,data_x,y):
        y_pred =  self.predict(data_x)
        y_event = y['Status']
        y_time = y['Survival_in_days']
        c = concordance_index_censored(y_event,y_time,y_pred)[0]
        return c