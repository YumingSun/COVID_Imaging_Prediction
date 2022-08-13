# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 15:32:03 2022

@author: sunym
"""
from sksurv.linear_model import CoxPHSurvivalAnalysis
import numpy as np
from sksurv.metrics import integrated_brier_score
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
import os
import sys
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.ensemble import RandomSurvivalForest
from sksurv.svm import FastKernelSurvivalSVM
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
import eli5
from eli5.sklearn import PermutationImportance
from preprocess import train_test_preprocess
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


def get_feature_importance(model,x_train,x_test,y_train,y_test,niter,bestParam):
    '''
    Parameters
    ----------
    model : STRING
        Type of the model used to calculate feature importance
    x_train : DataFrame
    x_test : DataFrame
    y_train : DataFrame
    y_test : DataFrame
    niter : INT
        The number of iterations to calculate feature importance
    bestParam : Dict
        The tunning parameters of the model
    Returns
    -------
    featureImportance : DataFrame
        Feature importance from testing dataset

    '''
    if model == 'svm':
        est = FastKernelSurvivalSVM(**bestParam)
    elif model == 'cox':
        est = CoxPHSurvivalAnalysis(n_iter = 200 )
    elif model == 'randomForest':
        est = RandomSurvivalForest(**bestParam)
    elif model == 'gradientBoosting':
        est = GradientBoostingSurvivalAnalysis(**bestParam)
    
    est.fit(x_train, y_train)
    perm = PermutationImportance(est, n_iter=niter)
    perm.fit(x_test, y_test)
    featureImportance = pd.DataFrame({'score' : perm.feature_importances_},
                 index = x_train.columns.tolist())
    
    return featureImportance

def get_ensemble_feature_importance(model,x_train,x_test,y_train,y_test,niter,
                                    bestSvmParam,bestRsfParam,bestGbParam):
    '''
    Parameters
    ----------
    model : STRING
        Type of the model used to calculate feature importance
    x_train : DataFrame
    x_test : DataFrame
    y_train : DataFrame
    y_test : DataFrame
    niter : INT
        The number of iterations to calculate feature importance
    bestSvmParam : Dict
        The tunning parameters of the SVM
    bestRsfParam : Dict
        The tunning parameters of the random forest
    bestGbParam : Dict
        The tunning parameters of the gradient boosting

    Returns
    -------
    featureImportance : DataFrame
        Feature importance from testing dataset

    '''
    est = Ensemble(svmParam = bestSvmParam,rsfParam=bestRsfParam,gbParam=bestGbParam)
    est.fit(x_train, y_train)
    perm = PermutationImportance(est, n_iter=niter)
    perm.fit(x_test, y_test)
    featureImportance = pd.DataFrame({'score' : perm.feature_importances_},
                 index = x_train.columns.tolist())
    
    return featureImportance


if __name__ == '__main__':
    numOfExp = int(sys.argv[1])
    dataTransform = sys.argv[2]
    model = sys.argv[3]
    
    dataSplitPath = ''
    dataPath = ''
    
    selectedClinicImagePath = ''
    selectedClinicPath = ''


    resultPath = ''
    
    paramPath = ''
    
    
    allDataAll = pd.read_csv(os.path.join(dataPath,'data.csv'))

    imageNamesAll = pickle.load(open(os.path.join(dataPath,'imageFeatureNames.pkl'),'rb'))
    clinicNamesAll = pickle.load(open(os.path.join(dataPath,'clinicFeatureNames.pkl'),'rb'))
    
    
    ids = pickle.load(open(os.path.join(dataSplitPath,'train_test_{:02d}.pkl'.format(numOfExp)),'rb'))
    
    
    selectedImageClinic = pickle.load(open(os.path.join(selectedClinicImagePath,
                                                  'selectedImageClinic.pkl'),
                                           'rb'))

    selectedClinic = [f for f in selectedImageClinic if f in clinicNamesAll]
    selectedImage = [f for f in selectedImageClinic if f in imageNamesAll]


    
    (trainClinic,testClinic,trainClinicImage,
     testClinicImage,outcomeTrain,OutcomeTest) = \
        train_test_preprocess(allDataAll,ids,selectedClinic,
                              selectedImageClinic,imageNamesAll,
                              clinicNamesAll,dataTransform)
    
    if model == 'svm':
        bestClinicParam = pickle.load(open(os.path.join(paramPath,'clinicParamSvm.pkl'),'rb'))
        bestClinicImageParam = pickle.load(open(os.path.join(paramPath,'clinicImageParamSvm.pkl'),'rb'))
    elif model == 'cox':
        bestClinicParam = None
        bestClinicImageParam = None
    elif model == 'randomForest':
        bestClinicParam = pickle.load(open(os.path.join(paramPath,'clinicParamRsf.pkl'),'rb'))
        bestClinicImageParam = pickle.load(open(os.path.join(paramPath,'clinicImageParamRsf.pkl'),'rb'))
    elif model == 'gradientBoosting':
        bestClinicParam = pickle.load(open(os.path.join(paramPath,'clinicParamGb.pkl'),'rb'))
        bestClinicImageParam = pickle.load(open(os.path.join(paramPath,'clinicImageParamGb.pkl'),'rb'))
    elif model == 'Ensemble':
        bestClinicParamSvm = pickle.load(open(os.path.join(paramPath,'clinicParamSvm.pkl'),'rb'))
        bestClinicImageParamSvm = pickle.load(open(os.path.join(paramPath,'clinicImageParamSvm.pkl'),'rb'))
        
        bestClinicParamRsf = pickle.load(open(os.path.join(paramPath,'clinicParamRsf.pkl'),'rb'))
        bestClinicImageParamRsf = pickle.load(open(os.path.join(paramPath,'clinicImageParamRsf.pkl'),'rb'))
        
        bestClinicParamGb = pickle.load(open(os.path.join(paramPath,'clinicParamGb.pkl'),'rb'))
        bestClinicImageParamGb = pickle.load(open(os.path.join(paramPath,'clinicImageParamGb.pkl'),'rb'))

    if model == 'Ensemble':
        fi = get_ensemble_feature_importance(model,trainClinicImage,testClinicImage,
                               outcomeTrain,OutcomeTest,20,
                               bestClinicImageParamSvm,
                               bestClinicImageParamRsf,
                               bestClinicImageParamGb)
    else:
        fi = get_feature_importance(model,trainClinicImage,testClinicImage,
                               outcomeTrain,OutcomeTest,20,bestClinicImageParam)
        
    pickle.dump(fi, open(os.path.join(resultPath,'results.pkl'),'wb'))
    
    

    
    
