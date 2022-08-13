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
from scipy.stats import boxcox
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
import eli5
from eli5.sklearn import PermutationImportance

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


def log_transform(X):
    negId = X.columns[X.min() <= 0]
    X.loc[:,negId] = X.loc[:,negId] - X.loc[:,negId].min() + 1
    return np.log(X)

def square_root_transform(X):
    negId = X.columns[X.min() <= 0]
    X.loc[:,negId] = X.loc[:,negId] - X.loc[:,negId].min() + 1
    return np.sqrt(X)

def box_cox(x):
    return boxcox(x)[0]

def box_cox_transform(X):
    negId = X.columns[X.min() <= 0]
    X.loc[:,negId] = X.loc[:,negId] - X.loc[:,negId].min() + 1
    return X.apply(box_cox,axis = 0)

def image_preprocess(image, imageTransform = None):
    if imageTransform is not None:
        if imageTransform == 'Standardize':
            scaler = StandardScaler().fit(image.to_numpy())
            imageScaled = scaler.transform(image.to_numpy())
            imageScaled = pd.DataFrame(imageScaled,index = image.index,
                                              columns = image.columns)
        elif imageTransform == 'LogTransform':
            imageScaled = log_transform(image)
        elif imageTransform == 'SquareRootTransform':
            imageScaled = square_root_transform(image)
        else:
            imageScaled = box_cox_transform(image)
    else:
        imageScaled = image
        
    return imageScaled

def clinic_preprocess(clinic):
    ctsAll = ['Age','BMI_mean','SpO2','Temperature',
                  'RespiratoryRate','BPMeanNonInvasive',
                  'BPDiaNonInvasive','BPSysNonInvasive',
                  'HeartRate','affluence13_17_qrtl',
                  'disadvantage13_17_qrtl',
                  'ethnicimmigrant13_17_qrtl',
                  'ped1_13_17_qrtl','TotalScore']

    allFeatures = clinic.columns.values.tolist()

    clinic = clinic.fillna(clinic.mean(axis = 0))
    ctsClinicIndex = [i for i in allFeatures if i in ctsAll]
    if len(ctsClinicIndex) > 1:
        ctsClinic = clinic[ctsClinicIndex]

        scaler = StandardScaler().fit(ctsClinic.to_numpy())
        ctsClinicScaled = scaler.transform(ctsClinic.to_numpy())

        clinic[ctsClinicIndex] = ctsClinicScaled

    return clinic

def image_clinic_preprocess(data, selectedFea, imageNameAll, clinicNameAll,
                            transform,deleteSkew  = True):
    selectedImage = [fea for fea in selectedFea if fea in imageNameAll]
    selectedClinic = [fea for fea in selectedFea if fea in clinicNameAll]

    image = read_feature(data,selectedImage,transform,deleteSkew)
    clinic = data.loc[:, selectedClinic]
    clinic  = clinic_preprocess(clinic)

    clinicImage = image.join(clinic)

    return clinicImage,clinicImage.columns.tolist()

def read_feature(data, selectedFeature,varTransform,deleteSkew = True):
    dataSub = data.loc[:,selectedFeature]
    if deleteSkew:
        unskewFeatures = dataSub.columns[dataSub.skew().abs() < 10]
        dataSub = dataSub.loc[:,unskewFeatures]
    dataSub = image_preprocess(dataSub,varTransform)
    return dataSub
    cvRes = pickle.load(open(
        os.path.join(resLoc,var + '_0718',
                     'cv_{}_Exp{:02d}_0718.pkl'.format(var,numOfExp)),
        'rb'))
    bestParam = cvRes.loc[cvRes.rank_test_score == 1,'params'].values[0]
    bestParam['random_state'] = 0
    return bestParam

def train_test_preprocess(allData, trainTestId, selectedClinic,
                         selectedImageClinic,
                         imageNameAll,clinicNameAll,
                         transform):

    trainId = trainTestId['TrainId']
    testId = trainTestId['TestId']


    trainClinicImage = allData.loc[trainId,selectedImageClinic]
    testClinicImage = allData.loc[testId,selectedImageClinic]

    trainClinic = allData.loc[trainId,selectedClinic]
    testClinic = allData.loc[testId,selectedClinic]


    trainClinic  = clinic_preprocess(trainClinic)
    testClinic  = clinic_preprocess(testClinic)

    trainClinicImage,trainClinicImageNames = image_clinic_preprocess(
        trainClinicImage,selectedImageClinic,imageNameAll, clinicNameAll,
        transform,deleteSkew=False)
    testClinicImage,_ = image_clinic_preprocess(testClinicImage,
                                          trainClinicImageNames,
                                          imageNameAll, clinicNameAll,
                                          transform,deleteSkew=False)

    outcomeTrain = allData.loc[trainId,['Event_Hosp_to_Death', 'Time_Hosp_to_Death']]
    outcomeTest = allData.loc[testId,['Event_Hosp_to_Death', 'Time_Hosp_to_Death']]
    
    outcomeTrain = np.core.records.fromarrays(outcomeTrain.to_numpy().transpose(),names='Status, Survival_in_days',
                                         formats = 'bool, f8')
    outcomeTest = np.core.records.fromarrays(outcomeTest.to_numpy().transpose(),names='Status, Survival_in_days',
                                         formats = 'bool, f8')
    
    return (trainClinic,testClinic,trainClinicImage,testClinicImage,
            outcomeTrain,outcomeTest)


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
    
    
    allDataAll =  pd.read_csv('')

    imageNamesAll = pickle.load(open('','rb'))
    clinicNamesAll = pickle.load(open('','rb'))
    
    ids = pickle.load(open(os.path.join(dataSplitPath,'train_test_{:02d}.pkl'.format(numOfExp)),'rb'))
    
    
    selectedImageClinic = pickle.load(open(os.path.join(selectedClinicImagePath,
                                                  ''),
                                           'rb'))
    selectedClinic = pickle.load(open(os.path.join(selectedClinicPath,
                                                  ''),
                                      'rb'))

    
    (trainClinic,testClinic,trainClinicImage,
     testClinicImage,outcomeTrain,OutcomeTest) = \
        train_test_preprocess(allDataAll,ids,selectedClinic,
                              selectedImageClinic,imageNamesAll,
                              clinicNamesAll,dataTransform)
    
    if model == 'svm':
        bestClinicParam = pickle.load(open('','rb'))
        bestClinicImageParam = pickle.load(open('','rb'))
    elif model == 'cox':
        bestClinicParam = pickle.load(open('','rb'))
        bestClinicImageParam = pickle.load(open('','rb'))
    elif model == 'randomForest':
        bestClinicParam = pickle.load(open('','rb'))
        bestClinicImageParam = pickle.load(open('','rb'))
    elif model == 'gradientBoosting':
        bestClinicParam = pickle.load(open('','rb'))
        bestClinicImageParam = pickle.load(open('','rb'))
    elif model == 'Ensemble':
        bestClinicParamSvm = pickle.load(open('','rb'))
        bestClinicImageParamSvm = pickle.load(open('','rb'))
        
        bestClinicParamRsf = pickle.load(open('','rb'))
        bestClinicImageParamRsf = pickle.load(open('','rb'))
        
        bestClinicParamGb = pickle.load(open('','rb'))
        bestClinicImageParamGb = pickle.load(open('','rb'))

    if model == 'Ensemble':
        fi = get_ensemble_feature_importance(model,trainClinicImage,testClinicImage,
                               outcomeTrain,OutcomeTest,20,
                               bestClinicImageParamSvm,
                               bestClinicImageParamRsf,
                               bestClinicImageParamGb)
    else:
        fi = get_feature_importance(model,trainClinicImage,testClinicImage,
                               outcomeTrain,OutcomeTest,20,bestClinicImageParam)
        
    pickle.dump(fi, open(os.path.join(resultPath,''),'wb'))
    
    

    
    
