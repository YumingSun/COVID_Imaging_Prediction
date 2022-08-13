# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 15:32:03 2022

@author: sunym
"""
from sksurv.linear_model import CoxPHSurvivalAnalysis
import numpy as np
from sksurv.metrics import integrated_brier_score
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
from preprocess import train_test_preprocess_feature_importance
from ensemble_averaging import Ensemble

def get_feature_importance(model,x_train,x_test,y_train,y_test,niter):
    if model == 'svm':
        est = FastKernelSurvivalSVM()
    elif model == 'cox':
        est = CoxPHSurvivalAnalysis(n_iter = 200 )
    elif model == 'randomForest':
        est = RandomSurvivalForest()
    elif model == 'gradientBoosting':
        est = GradientBoostingSurvivalAnalysis()
    elif model == 'Ensemble':
        est = Ensemble()

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


    
    (trainClinic,testClinic,trainImage,testImage,
     outcomeTrain,outcomeTest) = \
        train_test_preprocess_feature_importance(allDataAll,ids,selectedClinic,
                                                 selectedImage,dataTransform)
    
    fiClinic = get_feature_importance(model,trainClinic,testClinic,
                           outcomeTrain,outcomeTest,20)
    fiImage = get_feature_importance(model,trainImage,testImage,
                           outcomeTrain,outcomeTest,20)

    fi = pd.concat([fiClinic,fiImage])

    pickle.dump(fi, 
                open(os.path.join(resultPath,
                                  'results_{:02d}.pkl'.format(numOfExp)),
                     'wb'))
    
    

    
    
