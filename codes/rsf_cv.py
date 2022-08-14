# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 12:49:39 2022

@author: sunym
"""
import numpy as np
from sksurv.metrics import integrated_brier_score
from sklearn.model_selection import GridSearchCV
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
from preprocess import clinic_preprocess,image_clinic_preprocess_model_fitting
from stratified_cv import stratified_cv


def rf_tuner(data,outcome,trainSize = 0.8, nFold = 5):
    '''
    Parameters
    ----------
    data : DataFrame
        Features.
    outcome : DataFrame
        first column is event indicator
        second clolumn is survival time
    trainSize : Float, optional
        DESCRIPTION. The default is 0.8.
    nFold : Int, optional
        DESCRIPTION. The default is 5.

    Returns
    -------
    dataframe, cross validation results

    '''
    
    outcomeCensor = np.core.records.fromarrays(outcome.to_numpy().transpose(),names='Status, Survival_in_days',
                                             formats = 'bool, f8') 
    param_grid = {
        'max_features': ['sqrt', 'log2', None, 0.2, 0.5, 0.8],
        'n_estimators': [10,100,200,300,500]
    }
    
    rsf = RandomSurvivalForest(n_jobs=-1)
    rsf_grid = GridSearchCV(estimator = rsf,
                            param_grid = param_grid,
                            cv = stratified_cv(outcome,trainSize=trainSize, nFold = nFold),
                            n_jobs = -1)
    rsf_grid.fit(data, outcomeCensor)
    cvRes = pd.DataFrame(rsf_grid.cv_results_)
    return cvRes


if __name__ == '__main__':
    numOfExp = int(sys.argv[1])
    dataTransform = sys.argv[2]
    var = sys.argv[3]
    
    trainSize = 0.8
    dataSplitPath = ''
    dataPath = ''
    
    selectedClinicImagePath = ''
    selectedClinicPath = ''

    paramPath = ''
    resultPath = ''
    
    allDataAll =  pd.read_csv(os.path.join(dataPath,'data.csv'),
                           index_col = ['PatientID'])
    
    ids = pickle.load(open(os.path.join(dataSplitPath,'train_test_{:02d}.pkl'.format(numOfExp)),'rb'))
   
    selectedImageClinic = pickle.load(open(os.path.join(selectedClinicImagePath,
                                                  'selectImageClinic.pkl'),
                                           'rb'))
    
    imageNamesAll = pickle.load(open(os.path.join(dataPath,
                                               'imageFeatureNames.pkl'),'rb'))
    clinicNamesAll = pickle.load(open(os.path.join(dataPath,
                                                   'clinicFeatureNames.pkl'),'rb'))

    selectedClinic = [f for f in selectedImageClinic if f in clinicNamesAll]
    
    trainId = ids['TrainId']
    allDataAll  = allDataAll.loc[trainId,:]

    clinic = allDataAll.loc[:,selectedClinic]
    clinic = clinic_preprocess(clinic)
    clinicImage,_ = image_clinic_preprocess_model_fitting(
        allDataAll,selectedImageClinic,imageNamesAll, clinicNamesAll,
        dataTransform,deleteSkew=False)
    
    outcome = allDataAll[['Event_Hosp_to_Death', 'Time_Hosp_to_Death']]
    
    if var == 'clinic':
        cvRes = rf_tuner(data = clinic, outcome = outcome,
                               trainSize = trainSize, nFold = 5)
    elif var == 'clinicImage':
        cvRes = rf_tuner(data = clinicImage, outcome = outcome,
                               trainSize = trainSize, nFold = 5)
    
    pickle.dump(cvRes,open(
        os.path.join(resultPath,
                     'cv_{}_Exp{:02d}.pkl'.format(var,numOfExp)),
        'wb'))

