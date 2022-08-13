# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 23:33:31 2022

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
from sklearn.feature_selection import SequentialFeatureSelector
from preprocess import clinic_preprocess,image_clinic_preprocess_model_fitting

def get_predictive_features(features,outcome):
    '''

    Parameters
    ----------
    features : DataFrame
    outcome : DataFrame

    Returns
    -------
    selectedFeatures : List
        Selected Features by forward selection based on C-index

    '''
    outcomeCensor = np.core.records.fromarrays(outcome.to_numpy().transpose(),names='Status, Survival_in_days',
                                             formats = 'bool, f8')
    sfs = SequentialFeatureSelector(CoxPHSurvivalAnalysis(n_iter = 200 ), 
                                    n_features_to_select="auto",
                                    tol = 1e-4, cv = 5,
                                    n_jobs = -1)
    sfs.fit(features, outcomeCensor)
    selectedFeatures = features.columns[sfs.get_support()].tolist()
    return selectedFeatures



if __name__ == '__main__':
    dataTransform = sys.argv[1]
    
    dataPath = ''
    selectedClinicImagePath = ''
    selectedClinicPath = ''
    
    allDataAll =  pd.read_csv(os.path.join(dataPath,'data.csv'))

    imageNamesAll = pickle.load(open(os.path.join(dataPath,'imageFeatureNames.pkl'),'rb'))
    clinicNamesAll = pickle.load(open(os.path.join(dataPath,'clinicFeatureNames.pkl'),'rb'))
    
    selectedImageClinic = pickle.load(open(os.path.join(
        selectedClinicImagePath,'lowCorImageClinic.pkl'),'rb'))
    selectedClinic = pickle.load(open(os.path.join(
        selectedClinicPath,'lowCorClinic.pkl'),'rb'))
    
    outcome = allDataAll.loc[:,['Event_Hosp_to_Death', 'Time_Hosp_to_Death']]
    clinic = allDataAll.loc[:,selectedClinic]
    clinic  = clinic_preprocess(clinic)
    
    clinicImage = allDataAll.loc[:,selectedImageClinic]
    clinicImage,_ = image_clinic_preprocess_model_fitting(clinicImage,selectedImageClinic,
                                          imageNamesAll,clinicNamesAll,
                                          dataTransform)

    predictiveClinicFeatures = get_predictive_features(clinic,outcome)
    print(predictiveClinicFeatures)
    predictiveImageClinicFeatures = get_predictive_features(clinicImage,
                                                            outcome)
    print(predictiveImageClinicFeatures)
    
    pickle.dump(predictiveClinicFeatures,
                open(os.path.join(selectedClinicPath,
                                  'selectedClinic.pkl'),
                     'wb'))
    pickle.dump(predictiveImageClinicFeatures,
                open(os.path.join(selectedClinicImagePath,
                                  'selectedImageClinic.pkl'),
                     'wb'))