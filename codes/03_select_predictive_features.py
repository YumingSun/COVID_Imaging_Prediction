# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 23:33:31 2022

@author: sunym
"""
from sksurv.linear_model import CoxPHSurvivalAnalysis
import numpy as np
from sksurv.metrics import integrated_brier_score
from sklearn.model_selection import GridSearchCV, KFold
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
from scipy.stats import boxcox
from sklearn.feature_selection import SequentialFeatureSelector

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
    if len(ctsClinicIndex) > 0:
        ctsClinic = clinic[ctsClinicIndex]
        
        scaler = StandardScaler().fit(ctsClinic.to_numpy())
        ctsClinicScaled = scaler.transform(ctsClinic.to_numpy())
        
        clinic[ctsClinicIndex] = ctsClinicScaled
    
    return clinic

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

def image_clinic_preprocess(data, selectedFea, imageNameAll, clinicNameAll,
                            transform):
    selectedImage = [fea for fea in selectedFea if fea in imageNameAll]
    selectedClinic = [fea for fea in selectedFea if fea in clinicNameAll]
    
    clinic = data.loc[:, selectedClinic]
    image = data.loc[:, selectedImage]
    
    clinic = clinic_preprocess(clinic)
    image = image_preprocess(image,transform)
    
    clinicImage = image.merge(clinic,left_index=True, right_index=True)
    
    return clinicImage,clinicImage.columns.tolist()

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

    selectedClinicImagePath = ''
    selectedClinicPath = ''
    
    allDataAll =  pd.read_csv('')

    imageNamesAll = pickle.load(open('','rb'))
    clinicNamesAll = pickle.load(open('','rb'))
    
    selectedImageClinic = pickle.load(open('','rb'))
    selectedClinic = pickle.load(open('','rb'))
    
    outcome = allDataAll.loc[:,['Event_Hosp_to_Death', 'Time_Hosp_to_Death']]
    clinic = allDataAll.loc[:,selectedClinic]
    clinic  = clinic_preprocess(clinic)
    
    clinicImage = allDataAll.loc[:,selectedImageClinic]
    clinicImage,_ = image_clinic_preprocess(clinicImage,selectedImageClinic,
                                          imageNamesAll,clinicNamesAll,
                                          dataTransform)

    predictiveClinicFeatures = get_predictive_features(clinic,outcome)
    print(predictiveClinicFeatures)
    predictiveImageClinicFeatures = get_predictive_features(clinicImage,
                                                            outcome)
    print(predictiveImageClinicFeatures)
    
    pickle.dump(predictiveClinicFeatures,
                open(os.path.join(selectedClinicPath,
                                  ''),
                     'wb'))
    pickle.dump(predictiveImageClinicFeatures,
                open(os.path.join(selectedClinicImagePath,
                                  ''),
                     'wb'))