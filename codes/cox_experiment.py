# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 00:00:36 2022

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
from scipy.stats import boxcox


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


def evaluate_performance(allData, trainTestId, selectedClinic,
                         selectedImageClinic,
                         imageNameAll,clinicNameAll,
                         transform):
    '''
    Parameters
    ----------
    allData : DataFrame
    trainTestId : Dict
        Patient Ids used to split testing and training dataset
    selectedClinic : List
    selectedImageClinic : List
    imageNameAll : List
    clinicNameAll : List
    transform : String

    Returns
    -------
    Dictionary of C-index

    '''

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
    
    clinicImageCox = CoxPHSurvivalAnalysis(n_iter = 200).fit(trainClinicImage, 
                                               outcomeTrain)
    clinicCox = CoxPHSurvivalAnalysis(n_iter = 200).fit(trainClinic, 
                                               outcomeTrain)

    clinicResTrain = clinicCox.score(trainClinic,outcomeTrain)
    clinicResTest = clinicCox.score(testClinic,outcomeTest)
    
    clinicImageResTrain = clinicImageCox.score(trainClinicImage,
                                               outcomeTrain)
    clinicImageResTest = clinicImageCox.score(testClinicImage,
                                              outcomeTest)
    
        
    return {'Clinic Train': clinicResTrain, 'Clinic Test': clinicResTest,
            'Clinic Image Train': clinicImageResTrain, 
            'Clinic Image Test': clinicImageResTest
            }


if __name__ == '__main__':
    numOfExp = int(sys.argv[1])
    dataTransform = sys.argv[2]
    
    dataSplitPath = ''
    dataPath = ''
    
    selectedClinicImagePath = ''
    selectedClinicPath = ''

    
    resultPath = ''
    allDataAll =  pd.read_csv(os.path.join(dataPath,''))
    
    ids = pickle.load(open(os.path.join(dataSplitPath,'train_test_{:02d}.pkl'.format(numOfExp)),'rb'))

    selectedImageClinic = pickle.load(open(os.path.join(selectedClinicImagePath,
                                                  ''),
                                           'rb'))
    
    imageNamesAll = pickle.load(open(os.path.join(dataPath,
                                               ''),'rb'))
    clinicNamesAll = pickle.load(open(os.path.join(dataPath,
                                                   ''),'rb'))

    selectedClinic = [f for f in selectedImageClinic if f in clinicNamesAll]
    
    results = evaluate_performance(allDataAll,ids,selectedClinic,
                                   selectedImageClinic,
                                   imageNamesAll,
                                   clinicNamesAll,
                                   dataTransform)
    
    pickle.dump(results,open(os.path.join(resultPath,
                                           'performance_{:02d}.pkl'.format(numOfExp)),'wb'))
