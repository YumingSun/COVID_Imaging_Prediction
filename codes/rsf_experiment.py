# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 15:36:53 2022

@author: sunym
"""
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
from sksurv.ensemble import RandomSurvivalForest
from preprocess import clinic_preprocess,image_clinic_preprocess_model_fitting

def evaluate_performance(allData, trainTestId, selectedClinic,
                         selectedImageClinic,
                         imageNameAll,clinicNameAll,
                         transform,clinicParam, clinicImageParam):
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
    clinicParam : Dict
    clinicImageParam : Dict

    Returns
    -------
    dict
        ictionary of C-index for training and testing dataset

    '''

    trainId = trainTestId['TrainId']
    testId = trainTestId['TestId']
    
    
    trainClinicImage = allData.loc[trainId,selectedImageClinic]
    testClinicImage = allData.loc[testId,selectedImageClinic]
    
    trainClinic = allData.loc[trainId,selectedClinic]
    testClinic = allData.loc[testId,selectedClinic]
    
    
    trainClinic  = clinic_preprocess(trainClinic)
    testClinic  = clinic_preprocess(testClinic)
    
    trainClinicImage,trainClinicImageNames = image_clinic_preprocess_model_fitting(
        trainClinicImage,selectedImageClinic,imageNameAll, clinicNameAll,
        transform,deleteSkew=False)
    testClinicImage,_ = image_clinic_preprocess_model_fitting(testClinicImage,
                                          trainClinicImageNames,
                                          imageNameAll, clinicNameAll,
                                          transform,deleteSkew=False)
    
    outcomeTrain = allData.loc[trainId,['Event_Hosp_to_Death', 'Time_Hosp_to_Death']]
    outcomeTest = allData.loc[testId,['Event_Hosp_to_Death', 'Time_Hosp_to_Death']]
    
    outcomeTrain = np.core.records.fromarrays(outcomeTrain.to_numpy().transpose(),names='Status, Survival_in_days',
                                         formats = 'bool, f8')
    outcomeTest = np.core.records.fromarrays(outcomeTest.to_numpy().transpose(),names='Status, Survival_in_days',
                                         formats = 'bool, f8')
    
    clinicImageRsf = RandomSurvivalForest(**clinicImageParam).fit(trainClinicImage,
                                                                             outcomeTrain)
    clinicRsf = RandomSurvivalForest(**clinicParam).fit(trainClinic,
                                                                             outcomeTrain)

    clinicResTrain = clinicRsf.score(trainClinic,outcomeTrain)
    clinicResTest = clinicRsf.score(testClinic,outcomeTest)

    clinicImageResTrain = clinicImageRsf.score(trainClinicImage,
                                               outcomeTrain)
    clinicImageResTest = clinicImageRsf.score(testClinicImage,
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
    paramPath = ''
    
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
    
    bestClinicParam = pickle.load(open(os.path.join(paramPath,'clinicParamRsf.pkl'),'rb'))
    bestClinicImageParam = pickle.load(open(os.path.join(paramPath,'clinicImageParamRsf.pkl'),'rb'))
       
    
    results = evaluate_performance(allDataAll,ids,selectedClinic,
                                   selectedImageClinic,
                                   imageNamesAll,
                                   clinicNamesAll,
                                   dataTransform,
                                   bestClinicParam,
                                   bestClinicImageParam)

    pickle.dump(results,open(os.path.join(resultPath,
                                           'performance_{:02d}.pkl'.format(numOfExp)),'wb'))
