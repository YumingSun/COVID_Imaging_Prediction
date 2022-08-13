# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 11:39:43 2022

@author: sunym
"""
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.stats import boxcox
import pandas as pd

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
    '''

    Parameters
    ----------
    clinic : DataFrame
        Clinic data

    Returns
    -------
    clinic : DataFrame
        Fill na values with mean and normalize data

    '''
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
    '''
    Parameters
    ----------
    image : DataFrame
    imageTransform : String, optional
        Way of feature transformation. The default is None.

    Returns
    -------
    imageScaled : DataFrame
        Image features after feature transformation.

    '''
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

def image_clinic_preprocess_feature_selection(data, imageFeature, clinicFeature, 
                                              transform):
    '''
    Parameters
    ----------
    data : DataFra e
    imageFeature : List
    clinicFeature : List
    transform : String

    Returns
    -------
    output : DataFrame
        Normalize clinic features and transform image features

    '''
    
    image = data.loc[:,imageFeature]
    clinic = data.loc[:,clinicFeature]
    
    image = image_preprocess(image,transform)
    clinic = clinic_preprocess(clinic)
    
    output = clinic.merge(image,left_index = True, right_index = True)
    
    return output

def read_feature(data, selectedFeature,varTransform,deleteSkew = True):
    '''
    Parameters
    ----------
    data : DataFrame
    selectedFeature : List
    varTransform : String
    deleteSkew : BOOLEAN, optional
        Whether to delete skewed features. The default is True.

    Returns
    -------
    dataSub : DataFrame

    '''
    dataSub = data.loc[:,selectedFeature]
    if deleteSkew:
        unskewFeatures = dataSub.columns[dataSub.skew().abs() < 10]
        dataSub = dataSub.loc[:,unskewFeatures]
    dataSub = image_preprocess(dataSub,varTransform)
    return dataSub


def image_clinic_preprocess_model_fitting(data, selectedFea, imageNameAll, 
                                          clinicNameAll,transform,
                                          deleteSkew  = True):
    '''
    Parameters
    ----------
    data : DataFrame
    selectedFea : List
        Selected features including imaging and clinic features
    imageNameAll : List
        List of all the imaging features
    clinicNameAll : List
        List of all the clinic features
    transform : String
    deleteSkew : BOOLEAN, optional
        Whether to delete skewed features. The default is True.

    Returns
    -------
    clinicImage : DataFrame
        Preprocessed clinic and image features
    List
        List of selected clinic and imaging features

    '''
    selectedImage = [fea for fea in selectedFea if fea in imageNameAll]
    selectedClinic = [fea for fea in selectedFea if fea in clinicNameAll]
    
    if (len(selectedImage) == 0) and (len(selectedClinic) >0):
        clinic = data.loc[:, selectedClinic]
        clinic  = clinic_preprocess(clinic)
        output = clinic
    elif (len(selectedImage) > 0) and (len(selectedClinic) == 0):
        image = read_feature(data,selectedImage,transform,deleteSkew)
        output = image
    elif (len(selectedImage) > 0) and (len(selectedClinic) > 0):
        image = read_feature(data,selectedImage,transform,deleteSkew)
        clinic = data.loc[:, selectedClinic]
        clinic  = clinic_preprocess(clinic)
        output = image.join(clinic)
    else:
        raise ValueError("No feature is selected")
    return output,output.columns.tolist()



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
    
    return (trainClinic,testClinic,trainClinicImage,testClinicImage,
            outcomeTrain,outcomeTest)

def train_test_preprocess_feature_importance(allData, trainTestId, selectedClinic,
                                             selectedImage,transform):

    trainId = trainTestId['TrainId']
    testId = trainTestId['TestId']

    trainClinic = allData.loc[trainId,selectedClinic]
    testClinic = allData.loc[testId,selectedClinic]

    trainImage = allData.loc[trainId,selectedImage]
    testImage = allData.loc[testId,selectedImage]

    trainClinic  = clinic_preprocess(trainClinic)
    testClinic  = clinic_preprocess(testClinic)

    trainImage  = image_preprocess(trainImage,transform)
    testImage  = image_preprocess(testImage,transform)


    outcomeTrain = allData.loc[trainId,['Event_Hosp_to_Death', 'Time_Hosp_to_Death']]
    outcomeTest = allData.loc[testId,['Event_Hosp_to_Death', 'Time_Hosp_to_Death']]

    outcomeTrain = np.core.records.fromarrays(outcomeTrain.to_numpy().transpose(),names='Status, Survival_in_days',
                                         formats = 'bool, f8')
    outcomeTest = np.core.records.fromarrays(outcomeTest.to_numpy().transpose(),names='Status, Survival_in_days',
                                         formats = 'bool, f8')

    return (trainClinic,testClinic,trainImage,testImage,
            outcomeTrain,outcomeTest)


if __name__ == '__main__':
    pass