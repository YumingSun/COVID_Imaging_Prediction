# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 23:09:37 2022

@author: sunym
"""
import os
from sksurv.linear_model import CoxPHSurvivalAnalysis
import pickle
import sys
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

def image_clinic_preprocess(data, imageFeature, clinicFeature,
                            transform):
    
    image = data.loc[:,imageFeature]
    clinic = data.loc[:,clinicFeature]
    
    image = image_preprocess(image,transform)
    clinic = clinic_preprocess(clinic)
    
    output = clinic.merge(image,left_index = True, right_index = True)
    
    return output



def delete_bad_variables(data):
    # delete variables that are same for all patients
    data = data.loc[:,(data != data.iloc[0]).any()] 
    # delete variables that are infinity and na
    data = data.loc[:, ~data.isin([np.nan, np.inf, -np.inf]).any('index')]
    # delete skew variables
    unskewFeatures = data.columns[data.skew().abs() < 10]
    data = data.loc[:,unskewFeatures]
    return data

def sort_by_c(data,outcome):
    '''

    Parameters
    ----------
    data : DataFrame
    outcome : DataFrame
        Survival outcome

    Returns
    -------
    data_sort : DataFrame
        Data with features sorted by C-index
    c_index_sort : DataFrame
        Sorted features and corresponding C-index

    '''
    featureName = data.columns.tolist()
    c_index_df = pd.DataFrame(index = featureName,
                              columns = ['c_index'])
    for f in featureName:
        coxData = outcome.merge(data.loc[:,[f]],left_index = True,
                                right_index = True)
        
        features = coxData.loc[: ,[f]]
        outcomeTrain = np.core.records.fromarrays(
            coxData.loc[:,
                        ['Event_Hosp_to_Death', 'Time_Hosp_to_Death']].to_numpy().transpose(),
            names='Status, Survival_in_days',formats = 'bool, f8')
        
        model = CoxPHSurvivalAnalysis(n_iter=200).fit(features,outcomeTrain)
        
        c_index_df.loc[f,'c_index'] = model.score(features,outcomeTrain)
        
    
    c_index_sort = c_index_df.dropna().sort_values(by = ['c_index'],
                                                   ascending = True)
    
    data_sort = data.loc[:,c_index_sort.index]
    
    return data_sort, c_index_sort

def drop_high_corr(all_data,threshold):
    '''
    Parameters
    ----------
    all_data : DataFrame
        Data with features sorted by C-index
    threshold : Float
        The threshod of high correlation

    Returns
    -------
    all_data_no_high_corr : DataFrame
        Features correlation are less than threshold
    high_corr_index_all : List
        Dropped features

    '''
    corr_matrix = all_data.corr()
    high_corr_index_all = []
    for i in range(1,len(all_data.columns)):
        high_corr_index = (corr_matrix.columns[:i])[corr_matrix.iloc[i,:i] > threshold]
        high_corr_index_all.extend(list(high_corr_index)) 
        high_corr_index = (corr_matrix.columns[:i])[corr_matrix.iloc[i,:i] < -threshold] 
        high_corr_index_all.extend(list(high_corr_index))
        
    high_corr_index_all = list(set(high_corr_index_all))
    all_data_no_high_corr = all_data.drop(columns = high_corr_index_all)
    return all_data_no_high_corr,high_corr_index_all


if __name__ == '__main__':
    dataTransform = sys.argv[1]

    clinicPath = ''
    imageClinicPath = ''
    dataPath = ''
    
    allData =  pd.read_csv('')
    
    imageNames = pickle.load(open(os.path.join(dataPath,'imageFeatureNames.pkl'),'rb'))
    clinicNames = pickle.load(open(os.path.join(dataPath,'clinicFeatureNames.pkl'),'rb'))
    
    imageSub = allData.loc[:,imageNames]
    clinicSub = allData.loc[:,clinicNames]
    outcomeSub = allData.loc[:,['Event_Hosp_to_Death', 'Time_Hosp_to_Death']]
    
    clinicImageSub = clinicSub.merge(imageSub, left_index = True,
                                     right_index = True)
    
    clinic = clinic_preprocess(clinicSub)
    clinicImage = image_clinic_preprocess(clinicImageSub,imageNames,clinicNames,
                                          dataTransform)
    
    clinic = delete_bad_variables(clinic)
    clinicImage = delete_bad_variables(clinicImage)
    
    clinic,_ = sort_by_c(clinic,outcomeSub)
    clinicImage,_ = sort_by_c(clinicImage,outcomeSub)
    
    clinicLowCor,_ = drop_high_corr(clinic,0.75)
    clinicImageCor,_ = drop_high_corr(clinicImage, 0.75)
    
    pickle.dump(clinicLowCor.columns.tolist(),
                open(os.path.join(clinicPath,
                                  ''),
                     'wb'))
    pickle.dump(clinicImageCor.columns.tolist(),
                open(os.path.join(imageClinicPath,
                                  ''),
                     'wb'))
