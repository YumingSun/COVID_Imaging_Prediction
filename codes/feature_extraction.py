# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 22:30:22 2022

@author: sunym
"""
import pydicom
import numpy as np
import os
import pandas as pd
from pydicom.pixel_data_handlers.util import apply_voi_lut
import pickle
from radiomics import featureextractor
import SimpleITK as sitk
import sys
import re
import gc
import cv2

def read_scale_xray(file,voi_lut = True,fix_monochrome = True):
    '''

    Parameters
    ----------
    file : STRING
        Location of the dicom files
    voi_lut : BOOLEAN, optional
        The default is True.
    fix_monochrome : BOOLEAN, optional
       The default is True.

    Returns
    -------
    data : Numpy Array
        X-ray images with pixel intensity between 0-255

    '''
    dicom = pydicom.dcmread(file)
    
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array,dicom)
    else:
        data = dicom.pixel_array
    
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    
    data = data - np.min(data)
    data = data/np.max(data)
    data = (data*255).astype(np.uint8)
    return data

def extract_one_image_feature(path,extractor):
    '''

    Parameters
    ----------
    path : STRING
        Location of the dicom files
    extractor : feature extractor

    Returns
    -------
    features : DataFrame
    '''
    try:
        data1 = read_scale_xray(path)
        if len(data1.shape) > 2:
            data1 = data1[-1]
        data1 = cv2.equalizeHist(data1)
        header = pydicom.dcmread(path,stop_before_pixels = True)
        segAll = np.ones(data1.shape,dtype = int)
        segAll[0,0] = 0
        image_itk = sitk.GetImageFromArray(data1)
        seg_itk = sitk.GetImageFromArray(segAll)
        result = extractor.execute(image_itk,seg_itk)
        result_rmv_general = {key: val for key,val in result.items() if 'diagnostics' not in key}
        result_rmv_general['PatientID'] = header['PatientID'].value
        result_rmv_general['fileAd'] = path
        result_rmv_general['error'] = 0
        result_rmv_general['errorMessage'] = None
        features = pd.DataFrame.from_dict(result_rmv_general,orient = 'index')
        features = features.T
    except Exception as ex:
        features = pd.DataFrame([[path,1,str(ex)]],
                                  columns = ['fileAd','error','errorMessage'])
    return features

if __name__ == '__main__':
    imageAdLoc = ''
    resLoc = ''
    extractor = featureextractor.RadiomicsFeatureExtractor('Params2D.yaml')
    allPateints = pickle.load(open('','rb'))
    patients = allPateints[int(sys.argv[1])]
    imageFeatures = pd.DataFrame()
    i = 0
    for patient in patients:
        i = i + 1
        print(i)
        imageFeatures = imageFeatures.append(extract_one_image_feature(patient,extractor))
        gc.collect()

    firstList = re.split(r'/',patients[0])[7:9]
    firstStr = '-'.join(firstList)
    lastList = re.split(r'/',patients[-1])[7:9]
    lastStr = '-'.join(lastList)
    pickle.dump(imageFeatures,open(os.path.join(resLoc,'ImageFeatures_{}_{}.pkl'.format(firstStr,lastStr)),'wb'))
