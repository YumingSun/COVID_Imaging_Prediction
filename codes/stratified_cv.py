# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 15:30:10 2022

@author: sunym
"""
import numpy as np
from sklearn.model_selection import train_test_split

def stratified_cv(data,trainSize = 0.8, nFold = 5):
    '''

    Parameters
    ----------
    data : dataframe
        the first column is event indicator
    train_size : float
        The default is 0.8.
    nFold : float
        The default is 5.

    Returns
    -------
    An iterable of length n_folds, each element of which is a 2-tuple of numpy 
    1-d arrays (train_index, test_index) containing the indices of the test and 
    training sets for that cross-validation run

    '''
    id1 = np.where(data.iloc[:,0] == True)[0]
    id0 = np.where(data.iloc[:,0] == False)[0]
    myCvIterator = []
    for i in range(nFold):
        id1_train,id1_test = train_test_split(id1, train_size=trainSize)
        id0_train,id0_test = train_test_split(id0, train_size=trainSize)
        idTrain = np.concatenate((id1_train,id0_train))
        np.random.shuffle(idTrain)
        idTest = np.concatenate((id1_test,id0_test))
        np.random.shuffle(idTest)
        myCvIterator.append( (idTrain,idTest) )
    return myCvIterator

if __name__ == '__main__':
    pass