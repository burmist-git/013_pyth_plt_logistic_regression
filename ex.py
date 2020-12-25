#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Date        : 
Autor       : Leonid Burmistrov
Description : Simple reminder-training example.
'''

import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

np.random.seed(1234567890)

def generateData(nn,x0,sigma,classType):
    x = np.array([sigma[i]*np.random.randn(nn[i], 1) + x0[i] for i in range(len(nn))])
    y = np.array([classType[i]*np.ones(nn[i]) for i in range(len(x0))])

    #x = np.arange(10).reshape(-1, 1)
    #y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    return x, y 

def getHistBins(nBins,xMin,xMax):
    binW = (xMax - xMin)/nBins
    return [xMin + binW*i for i in range(nBins+1)]

def plot(datax,datay,bins):

    nPlots=len(datax) + 2
    ncols=math.ceil(nPlots**(0.5))
    nrows=int(nPlots/ncols)+math.ceil(nPlots/ncols-int(nPlots/ncols))
    #print(ncols)
    #print(nrows)

    # using the variable axs for multiple Axes
    fig, axs = plt.subplots( nrows=nrows, ncols=ncols,
                             sharex=False, sharey=False,
                             squeeze=True, subplot_kw=None,
                             gridspec_kw=None,
                             figsize=(18,8))
    plt.tight_layout()

    for i in range(nrows):
        for j in range(ncols):
            plotID=i*ncols + j
            if(plotID<len(datax)):
                axs[i,j].hist(datax[plotID], bins=bins)
            elif (plotID == len(nn)):
                axs[i,j].hist(np.vstack(datax), bins=bins, weights=(1.0/700*np.ones(np.vstack(datax).shape)))
            elif (plotID == (len(nn)+1)):
                axs[i,j].scatter(np.vstack(datax),np.hstack(datay))
    
    plt.show()


if __name__ == "__main__":

    classType = [0,1]
    nn    = [10000,10000]
    x0    = [1,2]
    sigma = [0.3,0.3]
    nPoints = 4
    nBins = 100
    nSigma = 4
    
    datax, datay = generateData(nn,x0,sigma,classType)
    print('')
    print('datax')
    print('datax.ndim  = ', datax.ndim)
    print('datax.shape = ', datax.shape)
    print('type(datax) = ', type(datax))
    print('datax = ', datax)
    print('')
    print('datay')
    print('datay.ndim  = ', datay.ndim)
    print('datay.shape = ', datay.shape)
    print('type(datay) = ', type(datay))
    print('datay = ', datay)
    
    xMin = int(np.array(x0).min() - math.ceil(nSigma*np.array(sigma).max()))
    xMax = int(np.array(x0).max() + math.ceil(nSigma*np.array(sigma).max()))
    bins = getHistBins(nBins,xMin,xMax)
    #print(len(bins))

    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(datax.reshape((-1,1)), datay.reshape((-1,)))

    print('model.classes_   = ',model.classes_)
    print('model.intercept_ = ',model.intercept_) 
    print('model.coef_      = ',model.coef_)


    plot(datax,datay,bins)
    

    '''
    #np.column_stack((D_model, np.ones(D.shape[0])))
    x = np.ones(nPoints)
    x = np.column_stack((x,np.linspace(xMin, xMax, nPoints)))
    #x = np.column_stack((x,np.linspace(xMin, xMax, nPoints)))
    #x = np.expand_dims(np.linspace(xMin, xMax, nPoints),axis=1)
    #k = np.expand_dims(np.array([1,2,0.3]),axis=1)
    #print('theta.ndim  = ', k.ndim)
    #print('theta.shape = ', k.shape)
    print('x.ndim      = ', x.ndim)
    print('x.shape     = ', x.shape)
    print('x           = ', x)
    '''
    #x = np.arange(10)
    #x = np.stack((x,np.arange(10)))
    #x = x.reshape(-1,1,2)
    #x = np.squeeze(x)
    #print('x.ndim      = ', x.ndim)
    #print('x.shape     = ', x.shape)
    #print('x           = ', x)
