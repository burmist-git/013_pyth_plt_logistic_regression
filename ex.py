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
from optparse import OptionParser
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import classification_report, confusion_matrix

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

def sigmoidFunction(k,x):
    pol=polynomialFunction(k,x)
    return 1/(1+np.exp(-pol))

def polynomialFunction(k,x):
    return np.dot(k,x)

def printArrayInfo(ar,ar_name,printkey=False):
    print('')
    print(ar_name)
    print('{}.ndim  = '.format(ar_name), ar.ndim)
    print('{}.shape = '.format(ar_name), ar.shape)
    print('type({}) = '.format(ar_name), type(ar))
    if printkey :
        print('{} = '.format(ar_name), ar)

def plot(nn,datax,datay,bins,weights,binsCenter,y,y_s):

    #printArrayInfo(y,'y')
    #printArrayInfo(y_s,'y_s')
    
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
                axs[i,j].hist(np.vstack(datax), bins=bins, weights=(weights*np.ones(np.vstack(datax).shape)),alpha=0.5)
                #axs[i,j].scatter(binsCenter,y)
                if type(binsCenter) != 'NoneType':
                    axs[i,j].scatter(binsCenter,y_s,facecolor='blue')
            elif (plotID == (len(nn)+1)):
                axs[i,j].scatter(np.vstack(datax),np.hstack(datay))
    
    plt.show()

def pol01():
    classType = [0,1]
    nn = [10000,10000]
    x0 = [-6-10,3]
    sigma = [0.6,3.0]
    nPoints = 4
    nBins = 100
    nSigma = 4
    
    datax, datay = generateData(nn,x0,sigma,classType)
    printArrayInfo(datax,'datax',printkey=False)
    printArrayInfo(datay,'datay',printkey=False)
    
    xMin = int(np.array(x0).min() - math.ceil(nSigma*np.array(sigma).max()))
    xMax = int(np.array(x0).max() + math.ceil(nSigma*np.array(sigma).max()))
    bins = getHistBins(nBins,xMin,xMax)
    binsCenter = (bins[1]-bins[0])/2.0 + np.array(bins)
    binsCenter = binsCenter[0:(len(binsCenter)-1)]    
    #print(len(bins))

    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(datax.reshape((-1,1)), datay.reshape((-1,)))

    print('')
    print('model.classes_   = ',model.classes_)
    print('model.intercept_ = ',model.intercept_) 
    print('model.coef_      = ',model.coef_)

    k = np.hstack(np.array([np.squeeze(model.intercept_),np.squeeze(model.coef_)]))
    printArrayInfo(k,'k',printkey=True)

    y = polynomialFunction(k, (np.row_stack((np.ones(len(binsCenter)),binsCenter))))
    y_s = sigmoidFunction(k, (np.row_stack((np.ones(len(binsCenter)),binsCenter))))
    
    plot(nn,datax,datay,bins,1.0/700,binsCenter,y,y_s)
    #plot(datax,datay,bins,1.0/700,binsCenter=None,y=None,y_s=None)

def pol02():
    classType = [0,1,0]
    nn = [10000,10000,10000]
    x0 = [-6-10,3,6+10]
    sigma = [0.6,0.1,0.6]
    nPoints = 4
    nBins = 100
    nSigma = 4
    
    datax, datay = generateData(nn,x0,sigma,classType)
    printArrayInfo(datax,'datax',printkey=False)
    printArrayInfo(datay,'datay',printkey=False)

    printArrayInfo(datax.reshape((-1,1)),'datax.reshape((-1,1))',printkey=False)
    datax_r=np.column_stack([datax.reshape((-1,1)),datax.reshape((-1,1))*datax.reshape((-1,1))])
    printArrayInfo(datax_r,'datax_r',printkey=False)
    
    xMin = int(np.array(x0).min() - math.ceil(nSigma*np.array(sigma).max()))
    xMax = int(np.array(x0).max() + math.ceil(nSigma*np.array(sigma).max()))
    bins = getHistBins(nBins,xMin,xMax)
    binsCenter = (bins[1]-bins[0])/2.0 + np.array(bins)
    binsCenter = binsCenter[0:(len(binsCenter)-1)]    
    #print(len(bins))

    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(datax_r, datay.reshape((-1,)))

    print('')
    print('model.classes_   = ',model.classes_)
    print('model.intercept_ = ',model.intercept_) 
    print('model.coef_      = ',model.coef_)

    k = np.hstack([np.squeeze(model.intercept_),np.squeeze(model.coef_)])
    printArrayInfo(k,'k',printkey=True)
    #printArrayInfo(np.row_stack((np.ones(len(binsCenter)),binsCenter)),'a',printkey=False)
    #printArrayInfo(np.row_stack((np.row_stack((np.ones(len(binsCenter)),binsCenter)), binsCenter*binsCenter)),'b',printkey=False)
    y = polynomialFunction(k, np.row_stack((np.row_stack((np.ones(len(binsCenter)),binsCenter)), binsCenter*binsCenter)))
    y_s = sigmoidFunction(k, np.row_stack((np.row_stack((np.ones(len(binsCenter)),binsCenter)), binsCenter*binsCenter)))
    
    plot(nn,datax,datay,bins,1.0/700,binsCenter,y,y_s)
    #plot(datax,datay,bins,1.0/700,binsCenter=None,y=None,y_s=None)

parser = OptionParser()
parser.add_option('-p', '--pol',
                  dest='poldeg', type="int",default=1,
                  help="polinom degree")

(options, args) = parser.parse_args()
    
if __name__ == "__main__":
    if options.poldeg == 1:
        pol01()
    elif options.poldeg == 2:
        pol02()
    else :
        print(options)
        print(type(options))
        print(args)
        print(type(args))
        print('len(args) = ',len(args))
        print('type(options.poldeg)        = ',type(options.poldeg))
        print('parser.get_usage()          = ',parser.get_usage())
        print('parser.get_default_values() = ',parser.get_default_values())
        print('parser.get_prog_name()      = ',parser.get_prog_name())
        #print(parser.get_version())
        #print(parser.get_description())
        #print(parser.get_option_group())
