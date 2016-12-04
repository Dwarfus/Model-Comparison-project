#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 10:55:38 2016

@author: donukb

This is (hopefully) the (semi)final version of this code.
~ Only applies to 2D gaussian distribution ~
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import math
from scipy import integrate
import os.path
from scipy.optimize import curve_fit
from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy.special import binom
from scipy.optimize import curve_fit, minimize_scalar



class Generate():
    """
    Generates data from a 2D Gaussian distribution
    """
    def __init__(self, dimension, npoints, mean = [0.0,0.0], variance = [1.0,1.0]):
        self.dim = dimension
        self.npoints = npoints        
        self.data = np.array([[0.0]*self.dim]*self.npoints)
        self.means = mean
        self.variances = variance
        
        self.gaussian()
        #self.writetofile()
        #self.testplot()
        
    def gaussian(self):
        j=0
        while (j < self.npoints):
            k=0
            while (k < self.dim):                
                self.data[j,k]=np.random.normal(self.means[k], self.variances[k])
                k+=1       
            j+=1
            
    #def writetofile(self):
        #file = open('GaussianData.txt','w')
        #file.write(self.data) <= needs to be string argument, change later.
        #file.close()
        
    #def testplot(self):
        #self.plotdata = [[a for a,b in self.data],[b for a,b in self.data]]
        #plt.figure(1)
        #self.testplot = plt.scatter(self.plotdata[0],self.plotdata[1])
        
class NumericalA():
    """
    Calculates A through generating samples from the likelihood function in parameter space with a fixed dataset.
    Please input a dataset generated from the Generate class.
    """
    def __init__(self,dataset, npoints):
        self.dim = dataset.dim
        self.bigM = dataset.npoints
        self.npoints = npoints
        self.xydata = dataset.data
        self.xdata = [a for a,b in self.xydata]
        self.ydata = [b for a,b in self.xydata]
        self.xdatamean = sum(self.xdata) / self.bigM
        self.ydatamean = sum(self.ydata) / self.bigM

        self.call()
        
    def sums(self):
        """A module to define all the sums needed in the likelihood module"""
        i = 0
        xsqr = []
        ysqr = []
        xixj = []
        yiyj = []
        while i < self.bigM:
            xsqr.append(self.xdata[i] ** 2)
            ysqr.append(self.ydata[i] ** 2)
            j = 0
            while j < self.bigM:
                if i == j:
                    j += 1
                else:
                    xixj.append(self.xdata[i] * self.xdata[j])
                    yiyj.append(self.ydata[i] * self.ydata[j])
                    j += 1
            i += 1
        self.xsqrsum = sum(xsqr)
        self.ysqrsum = sum(ysqr)
        self.xixjsum = sum(xixj) / 2
        self.yiyjsum = sum(yiyj) / 2
        
    def likelihood(self):
        """This should two lists of generated points from the likelihood functions in parameter space with a fixed dataset"""
        self.mu1gauss = np.random.normal(self.xdatamean, 1/np.sqrt(self.bigM), self.npoints)
        self.mu2gauss = np.random.normal(self.ydatamean, 1/np.sqrt(self.bigM), self.npoints)
        self.mu1data = np.sqrt(2*np.pi)/self.bigM * 1/((2*np.pi)**(self.bigM/2)) * np.exp(1/(self.bigM ** 2) * self.xsqrsum - self.xixjsum) * self.mu1gauss
        self.mu2data = np.sqrt(2*np.pi)/self.bigM * 1/((2*np.pi)**(self.bigM/2)) * np.exp(1/(self.bigM ** 2) * self.ysqrsum - self.yiyjsum) * self.mu2gauss
        i = 0
        self.mudata = []
        while i < self.npoints:
            self.mudata.append([self.mu1data[i],self.mu2data[i]])
            i += 1
            
        mu1sorted = sorted(self.mu1data)
        mu2sorted = sorted(self.mu2data)
        plt.figure(2)
        self.likeplot = plt.scatter(self.mu1data, self.mu2data)
        plt.axis([mu1sorted[0], mu1sorted[-1], mu2sorted[0], mu2sorted[-1]])
        plt.title("Generated Set from Likelihood Function in Parameter Space from Fixed Dataset")
        plt.xlabel("mu1")
        plt.ylabel("mu2")
        plt.show()


    def NNdistance(self):
        self.NNdistlist = []
        i = 0
        for point1 in self.mudata:
            j = 0
            distances = []
            for point2 in self.mudata:
                if i == j:
                    j += 1
                else:
                    distances.append(distance.euclidean(point1,point2))
                    j += 1
            distances.sort()
            self.NNdistlist.append(distances[0]) #Can actually use this to set which nearest neighbour distance you want to use. distances[n-1]
            i += 1
    
    def likefunc1D(self,x,npoints,xsqrsum,xixjsum,xmean):
        """The likelihood functions for each dimension should be separable, thus only 1D is required"""
        return 1/((2*np.pi) ** (npoints / 2)) * np.exp(1/(npoints ** 2) * (xsqrsum - xixjsum)) * np.exp(-1 * npoints/2 * ((x - xmean) ** 2))
        
    def Probability(self):
        """ p(x) = integral(dmu1 dmu2 L(mu1, mu2) * uniform prior)"""
        #self.testfunc = lambda x: x**2
        #self.testintegral = integrate.quad(self.testfunc,1,2)
        #print("testfunc", self.testintegral)
        self.mu1likefunc = lambda x: self.likefunc1D(x,self.bigM,self.xsqrsum,self.xixjsum,self.xdatamean)
        self.mu1probability = integrate.quad(self.mu1likefunc,-np.inf,np.inf)
        self.mu2likefunc = lambda y: self.likefunc1D(y,self.bigM,self.ysqrsum,self.yiyjsum,self.ydatamean)
        self.mu2probability = integrate.quad(self.mu2likefunc,-np.inf,np.inf)
        print("p(x) / prior =", self.mu1probability[0] * self.mu2probability[0])
            
    def call(self):
        self.sums()
        self.likelihood()
        #self.NNdistance()
        self.Probability()
        
"""Below are just some test values to play around with."""
c = Generate(2,200)
b = NumericalA(c,500)
