# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 21:11:00 2016

@author: Pavel
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import math
from scipy import integrate
from scipy.special import binom, gamma, gammaln

class DataSpace():
    """
    This method generates points in data space and calculates the various sums 
    and means needed in the rest of the code: x**2, xixj, y**2, yiyj and ymean, xmean.
    """
    def __init__(self, mpoints, mean = [0.0,0.0], variance = [1.0,1.0]):
        self.mpoints = mpoints
        self.data = np.zeros((self.mpoints,2))
        
        self.mean = mean
        self.variance = variance
        
        self.gendata()
        self.testplot()
        self.sums()
    
    def gendata(self):
        
 
        j=0
        while (j < self.mpoints):
            
            self.data[j,0]=np.random.normal(self.mean[0], self.variance[0]) # THe variance is really standard deviation but as it is equal to one it does not matter
            self.data[j,1]=np.random.normal(self.mean[1], self.variance[1])       
            j+=1 
    
    def testplot(self):
        self.Tdata = self.data.T
        self.xdata = self.Tdata[0]
        self.ydata = self.Tdata[1]
        plt.figure(1)
        self.testplot = plt.scatter(self.xdata, self.ydata)
        plt.show()
        print("data generated")
            
    def sums(self):
        self.xmean = self.xdata.mean()
        self.ymean = self.ydata.mean()
        self.xsqr = np.sum(self.xdata**2)
        self.ysqr = np.sum(self.ydata**2)
        xixj = []
        yiyj = []
        i=0
        while i < self.mpoints:
            
            xixj.append(np.sum(self.xdata[i] * self.xdata))
            yiyj.append(np.sum(self.ydata[i] * self.ydata))
            i += 1
        
        self.xixj = (np.sum(xixj) - self.xsqr)/2
        self.yiyj = (np.sum(yiyj) - self.ysqr)/2        
        #print(self.xixj, self.yiyj)
        
class Parameterspace():
    """
    This class generates points in parameter space, calculates the distances
    to all other points and save it as huge array. Also calculates the likelihood
    of each point in parameter space.
    """
    def __init__(self, npoints, xmean, ymean, xsqr, ysqr, xixj, yiyj, mpoints):
        self.npoints = npoints
        self.xmean = xmean
        self.ymean = ymean
        self.xsqr = xsqr
        self.ysqr = ysqr
        self.xixj = xixj
        self.yiyj = yiyj
        self.mpoints = mpoints
        self.mudata=np.zeros((self.npoints,2))
        
        self.pargen()
        self.NNdistance()
        self.likecalc()
        
    def pargen(self):
                
        j=0
        while (j < self.npoints):
            
            self.mudata[j,0]=np.random.normal(self.xmean, 1/np.sqrt(self.mpoints)) # THe variance is really standard deviation but as it is equal to one it does not matter
            self.mudata[j,1]=np.random.normal(self.ymean, 1/np.sqrt(self.mpoints))       
            j+=1
        print('Parameter space data generated')
        
    def NNdistance(self):
       self.NNdistances = np.zeros((self.npoints, self.npoints-1))
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
            #self.kth = 2
            self.NNdistances[i]=distances
            
            i += 1
            
    def likecalc(self):
       self.pardata = self.mudata.T     
       self.like = self.likelihood(self.pardata[0], self.pardata[1])   
       #print(self.like)
       
    def likelihood(self, x,y):
    
        """
        The likelihood functions for each dimension should be separable,
        """
        probx = np.exp(1/(self.npoints ** 2) * (self.xsqr - self.xixj)) * np.exp(-1 * self.npoints/2 * ((x - self.xmean) ** 2))
        proby = np.exp(1/(self.npoints ** 2) * (self.ysqr - self.yiyj)) * np.exp(-1 * self.npoints/2 * ((y - self.ymean) ** 2))
        return probx*proby
        # took out factor of 1/(2 * np.pi) ** (npoints/2) * 
        

class Analytical():
    def __init__(self, npoints, xmean, ymean, xsqr, ysqr, xixj, yiyj, mpoints):
        self.npoints = npoints
        self.xmean = xmean
        self.ymean = ymean
        self.xsqr = xsqr
        self.ysqr = ysqr
        self.xixj = xixj
        self.yiyj = yiyj
        self.mpoints = mpoints
        self.probability()
        
    
    
    def likefunc1D(self,x,npoints,xsqrsum,xixjsum,xmean):
        """
        The likelihood functions for each dimension should be separable, thus only 1D is required
        """
        return np.exp(1/(npoints ** 2) * (xsqrsum - xixjsum)) * np.exp(-1 * npoints/2 * ((x - xmean) ** 2))
        
    def probability(self):
        """ p(x) = integral(dmu1 dmu2 L(mu1, mu2) * uniform prior)"""
        #self.testfunc = lambda x: x**2
        #self.testintegral = integrate.quad(self.testfunc,1,2)
        #print("testfunc", self.testintegral)
        self.mu1likefunc = lambda x: self.likefunc1D(x,self.mpoints,self.xsqr,self.xixj,self.xmean)
        self.mu1probability = integrate.quad(self.mu1likefunc,-np.inf,np.inf)
        self.mu2likefunc = lambda y: self.likefunc1D(y,self.mpoints,self.ysqr,self.yiyj,self.ymean)
        self.mu2probability = integrate.quad(self.mu2likefunc,-np.inf,np.inf)
        self.posterior = self.mu1probability[0] * self.mu2probability[0] # it is really posterior/prior and ignored the 1/2pi to m factor
        print("p({x}) / prior =", self.posterior)


class Numerical():
    def __init__(self, kth, distances, likelihood):
        self.kth = kth # nearest neighbour is k=1
        self.distances=distances
        self.likelihood = likelihood
        self.kdist = np.zeros((len(self.likelihood)))
        i=0
        for point in self.distances:
            self.kdist[i] = point[self.kth-1]
            i+=1

        self.density = self.kth/(math.pi*self.kdist**2)
        
        self.aestimator()
        
    def poisson(self, distance, likelihood, avalue):
        #return ((self.kth-1)*np.log(np.pi*distance**2*avalue*likelihood)-(np.pi*distance**2*avalue*likelihood)+(2*np.pi*distance*avalue*likelihood**2)-gammaln(self.kth))
        #below is an alternative without the log, e.g. the original probability but with the last bit put into the first bracket
        return ((np.pi*distance**2*avalue*likelihood)**(self.kth))*np.exp(-np.pi*distance**2*avalue*likelihood)*2*likelihood/(gamma(self.kth))

    def aestimator(self):
        self.aguess = self.density/self.likelihood
        self.max = max(self.aguess)
        self.min = min(self.aguess)
       
        #print(self.max, self.min)
        self.asteps = np.linspace(self.min/100, self.max,15000)
        
        self.prob = np.zeros((len(self.kdist), len(self.asteps)))
        self.logsum = np.zeros((1,len(self.asteps)))
        #print(self.logsum.shape)
        i=0
        plt.figure(1)
        plt.title("P(A|D,L)")
        for distance in self.kdist:
            probab = self.poisson(distance, self.likelihood[0], self.asteps)
            self.prob[i] = probab
            self.log = np.log(probab)
            self.logsum = np.add(self.logsum, self.log)
           # print(self.logsum.shape)
            plt.plot(self.asteps,self.prob[i])        
            i+=1
       # plt.xlim([self.min/100, 0.1e8])
        plt.show()
        
        plt.figure(2)
        plt.title("peak")
        plt.plot(self.asteps, self.logsum[0])
        plt.show()
        
        self.max_a = self.asteps[self.logsum.argmax()]
        self.max_p = max(self.logsum)
        print(self.max_a)
        # posterior guess calculation
        self.posteriorguess = len(self.likelihood)/self.max_a
        print(self.posteriorguess, c.posterior)
        
       
        
        
        
a= DataSpace(5000, [0,0], [1,1])
b = Parameterspace(5000,a.xmean, a.ymean, a.xsqr, a.ysqr, a.xixj, a.yiyj, a.mpoints)
c = Analytical(b.npoints,a.xmean, a.ymean, a.xsqr, a.ysqr, a.xixj, a.yiyj, a.mpoints)
d = Numerical(5, b.NNdistances, b.like)
