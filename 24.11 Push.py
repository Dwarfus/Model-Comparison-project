# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 12:09:21 2016

@author: Pavel
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

class Main:
    def __init__(self, dim, npoints, mean=[0.0,0.0], kth=1 ):
        self.data=np.array([[0.0]*dim]*npoints)
        self.dim=dim
        self.npoints=npoints
        self.mean=mean
        self.kth=kth
        self.like=np.array([0.0]*self.npoints) # Here the likelihood of each point will be stored
        
        self.call()
        
   
    def gaussian(self):
        # Simple loops, for each point, it generates value for each dimension.
        i=0
        variance=1
        while (i < self.npoints):
          
                            
            self.data[i] = np.random.normal(self.mean, variance)
            self.like[i] = 1/(2*np.pi)*np.exp(-(self.data[i,0]-self.mean[0])**2/2*variance**2)*np.exp(-(self.data[i,1]-self.mean[1])**2/2*variance**2)
            i+=1
        self.volume = math.pi**(self.dim/2)*(variance*3)**(self.dim)/math.gamma(self.dim/2+1)    # the factor around variance is the number of sigma
        self.maindensity=self.npoints/self.volume    
        print("main density is", self.maindensity)
        #print("the probabilities are", self.like)
        
    def flat(self):
        # Simple loops, for each point, it generates value for each dimension
        i=0
        self.x=3
        self.volume=self.x**(self.dim)
        while(i<self.npoints):
            j=0
            while (j<self.dim):
                self.data[i,j]=np.random.uniform(0,self.x)
                j+=1
            self.like[i]=1/self.volume
            i+=1 
        self.maindensity=self.npoints/self.volume
        print("main density is",self.maindensity)
       # print("the probabilities are", self.like)
    
    def distance(self):
        # this calculat the distance to every other point of every point and for each point find the minimu, create an array of these minimas
        i=0
        self.distance=np.array([0.0]*self.npoints)
        for point in self.data:
            j=0
            dist=[]
            for other in self.data:
                if (i==j):
                    None
                else:
                   dist.append(distance.euclidean(point, other)) 
                j+=1
         
            a = np.array(dist)
            self.distance[i] = np.partition(a, self.kth-1)[self.kth-1]
            i+=1
            
    def density(self):
        """Here the local density around each point will be calculated"""
        self.density=np.array([0.0]*self.npoints)
        i=0
        while i<len(self.distance):
            self.density[i] = self.kth*math.gamma(self.dim/2+1)/(math.pi**(self.dim/2)*self.distance[i]**self.dim)
            i+=1
       # print(self.density)    
            
            
    def plot(self):
        plt.figure(1)   
        self.histogram = plt.hist(self.distance,bins="fd", normed=True )       
        plt.title("Nearest neighbour")
        plt.xlabel("distance")
        plt.ylabel("probability")
        
        

    def gauss( x,a,x0,sigma):
        return a*math.e**(-(x-x0)**2/(2*sigma**2))
        
    def fit(self):
               
        j=0
        self.bincentres=np.array([0.0]*(len(self.histogram[1])-1))
        
        while j<(len(self.histogram[1])-1):
            
            self.bincentres[j]=(self.histogram[1][j]+self.histogram[1][j+1])/2
            
            j+=1
        
        self.gaussdata,pcov = curve_fit(Main.gauss,self.bincentres,self.histogram[0],p0=[1,np.mean(self.bincentres),1])
        plt.plot(self.bincentres,Main.gauss(self.bincentres,*self.gaussdata),'ro:',label='fit')
        print("the fitted variables are: normalisation constant, mean, standard dev")
        print(self.gaussdata)        
        

    def analytic(self, r, n):
        lamda = np.pi**(self.dim/2)*r**(self.dim)/math.gamma(self.dim/2+1)*n
        dkpdf= (lamda**(self.kth-1)*np.exp(-lamda)*2*np.pi*r*n/math.factorial(self.kth-1))
        return dkpdf
        
    def analyticfit(self):
        self.maxbin=max(self.histogram[1])
        self.minbin=min(self.histogram[1])
        self.steps=np.linspace(self.minbin, self.maxbin, 100)
        
        self.gaussanalyt,pcov2 =curve_fit(Main.gauss,self.steps,self.analytic(self.steps, self.maindensity),p0=[1,np.mean(self.steps),1])
        plt.plot(self.steps,Main.gauss(self.steps,*self.gaussanalyt),'ro:',label='fit')
        print("the fitted variables of analytic function are: normalisation constant, mean, standard dev")
        print(self.gaussanalyt)
        plt.show()

        
    def call(self):
        #self.gaussian()
        self.flat()
        self.distance()
        self.density()
        self.plot()
        self.fit()
        self.analyticfit()


        
a=Main(2,250,[1.0,1.0],1)            
            