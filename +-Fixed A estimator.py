 # -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 12:09:21 2016

@author: Pavel Kroupa, Don Ma
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
from scipy.special import binom, gammaln
from scipy.optimize import minimize_scalar
"""
THIS WORKS! NEVER ALTER THIS ONE. wORKS FOR 2D BUT VERY NICELY FOR LARGE KTH
"""


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
        """
        The generator of data with gaussian distribution of points
        """
        i=0
        variance=1
        while (i < self.npoints):
          
                            
            self.data[i] = np.random.normal(self.mean, variance)
            self.like[i] = 1/(2*np.pi)*np.exp(-(self.data[i,0]-self.mean[0])**2/2*variance**2)*np.exp(-(self.data[i,1]-self.mean[1])**2/2*variance**2)
            i+=1
        self.volume = math.pi**(self.dim/2)*(variance*2)**(self.dim)/math.gamma(self.dim/2+1)    # the factor around variance is the number of sigma
        self.maindensity=self.npoints/self.volume    
        print("main density is", self.maindensity)

        
    def flat(self):
        """
        This method generate data according to a flat distribution
        """
        i=0
        self.x=4
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

    
    def distance(self):
        """
        This method calculates the kth nearest neighbour distance. The results are saved in 1D array
        """
        i=0
        self.distance=np.array([0.0]*self.npoints)
        for point in self.data:
            j=0
            dist=[]
            for other in self.data:
                if (i==j):
                    j+=1
                else:
                   dist.append(distance.euclidean(point, other)) 
                j+=1
         
            a = np.array(dist)
            self.distance[i] = np.partition(a, self.kth-1)[self.kth-1]
            i+=1
            
    def density(self):
        """Here the local density around each point is calculated"""
        self.density=np.array([0.0]*self.npoints)
        i=0
        while i<len(self.distance):
            self.density[i] = self.kth*math.gamma(self.dim/2+1)/(math.pi**(self.dim/2)*self.distance[i]**self.dim)
            i+=1  
            
            
    def plot(self):
        """
        This method plots the histogram of the kth nearest neighbout distances
        """
        plt.figure(1)   
        self.histogram = plt.hist(self.distance,bins="auto", normed=True )       
        plt.title("Nearest neighbour")
        plt.xlabel("distance")
        plt.ylabel("probability")
        
        

    def gauss( x,a,x0,sigma):
        """Prescription for gauss distribution"""
        return a*math.e**(-(x-x0)**2/(2*sigma**2))
        
    def fit(self):
        """This methods fits the histogram with gaussian curve and finds the parameters of the curve."""               
        j=0
        self.bincentres=np.array([0.0]*(len(self.histogram[1])-1))        
        while j<(len(self.histogram[1])-1):        
            self.bincentres[j]=(self.histogram[1][j]+self.histogram[1][j+1])/2            
            j+=1
        
        self.gaussdata,pcov = curve_fit(Main.gauss,self.bincentres,self.histogram[0],p0=[1,np.mean(self.bincentres),1])
        plt.plot(self.bincentres,Main.gauss(self.bincentres,*self.gaussdata),'ro:',label='fit')
       # print("the fitted variables are: normalisation constant, mean, standard dev")
       # print(self.gaussdata)        
        

    def analytic(self, r, n):
        """
        Analytic function obtained through Poisson method"""
        lamda = np.pi**(self.dim/2)*r**(self.dim)/math.gamma(self.dim/2+1)*n
        #dkpdf= (lamda**(self.kth-1)*np.exp(-lamda)*2*np.pi*r*n/math.factorial(self.kth-1))
        dkpdf = (np.exp((self.kth-1)*np.log(lamda)-lamda+np.log(2*np.pi*r*n)-gammaln(self.kth)))
        return dkpdf

    def analytic2(self,r):
        """Analytic function as from the paper"""
        self.V0=np.pi**(self.dim/2)/(math.gamma(self.dim/2+1))
        self.binomial = (binom(self.maindensity-1, self.kth-1))*(self.maindensity-self.kth)
        anpdf=self.binomial*self.V0**(self.kth)*(1-self.V0*r**self.dim)**(self.maindensity-self.kth-1)*self.dim*r**(self.dim*self.kth-1)
        
        return anpdf        
        
    def analyticfit(self):
        """
        Fits the analytic function with a gaussian to get the parameters
        """
        self.maxbin=max(self.histogram[1])
        self.minbin=min(self.histogram[1])
        self.steps=np.linspace(self.minbin, self.maxbin, 100)
        
        self.gaussanalyt,pcov2 =curve_fit(Main.gauss,self.steps,self.analytic(self.steps, self.maindensity),p0=[1,np.mean(self.steps),1])
        plt.plot(self.steps,Main.gauss(self.steps,*self.gaussanalyt),'ro:',label='fit')
        #print("the fitted variables of analytic function are: normalisation constant, mean, standard dev")
        #print(self.gaussanalyt)
               
       # plt.show()
        
 
        
    def scale(self):
        """
        This method takes p(D|n) scales it for every point and its local density ni to p(D|ni)
        Then it takes this prob and using Bayes theorem scales it into p(n|Di).
        PDF for each point is then evaluated on range and the gaussians are multiplied (sum of logs). 
        This leads to a peak which represents the A estimate
        """
        self.localgauss=np.zeros((len(self.density),3))
        i=0
        for value in self.density:
            scale = (self.maindensity/value)**(1/self.dim)
            self.localgauss[i][0]= self.gaussanalyt[0]/scale
            self.localgauss[i][1]=self.gaussanalyt[1]*scale
            self.localgauss[i][2] = self.gaussanalyt[2]*scale
            
               
            i+=1
        #print(self.localgauss) 
        self.steps2=np.linspace(0.1,self.gaussanalyt[1]*3,400)
        self.probabilities=np.zeros((len(self.density), len(self.steps2)))
        j=0
        plt.figure(3)
        for gauss in self.localgauss:
            self.probabilities[j]=Main.gauss(self.steps2,*gauss)
            plt.plot(self.steps2, self.probabilities[j])
            
            plt.title("scale fit")
            j+=1
        plt.show()    
        self.newsteps=self.kth/(math.pi*(self.steps2)**2)
        
        ##########################################################
        self.parameters=np.zeros((self.npoints,3))
        m=0
        plt.figure(10)
        plt.title("scale fit scaled by L")
        #plt.plot(self.newsteps, self.probabilities[0])
        #plt.show()
        #print(len(self.probabilities[0]), len(self.newsteps))
        #print(self.probabilities[0])
        #print(self.like)
        for value in self.like:

            newsteps = self.newsteps/value
            #print
 
            self.para, pcov2 =curve_fit(Main.gauss,newsteps,self.probabilities[m],p0=[10,1500,400])
            plt.plot(newsteps, self.probabilities[m])
            #print("hi",self.para)
            self.parameters[m]=self.para
            #print("hi",self.parameters[m],m)
            
            
            m+=1
        #print(self.parameters)
        plt.show() 
        
        self.steps3 = np.linspace(1,5000,200)
        
        
        self.logsum2=np.zeros(( len(self.steps3)))
        plt.show()
       # print(self.logsum.shape)
        
        j=0
        self.probabilities2=np.zeros((len(self.density), len(self.steps3)))
        for value2 in self.parameters:
            #print("hi",value)
            self.probabilities2[j]=Main.gauss(self.steps3,*value2)
            plt.plot(self.steps3, self.probabilities2[j])
            
            plt.title("scale fit2222")
            j+=1
        plt.show()
        
       # print(self.probabilities2[1])
        #self.probabilities2
        k=0
        for prob2 in self.probabilities2:
            #print("hi",prob)
            prob2[prob2<=0]=1
            
           # print(self.logsum2[50], prob2[50])
            self.logsum2=np.add(self.logsum2,np.log(abs(prob2)))
            
            k+=1
       
        #print(self.logsum2)


        plt.figure(11)
        plt.title("Peak 222222")
        plt.plot(self.steps3,np.exp(self.logsum2))
        plt.show()
        self.max_x2 = self.steps3[self.logsum2.argmax()]
        self.max_y2 = max(self.logsum2)
        print("The estimate of A of gauss",self.max_x2, self.max_y2)
        
        
        
        
        
        
        
       
       ########################################################################  
       
       
       
        self.logsum=np.zeros(( len(self.steps2)))
        plt.show()
       # print(self.logsum.shape)
        k=0
        
        for prob in self.probabilities:
            #print(self.logsum[80], prob[80])
            self.logsum=np.add(self.logsum,np.log(prob))
            
            k+=1
       
        #print(self.logsum)
        self.newsteps=self.newsteps/self.like[1]

        plt.figure(5)
        plt.title("Peak")
        plt.plot(self.newsteps,np.exp(self.logsum))
        plt.show()
        self.max_x = self.newsteps[self.logsum.argmax()]
        self.max_y = max(self.logsum)
        print("The estimate of A",self.max_x)

        
        #other, self.nnewsteps=np.split(self.newsteps,[(self.logsum.argmax())*0.5])
       # other2, self.nlogsum=np.split(self.logsum,[(self.logsum.argmax())*0.5])

       # plt.figure(6)
       # plt.title("Peak2")
       # plt.plot(self.nnewsteps,np.exp(self.nlogsum))
        #plt.show()
        
    def parameterA(self):
        means = []
        sigmas = []
        for parameter in self.parameters:
            means.append(parameter[1])
            sigmas.append(parameter[2])
        invvars = []
        scaledmeans = []
        i = 0
        for sigma in sigmas:
            invvars.append(1 / sigma ** 2)
            scaledmeans.append(means[i]/ sigma ** 2)
            i += 1
        totalvar = 1/ sum(invvars)
        self.parameterA = sum(scaledmeans) * totalvar
        print(self.parameterA, totalvar)
        

    def call(self):
        #self.gaussian()
        self.flat()
        self.distance()
        self.density()
        self.plot()
        self.fit()
        self.analyticfit()
        #self.aestimator()
        self.scale()
        self.parameterA()
       
       
        """Comment> The flattening of the D distribution might be due to boundary effects. which makes the distance larger than it should actually be"""

        
a=Main(2,1500,[1.0,1.0],25)            
