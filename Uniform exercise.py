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
from scipy.special import binom
from scipy.optimize import curve_fit, minimize_scalar



class Main:
    def __init__(self, dim, npoints, mean=[0.0,0.0], kth=1 ):
        """
        Initiates the necessary values as given by the user
        """
        self.data=np.array([[0.0]*dim]*npoints)
        self.dim=dim
        self.npoints=npoints
        self.mean=mean
        self.kth=kth
        self.like=np.array([0.0]*self.npoints) 
        # Here the likelihood of each point will be stored
        
        self.call()
        

        
    def flat(self):
        """
        This method generate data according to a uniform distribution
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
        This method calculates the kth nearest neighbour distance. 
        The results are saved in 1D array
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
            self.density[i] =self.kth*math.gamma(self.dim/2+1) \
                            /(math.pi**(self.dim/2)*self.distance[i] \
                              **self.dim)
            i+=1  
            
            
    def plot(self):
        """
        This method plots the histogram of the kth nearest neighbout distance.
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
        """This methods fits the histogram with gaussian curve and finds the parameters of the curve. 
        This values is a check that the analytic function is simillar to this one"""               
        j=0
        self.bincentres=np.array([0.0]*(len(self.histogram[1])-1))        
        while j<(len(self.histogram[1])-1):        
            self.bincentres[j]=(self.histogram[1][j]+self.histogram[1][j+1])/2            
            j+=1
        
        self.gaussdata,pcov = curve_fit(Main.gauss,self.bincentres,self.histogram[0],p0=[1,np.mean(self.bincentres),1])
        plt.plot(self.bincentres,Main.gauss(self.bincentres,*self.gaussdata),'ro:',label='fit')


    def analytic(self, r, n):
        """
        Analytic function obtained through Poisson method for p(D|n)"""
        lamda = np.pi**(self.dim/2)*r**(self.dim)/math.gamma(self.dim/2+1)*n
        dkpdf= (lamda**(self.kth-1)*np.exp(-lamda)*2*np.pi*r*n/math.factorial(self.kth-1))
        return dkpdf

    def analytic2(self,r):
        """Analytic function as from the paper. Gives same answer as the above"""
        self.V0=np.pi**(self.dim/2)/(math.gamma(self.dim/2+1))
        self.binomial = (binom(self.maindensity-1, self.kth-1))*(self.maindensity-self.kth)
        anpdf=self.binomial*self.V0**(self.kth)*(1-self.V0*r**self.dim)**(self.maindensity-self.kth-1)*self.dim*r**(self.dim*self.kth-1)
        return anpdf        
        
    def analyticfit(self):
        """
        Fits the analytic function with a gaussian to get the parameters
        Used for initial estimates and x-axis interval choice.
        """
        self.maxbin=max(self.histogram[1])
        self.minbin=min(self.histogram[1])
        self.steps=np.linspace(self.minbin, self.maxbin, 100)
        
        self.gaussanalyt,pcov2 =curve_fit(Main.gauss,self.steps,self.analytic(self.steps, self.maindensity),p0=[1,np.mean(self.steps),1])
        plt.plot(self.steps,Main.gauss(self.steps,*self.gaussanalyt),'ro:',label='fit')

 

    def aestimator(self):
        """
        This is an alternative method to evaluate A. For each point the most probable n is evaluated from its D.
        From n, A was calculated. These values of A were plotted as a histogram.
        The histogram has a Gaussian shape, which mean was taken as a A estimate. Works well for uniform likelihood.
        """
        self.a=self.density/self.like
        
        plt.figure(2)   
        self.histogram2 = plt.hist(self.a,bins="fd", normed=True )       
        plt.title("A estimation")
        plt.xlabel("A")
        plt.ylabel("probability")
        self.average=np.mean(self.a)        

        # finding a good starting estimate for the mean
        i=0
        maxhist = max(self.histogram2[0])
        for hbin in self.histogram2[0]:
            if( hbin== maxhist):
                maxvalue=i
                None
            i+=1
            
                        
        j=0
        self.bincentres2=np.array([0.0]*(len(self.histogram2[1])-1))        
        while j<(len(self.histogram2[1])-1):            
            self.bincentres2[j]=(self.histogram2[1][j]+self.histogram2[1][j+1])/2           
            j+=1

        self.agauss,pcov3 = curve_fit(Main.gauss,self.bincentres2,self.histogram2[0],p0=[maxhist,self.bincentres2[maxvalue],self.bincentres2[maxvalue]/2])
        plt.plot(self.bincentres2,Main.gauss(self.bincentres2,*self.agauss),'ro:',label='fit')
     
        plt.show()
        mean = np.mean(self.a)
        print("The value of A using estimation is(peak, average)",self.agauss[1], mean)
        
    def scale(self):
        """
        This method takes p(D|n) scales it for every point and its local density ni to p(D|ni)
        Then it takes this prob and by recalculating D into n evaluates p(n).
        The n values are then divided by L to give p(A)
        The gaussians are multiplied (sum of logs). 
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
            

        self.steps2=np.linspace(0.03,self.gaussanalyt[1]*4,500) # D values
        self.probabilities=np.zeros((len(self.density), len(self.steps2)))
        j=0
        plt.figure(3)
        plt.title("p(D|n)")
        plt.xlabel("n")
        plt.ylabel("Relative Probability")
        for gauss in self.localgauss:
            self.probabilities[j]=Main.gauss(self.steps2,*gauss)
            plt.plot(self.steps2, self.probabilities[j])                        
            j+=1

        self.newsteps=self.kth/(math.pi*(self.steps2)**2) # n values
        
        self.logsum=np.zeros(( len(self.steps2)))

        k=0        
        for prob in self.probabilities: 
            self.logsum=np.add(self.logsum,np.log(prob))
            k+=1
       

        self.newsteps=self.newsteps/self.like[1] #A values

        plt.figure(4)
        plt.title("p(A|D)")
        plt.xlabel("A value")
        plt.ylabel("Relative Probability")
        for prob in self.probabilities:
            plt.plot(self.newsteps, prob)
            
        plt.show()

        plt.figure(5)
        plt.title("Peak")
        plt.xlabel("A value")
        plt.ylabel("Relative Probability")        
        plt.plot(self.newsteps,np.exp(self.logsum))
        plt.show()
        
        self.max_x = self.newsteps[self.logsum.argmax()]
        self.max_y = max(self.logsum)
        print("The estimate of A",self.max_x)

        
    def call(self):
        self.flat()
        self.distance()
        self.density()
        self.plot()
        self.fit()
        self.analyticfit()
        self.aestimator()
        self.scale()
       
                    
a=Main(2,500,[1.0,1.0],10)       
