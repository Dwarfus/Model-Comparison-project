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
THIS WORKS! NEVER ALTER THIS ONE. Works within a reliable accuracy for variable L
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
  

    def analytic(self, r, n):
        """
        Analytic function obtained through Poisson method"""
        lamda = np.pi**(self.dim/2)*r**(self.dim)/math.gamma(self.dim/2+1)*n
        #dkpdf= (lamda**(self.kth-1)*np.exp(-lamda)*2*np.pi*r*n/math.factorial(self.kth-1))
        dkpdf = (np.exp((self.kth-1)*np.log(lamda)-lamda+np.log(2*np.pi*r*n)-gammaln(self.kth)))
        return dkpdf

       
        
    def analyticfit(self):
        """
        Fits the analytic function with a gaussian to get the parameters
        """
        self.maxbin=max(self.histogram[1])
        self.minbin=min(self.histogram[1])
        self.steps=np.linspace(self.minbin, self.maxbin, 100)
        
        self.gaussanalyt,pcov2 =curve_fit(Main.gauss,self.steps,self.analytic(self.steps, self.maindensity),p0=[1,np.mean(self.steps),1])
        plt.plot(self.steps,Main.gauss(self.steps,*self.gaussanalyt),'ro:',label='fit')
        plt.show()
        
 
        
    def scale(self):
        """
        This method does the most maths. It produces the P(D|ni) for each local density from the original analytic function.
        This is possible as the original function is a function of n. It was checked to give the same results as the method od scaling one general gaussians parameters.
        
        The next step is to change into P(n|D) which is done by rescaling the x axis to find the corresponding n for D. 
        
        Next the x-axis is divided by likelihood so we get P(A|D) however likelihood is different for each point.
        THis means that the values of probabilities no longer correspond to each other. We need to find the parameteres of the gaussian. That is the next part.
        Here it is extremely important to carefully select the initial guesses for the fitting parameters. However with the use of the P(D|n0) parameters we can do it easily.
        
        With the parameters we can evaluate them at some universal points(same for all gaussians) and do the sum of logs. Getting a peaked value
        """

        # steps2 are for evaluation of pdf based on local density
        self.steps2=np.linspace(self.minbin/3,self.gaussanalyt[1]*3,400)
        self.probabilities=np.zeros((len(self.density), len(self.steps2)))
        j=0
        plt.figure(2)
        for value in self.density:
            self.probabilities[j]=self.analytic(self.steps2, value)
            plt.plot(self.steps2, self.probabilities[j])
            
            plt.title("Pdf of P(D|n)")
            j+=1
        plt.show()    
        
        
        
        # Now I need to change into P(n|D)
        self.newsteps=self.kth/(math.pi*(self.steps2)**2)
        
        # Now for each point the x-axis is divided by the likelihood. The probabilities no longer correspond to the same points
        # So we need to find the parameters of the gaussians to then evaluate all of them at the same points
        self.parameters=np.zeros((self.npoints,3))
        m=0

        
        guess = self.kth/(math.pi*self.gaussanalyt[1]**2) # estimate the mean of the densities of the gaussians to use as initial guess after dividing by likelihood
        totalguess=0
        for value in self.like:

            newsteps = self.newsteps/value
            totalguess=totalguess+guess/value    
            self.para, pcov2 =curve_fit(Main.gauss,newsteps,self.probabilities[m],p0=[self.gaussanalyt[0],guess/value,guess/(value*4)])
            self.parameters[m]=self.para            
            m+=1
        totalguess=totalguess/m # Total guess is an average of the means of all the gaussians of P(A|D) serves as a mid point for the fitting so no important values are omitted.
 
        # FRom parameters we evaluate the gaussians and add the log
        self.steps3 = np.linspace(1,totalguess*4,200) # these steps are used for plotting for logsums

        self.logsum=np.zeros(( len(self.steps3)))
        
        j=0
        plt.figure(4)
        plt.title("P(A|Di)")
        self.probabilities2=np.zeros((len(self.density), len(self.steps3)))
        for value2 in self.parameters:
            
            self.probabilities2[j]=Main.gauss(self.steps3,*value2)
            plt.plot(self.steps3, self.probabilities2[j])
                        
            j+=1
        plt.show()
        
        # Here we add the logsums
        k=0
        for prob2 in self.probabilities2:
            prob2[prob2<=0]=1 # for any forbiden value (zero or negative probability) we assign 1 which contributes nothing to the log sum. 
            
            self.logsum=np.add(self.logsum,np.log(abs(prob2)))
            
            k+=1

        plt.figure(11)
        plt.title("Peak")
        plt.plot(self.steps3,np.exp(self.logsum))
        plt.show()
        self.max_x = self.steps3[self.logsum.argmax()]
        self.max_y = max(self.logsum)
        print("The estimate of A and the 'probability' value at that point is",self.max_x, self.max_y)
        
           
      
        

    def call(self):

        self.flat()
        self.distance()
        self.density()
        self.plot()
        self.fit()
        self.analyticfit()
        self.scale()

       
       
        """Comment> The flattening of the D distribution might be due to boundary effects. which makes the distance larger than it should actually be"""

        
a=Main(2,650,[1.0,1.0],20)            

