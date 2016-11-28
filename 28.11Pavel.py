# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 11:44:17 2016

@author: Pavel
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 11:26:43 2016

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

class Main():
    """
    This code is the 2D flat distribution excercise to try evaluating the number density.
    As it is a flat distribution the n is known so we can check that our approach work.
    Then this method will be aplied on the other data in more D where we will try to evaluate A"""
    
    def __init__(self, npoints, x,y, nth):
        self.npoints = npoints
        self.x = x
        self.y = y
        self.data = np.array([[0.0]*2]*self.npoints)
        self.nth= nth # the nth nearest neighbour. For the nearest one write 1
        self.call() # this is a calling method
        
        

    def call(self):
        """
        This method is responsible for the calling of the other methods
        """
        self.generate()
        self.distance()
        self.plotdistance()
        self.dplot()

    def generate(self):
        i=0
        self.volume=self.x*self.y
        while(i<self.npoints):            
            self.data[i,0]=np.random.uniform(0,self.x)
            self.data[i,1]=np.random.uniform(0,self.y)
            i+=1      
            
            
    def distance(self):
        # this calculat the distance to every other point of every point and for each point find the minimu, create an array of these minimas
        i=0
        self.distance=np.array([0.0]*self.npoints)
        self.distance2=np.array([0.0]*self.npoints)
        for point in self.data:
            j=0
            dist=[]
            for other in self.data:
                if (i==j):
                    None
                else:
                   dist.append(distance.euclidean(point, other)) 
                j+=1
            #self.distance2[i]=min(dist)
            a = np.array(dist)
            self.distance[i] = np.partition(a, self.nth-1)[self.nth-1]
            i+=1
            
    def plotdistance(self):
        plt.figure(1)   
        self.pdf = plt.hist(self.distance,bins="fd", normed=True )
        plt.title("Nearest neighbour")
        plt.xlabel("distance")
        plt.ylabel("probability")
        plt.show()
        self.bincentres=np.array([0.0]*len(self.pdf[0]))
        
        
    def dprob(self, rn):
        
        self.Vo = (math.pi)
        self.binomial = (binom(self.npoints-1, self.nth-1))*(self.npoints-self.nth)
        self.dpdf = self.binomial*self.Vo**(self.nth)*(1-self.Vo*rn**2)**(self.npoints-self.nth-1)*2*rn**(2*self.nth-1)
        return self.dpdf

    def gauss(x,a,x0,sigma):
        return a*math.e**(-(x-x0)**2/(2*sigma**2))        
        
    def dplot(self):
        prob=[]
        self.step=self.pdf[1]
        i=0
        while(i<len(self.pdf[1])-1):
           self.bincentres[i]=((self.step[i]+self.step[i+1])/2)
           i+=1
        self.dsteps = np.arange(0,max(self.step),max(self.step)/100)
        for value in self.dsteps:
            prob.append(self.dprob(value))
            
        self.popt,pcov = curve_fit(Main.gauss,self.bincentres,self.pdf[0],p0=[1,np.mean(self.bincentres),1])

        self.dnnprob=Main.gauss(self.bincentres,*self.popt)    

        self.dprob=np.array(prob)     #   def distpdf(self):
            
        plt.figure(1)   
        plt.plot(self.dsteps,np.array(self.dprob))
        self.pdf = plt.hist(self.distance,bins="fd", normed=True )
        plt.plot(self.bincentres,self.dnnprob,'ro:',label='fit')
        plt.title("Nearest neighbour")
        plt.xlabel("distance")
        plt.ylabel("probability")
        plt.show()
        self.multipl=self.dnnprob*self.bincentres
        self.density=self.nth/self.Vo/(self.bincentres**2)
        
        plt.figure(2)   
        #plt.plot(self.dsteps,np.array(self.dprob))
        #self.pdf = plt.hist(self.distance,bins="fd", normed=True )
        plt.plot(self.density,self.dnnprob,'ro:',label='fit')
        plt.title("Nearest neighbour")
        plt.xlabel("density")
        plt.ylabel("probability")
        plt.show()
       
        
class Call():
    """
    This will call the script many times and collect the pdf of the densities for different nth 
    """        
    def __init__(self, npoints, x,y, nths=np.array([1,2,3,4,5,6,7,8,9,10])):
        self.totaldnnp=np.array([[0.0]*3]*len(nths))
        self.npoi=npoints
        self.nths=nths
        i=0
        for value in nths:
            print("test1")
            a=Main(npoints,x,y,value)
            self.totaldnnp[i]=a.popt
            
            i+=1
        
        #for each in self.totaldnnp    
        #print(self.totaldnnp)
        self.final()
        
    def gauss(self,x,a,x0,sigma):
        return a*math.e**(-(x-x0)**2/(2*sigma**2))
        
        
    def final(self):
        self.fin=[]
       # print("bincentres are", a.bincentres)
        steps=np.linspace(0.05, 0.60, 100)
        #print(steps)
        maxi=0
        for value in steps:
            self.npdf=1
            
            for triplet in self.totaldnnp:
                cons, mean, sigma=triplet
                self.npdf=self.npdf*self.gauss(value,*triplet)
            
            if maxi<self.npdf:
                maxi=self.npdf
                maxvalue=value            
            self.fin.append(self.npdf)

        print(maxvalue)
        self.dens=1/math.pi/(steps**2)*np.mean(self.nths)
        maxdens=1/math.pi/(maxvalue**2)*np.mean(self.nths)
        print(maxdens)
       # print (self.fin)
        """  
        i=0
        for value in total:
            if len(value)<maxlength:
                value.append(0.0)
                print(value)
            else:
                self.totaldnnp[i]=value
                i+=1
                
            
        print (self.totaldnnp)
        #self.delta = self.totaldnnp[0]*self.totaldnnp[1]*self.totaldnnp[2]*self.totaldnnp[3]*self.totaldnnp[4]*self.totaldnnp[5]*self.totaldnnp[6]
        """
        
        plt.figure(3)   
        plt.plot(steps,self.fin,'ro:',label='fit')
        plt.title("Nearest neighbour")
        plt.xlabel("nearest neighbour")
        plt.ylabel("probability")
        plt.show() 
        
b=Call(300,5,5,[3,4,5,6,7,8,9,10,11,12,13])

