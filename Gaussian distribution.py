#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 10:55:38 2016
@author: donukb and Pavel

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
from scipy.special import binom, gammaln
from scipy.optimize import curve_fit, minimize_scalar
from scipy import special



class Generate():
    """
    Generates data from a 2D Gaussian distribution with a user-defined mean and standard deviation
    """
    def __init__(self, dimension, npoints, mean = [0.0,0.0], variance = [1.0,1.0]):
        self.dim = dimension
        self.npoints = npoints        
        self.data = np.array([[0.0]*self.dim]*self.npoints)
        self.means = mean
        self.variances = variance        
        self.gaussian()
        self.testplot()
        
    def gaussian(self):
        j=0
        while (j < self.npoints):
            k=0
            while (k < self.dim):                
                self.data[j,k]=np.random.normal(self.means[k], self.variances[k]) # THe variance is really standard deviation but as it is equal to one it does not matter
                k+=1       
            j+=1   
    
        
    def testplot(self):
        self.plotdata = [[a for a,b in self.data],[b for a,b in self.data]]
        plt.figure(1)
        self.testplot = plt.scatter(self.plotdata[0],self.plotdata[1])
        plt.show()
        print("data generated")
        
class NumericalA():
    """
    Calculates A through generating samples from the likelihood function in parameter space with a fixed dataset.
    Please input a dataset generated from the Generate class.
    """
    def __init__(self,dataset, npoints, kth):
        self.kth = kth
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
        """A module to define all the sums needed in the likelihood module and the analytic posterior evaluation"""
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
        """This generates the points in parameter space for a given xmean and ymean and M number of points in data space"""
        self.mu1gauss = np.random.normal(self.xdatamean, 1/np.sqrt(self.bigM), self.npoints)
        self.mu2gauss = np.random.normal(self.ydatamean, 1/np.sqrt(self.bigM), self.npoints)
        i = 0
        self.mudata = []
        while i < self.npoints:
            self.mudata.append([self.mu1gauss[i],self.mu2gauss[i]])
            i += 1
            
        plt.figure(2)
        self.likeplot = plt.scatter(self.mu1gauss, self.mu2gauss)
        plt.axis([-3/np.sqrt(self.bigM), 3/np.sqrt(self.bigM), -3/np.sqrt(self.bigM), 3/np.sqrt(self.bigM)])
        plt.title("Generated Set from Likelihood Function in Parameter Space from Fixed Dataset")
        plt.xlabel("mu1")
        plt.ylabel("mu2")
        plt.show()

    
    def likefunc1D(x,npoints,xsqrsum,xixjsum,xmean):
        """
        The likelihood functions for each dimension should be separable, thus only 1D is required
        """
        return np.exp(1/(npoints ** 2) * (xsqrsum - xixjsum)) * np.exp(-1 * npoints/2 * ((x - xmean) ** 2))
        # took out factor of 1/(2 * np.pi) ** (npoints/2) * 
        

    def gauss( x,a,x0,sigma):
        """Prescription for gauss distribution"""
        return a*math.e**(-(x-x0)**2/(2*sigma**2))
        
    def Probability(self):
        """ p(x) = integral(dmu1 dmu2 L(mu1, mu2) * uniform prior) This evaluate analyticaly the posterior*prior"""
        self.mu1likefunc = lambda x: NumericalA.likefunc1D(x,self.bigM,self.xsqrsum,self.xixjsum,self.xdatamean)
        self.mu1probability = integrate.quad(self.mu1likefunc,-np.inf,np.inf)
        self.mu2likefunc = lambda y: NumericalA.likefunc1D(y,self.bigM,self.ysqrsum,self.yiyjsum,self.ydatamean)
        self.mu2probability = integrate.quad(self.mu2likefunc,-np.inf,np.inf)
        print("p({x}) / prior =", self.mu1probability[0] * self.mu2probability[0])
        
    def NNdistance(self):
        """Calculates the kth nearest neighbour distance for each point in parameter space"""
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
            self.NNdistlist.append(distances[self.kth -1])
            i += 1
            
    def findradius(self):
        """Find an appropriate radius for the total volume"""
        centre = [self.xdatamean,self.ydatamean]
        i = 0
        for point1 in self.mudata:
            distancesfromcentre = []
            distancesfromcentre.append(distance.euclidean(centre,point1))
            i += 1
        distancesfromcentre.sort()
        self.radius = distancesfromcentre[-1]

    def maindensity(self):
        """Finds the overall density of the parameter space data"""
        self.volume = math.pi**(self.dim/2)*(self.radius)**(self.dim)/math.gamma(self.dim/2+1)    # the factor around variance is the number of sigma
        self.maindensity=self.npoints/self.volume
        print("Total Parameter Space Density =", self.maindensity)

    def localdensity(self):
        """Calculates the local density of each point using the nearest neighbour distances"""
        self.densitylist = []
        i = 0
        while i < self.npoints:
            volume = self.NNdistlist[i] ** self.dim * (np.pi ** (self.dim / 2)) / special.gamma(self.dim/2 + 1)
            self.densitylist.append((self.kth) / volume)
            i += 1
            
    def locallikelihood(self):
        """Calculates the local likelihood of each point"""
        self.locallike = []
        for point in self.mudata:
            self.locallike.append(NumericalA.likefunc1D(point[0],self.npoints,self.xsqrsum,self.xixjsum,self.xdatamean) * NumericalA.likefunc1D(point[1],self.npoints,self.ysqrsum,self.yiyjsum,self.ydatamean))
           
 
    def plot(self):
        """
        This method plots the histogram of the kth nearest neighbout distances
        """
        plt.figure(3)   
        self.histogram = plt.hist(self.NNdistlist,bins="auto", normed=True ) 
        
        plt.title("Nearest Neighbour #%s" %(self.kth))
        plt.xlabel("Distance")
        plt.ylabel("Probability") 
        
        
    def fit(self):
        """This methods fits the histogram with gaussian curve and finds the parameters of the curve. 
        Should be similar to the analytic function below"""               
        j=0
        self.bincentres=np.array([0.0]*(len(self.histogram[1])-1))        
        while j<(len(self.histogram[1])-1):        
            self.bincentres[j]=(self.histogram[1][j]+self.histogram[1][j+1])/2            
            j+=1
        
        self.gaussdata,pcov = curve_fit(NumericalA.gauss,self.bincentres,self.histogram[0],p0=[1,np.mean(self.bincentres),1])
        plt.plot(self.bincentres,NumericalA.gauss(self.bincentres,*self.gaussdata),'ro:',label='fit')
        
    def analytic(self, r, n):
        """
        Analytic function obtained through Poisson method. 
        Gives the p(D|n)"""
        lamda = np.pi**(self.dim/2)*r**(self.dim)/math.gamma(self.dim/2+1)*n        
        dkpdf = (np.exp((self.kth-1)*np.log(lamda)-lamda+np.log(2*np.pi*r*n)-gammaln(self.kth)))
        return dkpdf

       
        
    def analyticfit(self):
        """
        Fits the analytic function with a gaussian to get the parameters for the overall density.
        This is used as a guess for picking the steps in next section.
        Steps= the values of D on which to evaluate the function.
        """
        self.maxbin=max(self.histogram[1])
        self.minbin=min(self.histogram[1])
        self.steps=np.linspace(self.minbin/5, self.maxbin, 500)
        intmax= np.argmax(self.histogram[0])      
        probs = self.analytic(self.steps, self.maindensity)
        self.gaussanalyt,pcov2 =curve_fit(NumericalA.gauss,self.steps,self.analytic(self.steps, self.maindensity),p0=[self.histogram[0][intmax]/3 ,self.histogram[1][intmax],self.histogram[1][intmax]/5])
        plt.plot(self.steps,NumericalA.gauss(self.steps,*self.gaussanalyt),'ro:',label='fit')
        plt.plot(self.steps, probs )
        plt.show()



 
        
    def scale(self):
        """
        This method does the most maths. It produces the P(D|ni) for each local densitylist from the original analytic function.
        This is possible as the original function is a function of n. It was checked to give the same results as the method using the pdf from the paper.
        
        The next step is to change into P(n|D) which is done by rescaling the x axis to find the corresponding n for D. 
        
        Next the x-axis is divided by likelihood so we get P(A|D) however likelihood is different for each point.
        This means that the values of probabilities no longer correspond to each other. We need to find the parameteres of the gaussian. That is the next part.
        Here it is extremely important to carefully select the initial guesses for the fitting parameters. However with the use of the P(D|n0) parameters we can do it easily.
        
        With the parameters we can evaluate them at some universal points(same for all gaussians) and do the sum of logs. Getting a peaked value
        
        """
        
       
        self.steps2=np.linspace(self.minbin/6,self.gaussanalyt[1]*40,20000) # steps2 are D values
        self.probabilities=np.zeros((len(self.densitylist), len(self.steps2)))
        j=0
        plt.figure(5)
        for value in self.densitylist:
            self.probabilities[j]=self.analytic(self.steps2, value)
            plt.plot(self.steps2, self.probabilities[j])            
            plt.title("Pdf of P(D|n)")
            j+=1
        plt.show()    
        
        count = 0 # This was a check for negative values as they posed a problem during one stage of the development
        for prob in self.probabilities:
            for number in prob:
                if number<0:
                    count+=1
        #print(count)
        
        # Now I need to change into P(n|D)
        self.newsteps=self.kth/(math.pi*(self.steps2)**2) # newsteps are the n values 
       # print(self.newsteps)
        plt.figure(10)
        for prob in self.probabilities:
            plt.plot(self.newsteps, prob)
        plt.title("P(n)")
        plt.show()
        
        # Now for each point the x-axis is divided by the likelihood. The probabilities no longer correspond to the same points
        # So we need to find the parameters of the gaussians to then evaluate all of them at the same points
        self.parameters=np.zeros((self.npoints,3))
        m=0                
                
        guess = self.maindensity        
        totalguess=0
    
        for value in self.locallike:
            newstepsl = self.newsteps/value
            inti = self.probabilities[m].argmax()
            const = self.probabilities[m][inti]
            mean = newstepsl[inti]

            self.para, pcov2 =curve_fit(NumericalA.gauss,newstepsl,self.probabilities[m],p0=[const,mean,mean/4])
            totalguess=totalguess+mean                
            self.parameters[m]=self.para              
            m+=1
            
        totalguess=totalguess/m # Total guess is an average of the means of all the gaussians of P(A) serves as a mid point for the fitting so no important values are omitted.


        # From parameters we evaluate the gaussians and add the log of them             
        self.steps3 = np.linspace(1,totalguess*40,5000) # these steps are used for evaluating A
        self.logsum=np.zeros(( len(self.steps3)))
        
        j=0
        plt.figure(4)
        plt.title("P(A|Di)")

        self.probabilities2=np.zeros((len(self.densitylist), len(self.steps3)))
        for value2 in self.parameters:

            self.probabilities2[j]=NumericalA.gauss(self.steps3,*value2)
            plt.plot(self.steps3, self.probabilities2[j])
                        
            j+=1
        plt.show()
        
        # Here we add the logsums
        k=0
        for prob2 in self.probabilities2:
            prob2[prob2<=0]=1 # for any forbiden value (zero or negative probability) we assign 1 which contributes nothing to the log sum. 
            # The above was done as for a while negative values were found. However this is no longer the case with better initial guesses for the fits
            self.logsum=np.add(self.logsum,np.log(abs(prob2)))            
            k+=1

        plt.figure(11)
        plt.title("Peak")
        plt.plot(self.steps3,np.exp(self.logsum))
        plt.show()
        self.max_x = self.steps3[self.logsum.argmax()]
        self.max_y = max(self.logsum)
        print("The estimate of A and the 'probability' value at that point is",self.max_x, self.max_y)
        self.parameterA()


        
    def parameterA(self):
        """
        Another way of getting the final estimate A. 
        Using the fact that the peak is just mean of the overall Gaussian,
        which in turn is just a product of all the Gaussian distributions.
        There exist an analytical expression for this and is used below."""
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
        self.paramA = sum(scaledmeans) * totalvar 
        self.parA=self.npoints/(self.mu1probability[0] * self.mu2probability[0])
        print("expected value of A is:", self.parA)
        
    def call(self):
        self.sums()
        self.likelihood()
        self.Probability()
        self.NNdistance()
        print("distance done")
        self.findradius()
        self.maindensity()
        self.plot()
        self.localdensity()
        self.locallikelihood()
        self.analyticfit()
        print("analyticfit done")
        self.scale()
        

c = Generate(2,500)
b = NumericalA(c,500,5)
