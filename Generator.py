import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import math

# a=DGenerator(2,10,0,10)
class DGenerator():
    def __init__(self, dimension, npoints, mean, variance):
        self.dim = dimension
        self.npoints = npoints
        self.mean = mean
        self.variance = variance        
        self.data = np.array([[0.0]*self.dim]*self.npoints)
        self.generate()
        self.distance()
        self.probability()
        self.estimation()
        self.plot()
        
        
    def generate(self):
        i=0
        while (i < self.npoints):
            j=0
            while (j < self.dim):                
                self.data[i,j]=np.random.normal(self.mean, self.variance)
                j+=1       
            i+=1
        print(self.data)
        
    def distance(self):
        i=0
        self.minim=np.array([0.0]*self.npoints)
        for point in self.data:
            j=0
            dist=[]
            for other in self.data:
                if (i==j):
                    None
                else:
                   dist.append(distance.euclidean(point, other)) 
                j+=1
            self.minim[i]=min(dist)
            
            i+=1
        print self.minim
        
    def probability(self):
        i=0
        self.prob=np.array([0.0]*self.npoints)
        for point in self.data:
            #print type(-1*((float(point[0])-self.mean)**2+(float(point[1])-self.mean)**2))#/(2*self.variance**2))
            self.prob[i]=1/(2*math.pi*self.variance**2)*math.exp(-1*((float(point[0])-self.mean)**2+(float(point[1])-self.mean)**2)/(2*self.variance**2))
            i+=1
        print "probability is"
        print self.prob
        
    def estimation(self):
        i=0
        self.constant=np.array([0.0]*self.npoints)
        for prob in self.prob:
            self.constant[i]=1/(float(prob)*self.minim[i]**(self.dim)) #"""The 1 should be replaced by the constant of proportionality!!! which we dont know yet"""
            i+=1
        print self.constant
    
        plt.figure(2)       
        plt.hist(self.constant,100)
        plt.title("Constant")
        plt.xlabel("value")
        plt.ylabel("number of occurances")
        plt.show()
    
    def plot(self):
        k=0
        x=[]
        
        while(k<self.npoints):
            x.append(self.data[k,0])
            k+=1
        k=0
        y=[]
        while(k<self.npoints):
            y.append(self.data[k,1])
            k+=1
        plt.figure(1)
        plt.subplot2grid((3,2),(0,0))    
       
        plt.hist(x)
        plt.title("Guassian")
        plt.xlabel("x coordinate")
        plt.ylabel("number of occurances")
        
        
        plt.subplot2grid((3,2),(0,1))
        plt.hist(y)
        plt.title("Guassian")
        plt.xlabel("x coordinate")
        plt.ylabel("number of occurances")
        
        plt.subplot2grid((3,2),(1,0), colspan=2, rowspan=2, aspect='equal')
        plt.scatter(x,y)
        plt.title("Scatter")
        plt.xlabel("xcoordinate")
        plt.ylabel("ycoordinate")
        plt.grid()
        plt.show()

