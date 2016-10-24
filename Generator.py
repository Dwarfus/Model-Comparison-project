import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import math

class call():
    def __init__(self):
        self.res100=np.array([0.0]*5)
        self.res250=np.array([0.0]*5)
        self.res500=np.array([0.0]*5)
        self.res750=np.array([0.0]*5)
        self.res1000=np.array([0.0]*5)
        i=0       
        for value in self.res100:
            a=DGenerator(2,100,0,1)
            self.res100[i]=a.mean
            i+=1
            
        i=0       
        for value in self.res250:
            a=DGenerator(2,250,0,1)
            self.res250[i]=a.mean
            i+=1
            
        i=0       
        for value in self.res500:
            a=DGenerator(2,500,0,1)
            self.res500[i]=a.mean
            i+=1
        i=0       
        for value in self.res750:
            a=DGenerator(2,750,0,1)
            self.res750[i]=a.mean
            i+=1
            
        i=0       
        for value in self.res1000:
            a=DGenerator(2,1000,0,1)
            self.res1000[i]=a.mean
            i+=1

 
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
        #self.plot()
        
        
    def generate(self):
        i=0
        while (i < self.npoints):
            j=0
            while (j < self.dim):                
                self.data[i,j]=np.random.normal(self.mean, self.variance)
                j+=1       
            i+=1
       # print(self.data)
        
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
       # print self.minim
        
    def probability(self):
        i=0
        self.prob=np.array([0.0]*self.npoints)
        for point in self.data:
            #print type(-1*((float(point[0])-self.mean)**2+(float(point[1])-self.mean)**2))#/(2*self.variance**2))
            self.prob[i]=1/(2*math.pi*self.variance**2)*math.exp(-1*((float(point[0])-self.mean)**2+(float(point[1])-self.mean)**2)/(2*self.variance**2))
            i+=1
        #print "probability is"
        #print self.prob
        
    def estimation(self):
        i=0
        self.constant=np.array([0.0]*self.npoints)
        for prob in self.prob:
            self.constant[i]=math.gamma(self.dim/2+1)/(float(prob)*self.minim[i]**(self.dim)*(math.pi)**(self.dim/2)) #"""The 1 should be replaced by the constant of proportionality!!! which we dont know yet"""
            i+=1
        #print self.constant
        self.mean=np.mean(self.constant)
        print self.mean
    
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
        plt.hist(self.minim, bins=np.arange(min(self.minim), max(self.minim) + 0.1, 0.1))
        plt.title("Nearest neighbour")
        plt.xlabel("distance")
        plt.ylabel("number of occurances")
        plt.show()
        
        plt.figure(2)
        plt.scatter(x,y)
        plt.title("Scatter of points")
        plt.xlabel("xcoordinate")
        plt.ylabel("ycoordinate")
        plt.grid()
        plt.show()
        
        plt.figure(3)       
        plt.hist(self.constant, bins=np.arange(min(self.constant), max(self.constant) + 50, 50))
        plt.title("Constant")
        plt.xlabel("value")
        plt.ylabel("number of occurances")
        plt.show()

