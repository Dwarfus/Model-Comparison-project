import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

class DGenerator():
    def __init__(self, dimension, npoints, mean, variance):
        self.dim = dimension
        self.npoints = npoints
        self.mean = mean
        self.variance = variance        
        self.results = np.array([[0.0]*self.dim]*self.npoints)
        self.generate()
        self.distance()
        self.plot()
        
        
    def generate(self):
        i=0
        while (i < self.npoints):
            j=0
            while (j < self.dim):                
                self.results[i,j]=np.random.normal(self.mean, self.variance)
                j+=1       
            i+=1
        print(self.results)
        
    def distance(self):
        i=0
        self.minimums=np.array([0.0]*self.npoints)
        for point in self.results:
            j=0
            dist=[]
            for other in self.results:
                if (i==j):
                    None
                else:
                   dist.append(distance.euclidean(point, other)) 
                j+=1
            self.minimums[i]=min(dist)
            
            i+=1
        print self.minimums
    def plot(self):
        k=0
        x=[]
        
        while(k<self.npoints):
            x.append(self.results[k,0])
            k+=1
        k=0
        y=[]
        while(k<self.npoints):
            y.append(self.results[k,1])
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
        
