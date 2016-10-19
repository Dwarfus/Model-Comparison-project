import numpy as np
import matplotlib.pyplot as plt


class DGenerator():
    def __init__(self, dimension, npoints, mean, variance):
        self.dim = dimension
        self.npoints = npoints
        self.mean = mean
        self.variance = variance        
        self.results = np.array([[0.0]*self.dim]*self.npoints)
        self.generate()
        self.plot()
        
    def generate(self):
        i=0
        while (i < self.npoints):
            j=0
            while (j < self.dim):
                
                self.results[i,j]=np.random.normal(self.mean, self.variance)
                j+=1       
            i+=1
        print self.results
        
    def plot(self):
        k=0
        x=[]
        print(self.npoints)
        while(k<self.npoints):
            x.append(self.results[k,0])
            k+=1
            
        print(x)
        plt.hist(x)
        plt.title("Guassian")
        plt.xlabel("Don")
        plt.ylabel("Pavel")
        plt.show()

        
