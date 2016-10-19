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
       # self.plot()
        
    def generate(self):
        i=0
        while (i < self.npoints):
            j=0
            while (j < self.dim):
                
                self.results[i,j]=np.random.normal(self.mean, self.variance)
                j+=1       
            i+=1
        print self.results
        
#    def plot(self):
 #       self.plot =
 #       plt.hist(self.results[])
  #      plt.title
  #      fig = plt.gcf()
        


print("hello world")