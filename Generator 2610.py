import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import math
from scipy import integrate

class call():
    """
    This class is for mass processing of the data. It is very poorly written as it was needed quickly. Need to redo!
    If needed to call, set n to the number of runs for each number of data. Put the class DGenerator into comment and 
    rename the __init__ onto init. Or ask Pavel, he has written this ugly piece of code. 
    
    It will produce n runs for 100,250,500,750,1000 points generated and print the mean nearest neighbour distance for each run.
    """
    def __init__(self):
        n=50  
        self.res100=np.array([0.0]*n)
        self.res250=np.array([0.0]*n)
        self.res500=np.array([0.0]*n)
        self.res750=np.array([0.0]*n)
        self.res1000=np.array([0.0]*n)
        i=0  
           
        while i<n:
            print (i)
            a=self.init(2,100,0,1)
            self.res100[i]=self.meanA
            i+=1
            
        print ("100 done")
        i=0       
        while i<n:
            a=self.init(2,250,0,1)
            self.res250[i]=self.meanA
            i+=1
            
        print ("250 done")
        i=0       
        while i<n:
            a=self.init(2,500,0,1)
            self.res500[i]=self.meanA
            i+=1
        print ("500 done")
        i=0       
        while i<n:
            a=self.init(2,750,0,1)
            self.res750[i]=self.meanA
            i+=1
            
        print ("750 done")
        i=0       
        while i<n:
            a=self.init(2,1000,0,1)
            self.res1000[i]=self.meanA
            i+=1
        print ("100: \n")
        print (self.res100)
        print ("250: \n")
        print (self.res250)
        print ("500: \n")
        print (self.res500)
        print ("750: \n")
        print (self.res750)
        print ("1000: \n")
        print (self.res1000)
         
 
class DGenerator():
    
    """
    This class is the main part of the code. It is split into small functions each responsible for one small step.
    
    """
    def __init__(self, dimension, npoints, mean, variance):
        self.dim = dimension
        self.npoints = npoints
        self.mean = mean
        self.variance = variance        
        self.data = np.array([[0.0]*self.dim]*self.npoints) # this is the array in which the data will be generated into
        
        #self.gaussian() #pick which distribution you want to use, either gaussian or flat
        self.flat()
        self.distance()
        self.probability()
        self.estimation()
        #self.integration()
        #self.bincount()
        self.plot()
        self.integration()
        
        
    def gaussian(self):
        # Simple loops, for each point, it generates value for each dimension.
        i=0
        while (i < self.npoints):
            j=0
            while (j < self.dim):                
                self.data[i,j]=np.random.normal(self.mean, self.variance)
                j+=1       
            i+=1
       # print(self.data)
        
    def flat(  self):
        # Simple loops, for each point, it generates value for each dimension.        
        i=0
        while(i<self.npoints):
            j=0
            while (j<self.dim):
                self.data[i,j]=np.random.uniform(-3,3)
                j+=1
            i+=1    
    
    def distance(self):
        # this calculat the distance to every other point of every point and for each point find the minimu, create an array of these minimas
        i=0
        self.distance=np.array([0.0]*self.npoints)
        for point in self.data:
            j=0
            dist=[]
            for other in self.data:
                if (i==j):
                    None
                else:
                   dist.append(distance.euclidean(point, other)) 
                j+=1
            self.distance[i]=min(dist)
            
            i+=1
        print (self.distance)
        
    def probability(self):
        # This calculates for each point the probability of the point being generated. Creates an array where self.prob[i] is
        # the probability of point self.data[i] with nearest neighbour distance self.distance[i]
        i=0
        self.prob=np.array([0.0]*self.npoints)
        for point in self.data:
            self.prob[i]=1/(2*math.pi*self.variance**2)*math.exp(-1*((float(point[0])-self.mean)**2+(float(point[1])-self.mean)**2)/(2*self.variance**2))
            i+=1
        #print "probability is"
        #print self.prob
        
    def estimation(self):
        """
        This is our older method of estimating the constant A. self.meanA is the average value of A for all points.
        the self.meanD is the average of the mean neighbour distance
        """
        i=0
        self.constant=np.array([0.0]*self.npoints)
        for prob in self.prob:
            self.constant[i]=math.gamma(self.dim/2+1)/(float(prob)*self.distance[i]**(self.dim)*(math.pi)**(self.dim/2)) #"""The 1 should be replaced by the constant of proportionality!!! which we dont know yet"""
            i+=1
        self.meanA=np.mean(self.constant)
        self.meanD=np.mean(self.distance)
        #print self.meanA, self.meanD
    

    def bincount(self):
        """
        This method is responsible for evaluating the constant for the probability density of nearest neighbour distance.
        It splits the values of distances into bins of width (range of values)/ number of points. 
        It then calculates number of occurances in each bin. From this we can normalize the pdf as the integral must be one over whole range.
        From that we can calculate the actual constant of proportionality of D and number density, THIS STILL NEEDS TO BE DONE!!!
        right now it calculates a value that is approx constant with number of points but is not what we are looking for, I think....
        """
        self.iter=self.npoints
        self.count=np.array([0.0]*self.iter)
        a=np.linspace(0, max(self.distance), self.iter)
        i=0
        while i<self.iter-1:
            self.count[i]=((float(a[i]) < self.distance) & (self.distance <= float(a[i+1]))).sum() 
            i+=1

        self.normconst=math.fsum((self.count*a[1]))
        self.totconst=0
        i=0
        while i<self.iter-1:
            self.help =(self.count[i]/self.normconst*a[i]**2*a[1])
            self.totconst = self.totconst+self.help
            i+=1
        print (self.totconst)    
        """FINISH THIS PART. NEED THE WHOLE CONSTANT"""
       # print self.total
        
    def f1(self):
        self.bin+=1
        return self.pdf2[self.bin]
       
    def integration(self):
        self.bin =-1
        self.pdf2=self.pdf[0]
        self.step=self.pdf[1]
        i=0
        self.intsteps=np.array([0.0]*len(self.pdf2))        
        while(i<len(self.pdf[1])-1):
           self.intsteps[i]=((self.step[i]+self.step[i+1])/2)
           i+=1
          
        y1=self.pdf2
        self.normpdf= integrate.simps(y1,self.intsteps)
        print(self.normpdf)
        y2=y1/self.normpdf*(1/np.square(self.intsteps))
        print("test")
        print(y2)
        self.normpdf2= integrate.simps(y2,self.intsteps)
        print(self.normpdf2)
       # print(self.step, self.intsteps)
        
              
        
        
    def plot(self):
        """
        This just plots everything. Will probably not be very useful later so no need to keep it absolutely perfect"""
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
        self.pdf = plt.hist(self.distance,bins=self.npoints, normed=True )
        plt.title("Nearest neighbour")
        plt.xlabel("distance")
        plt.ylabel("probability??")
        plt.show()
        
        
        plt.figure(2)
        plt.scatter(x,y)
        plt.title("Scatter of points")
        plt.xlabel("xcoordinate")
        plt.ylabel("ycoordinate")
        plt.grid()
        plt.show()
        
        plt.figure(3)       
        plt.hist(self.constant, bins=self.npoints)
        plt.title("Constant")
        plt.xlabel("value")
        plt.ylabel("number of occurances")
        plt.show()

