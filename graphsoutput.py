# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 12:51:54 2016

@author: Pavel
"""
import numpy as np
import matplotlib.pyplot as plt

class Graph():
    def __init__(self, path):
        self.data =  np.genfromtxt(path, dtype=float, comments="#")
        #print (self.data)
        self.sepdata()
        
    def sepdata(self):
        i=0
        length=int(len(self.data))
        print(length, type(length))
        self.const=np.array([0.0]*int(length))
        self.factor=np.array([0.0]*int(length))
        while(i<length):
            self.factor[i]=self.data[i,0]
            self.const[i]=self.data[i,1]
            i+=1
        #print (self.factor, self.const)   
        self.mean=np.mean(self.const)
        print(self.mean)
        plt.figure(1)         
        self.pdf = plt.hist(self.const, bins="auto" )
        plt.title("Constant of proportionality")
        plt.xlabel("constant of prop")
        plt.ylabel("number of occurances")
        plt.show()            
            