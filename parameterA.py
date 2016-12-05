#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 17:05:59 2016

@author: donukb
"""

def parameterA(self):
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
        i += 0
    totalvar = 1/ sum(invvars)
    self.parameterA = sum(scaledmeans) * totalvar