# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 11:35:01 2018

@author: Arbeiten
"""

import numpy as np
import scipy.stats as stats
import pylab as pl

#f = open("ResultsN26C110ts20.txt", "r")
#dfli = []
#for line in f.readlines():
#    li = line.split("{")
#    if len(li)!=4:
#        score = float(li[1])
#        dfli.append(score)

#avg = np.array(dfli).mean()




array = np.load("200Preds_best_pca24_cluster240.npy")
avg = array.mean()
dfli = array.tolist()
h = sorted(dfli)
minimum = h[0]
maximum = h[-1]
print(avg, minimum, maximum)
fit = stats.norm.pdf(h, np.mean(h), np.std(h))  #this is a fitting indeed

pl.plot(h,fit,'-')
pl.axvline(x=avg, color="black")
pl.xlabel("Korrektklasifizierungsrate")
pl.ylabel("Anteil der Messungen")

pl.hist(h,normed=True)      #use this to draw histogram of your data

pl.show()   