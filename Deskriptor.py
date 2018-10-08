# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 10:36:40 2018

@author: Arbeiten
"""

import cv2
import numpy as np
import mahotas
import os
import matplotlib.pyplot as plt
from SVMs import MultiSVM, SVM
from Kernels import LinearKernel, PolyKernel, RBFKernel, MyKernel1, MyKernel2, MyKernel3
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from multiprocessing import Pool

class Deskriptor:
    #Deskriptoren aus https://github.com/Gogul09/image-classification-python
    
    #constants
    IMAGESIZE = tuple((100,100))
    BINS = 8
    
    def __init__(self, imgPath):
        img = cv2.imread(imgPath)
        self.img = cv2.resize(img, Deskriptor.IMAGESIZE)
        self.bwimg = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.vector = None
                
    
    def colhist(self):
        colhistogram = cv2.calcHist([cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)], [0,1,2], None, [Deskriptor.BINS, Deskriptor.BINS, Deskriptor.BINS], [0, 256, 0, 256, 0, 256])
        return cv2.normalize(colhistogram, colhistogram).flatten()
    
    def getVector(self):
        if self.vector is not None:
            return self.vector
        else:
            #Berechnet alle drei Deskriptor Vektoren und fügt sie zusammen
            #hu moments
            hmv = cv2.HuMoments(cv2.moments(self.bwimg)).flatten()
            #haralick
            hav = mahotas.features.haralick(self.bwimg).mean(axis=0)
            #colorhistogram
            colhistogram = cv2.calcHist([cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)], [0,1,2], None, [Deskriptor.BINS, Deskriptor.BINS, Deskriptor.BINS], [0, 256, 0, 256, 0, 256])
            chv = cv2.normalize(colhistogram, colhistogram).flatten()
            return np.hstack([ hmv, hav, chv])
        
        
    @staticmethod   
    def getXY(src="D:/Developement/Datasets/Leaves/", save=False, name="" , pcaComponents=16):
        # Deskriptoren aus https://github.com/Gogul09/image-classification-python
        X = []
        y = []    
        for nr, folder in enumerate(os.listdir(src)):
            print("[+] processing ", folder)
            path = src + folder
            for imgpath in os.listdir(path):
                d = Deskriptor(path+"/"+imgpath)
                v = d.getVector()
                X.append(v)
                y.append(nr)        
        X = np.array(X)
        y = np.array(y)
        
        std_clf = make_pipeline(StandardScaler(), PCA(n_components=pcaComponents))  #, GaussianNB()
        std_clf.fit(X, y)
        # Extract PCA from pipeline
        pca_std = std_clf.named_steps['pca']
        scaler = std_clf.named_steps['standardscaler']
        X_std = pca_std.transform(scaler.transform(X))
        
        if save:
            np.save(name+"_X", X_std)
            np.save(name+"_y", y)


    @staticmethod
    def getXYSingle(src="D:/Developement/Datasets/Leaves/", save=False, name="" , pcaComponents=16):
        # für jedes bild einzeln & für alle Zusammen
        X = []
        y = []    
        for nr, folder in enumerate(os.listdir(src)):
            nr+=1 #klasse startet bei 1 nicht 0
            print("[+] processing ", folder)
            path = src + folder
            for imgpath in os.listdir(path):
                d = Deskriptor(path+"/"+imgpath)
                v = d.getVector()
                X.append(v)
                y.append(nr)        
        X = np.array(X)
        y = np.array(y)
        std_clf = make_pipeline(StandardScaler(), PCA(n_components=16))  #, GaussianNB()
        std_clf.fit(X, y)
        pca_std = std_clf.named_steps['pca']
        scaler = std_clf.named_steps['standardscaler']
        X_std = pca_std.transform(scaler.transform(X))
        if save:
            np.save(name+"_X", X_std)
            np.save(name+"_y", y)
                    
                
        for clas,folder in enumerate(os.listdir(src)):
            clas +=1
            print("[+] processing single ", folder)
            path = src + folder
            for nr, imgpath in enumerate(os.listdir(path)):
                d = Deskriptor(path+"/"+imgpath)
                os.rename(path+"/"+imgpath, path+"/"+str(nr)+".jpg")
                v = d.getVector()
                v = np.array(v)
                v = v.reshape((1, v.shape[0]))
                v = pca_std.transform(scaler.transform(v))
                y = np.array([clas])
                np.save(path+"/"+str(nr)+"_X", v)
                np.save(path+"/"+str(nr)+"_y", y)


#-----------------------------------------------------------------------------------------------------


class Des2(Deskriptor):
    NCLUSTER = 8
    SURFPARAM = 400
    def __init__(self, imgPath, des, kmeansSurf, kmeansOrb, kmeansSift):
        img = cv2.imread(imgPath)
        self.img = cv2.resize(img, Deskriptor.IMAGESIZE)
        bwimg = img = cv2.imread(imgPath,0)
        self.bwimg = cv2.resize(bwimg, Deskriptor.IMAGESIZE)
        self.vector = None
        self.des = des
        self.kmeansSurf, self.kmeansOrb, self.kmeansSift = kmeansSurf, kmeansOrb, kmeansSift
        
    @staticmethod   
    def getXY(src="D:/Developement/Datasets/Leaves/", save=False, name="" , pcaComponents=16, des=False):
        # Deskriptoren aus https://github.com/Gogul09/image-classification-python
        kmeansSurf, kmeansOrb, kmeansSift = Des2.initLocalDescriptors(src)
        X = []
        y = []    
        for nr, folder in enumerate(os.listdir(src)):
            #print("[+] processing ", folder)
            path = src + folder
            for imgpath in os.listdir(path):
                d = Des2(path+"/"+imgpath, des, kmeansSurf, kmeansOrb, kmeansSift)
                v = d.getVector()
                X.append(v)
                y.append(nr)
        X = np.array(X)
        y = np.array(y)
        
        std_clf = make_pipeline(StandardScaler(), PCA(n_components=pcaComponents))  #, GaussianNB()
        std_clf.fit(X, y)
        # Extract PCA from pipeline
        pca_std = std_clf.named_steps['pca']
        scaler = std_clf.named_steps['standardscaler']
        X_std = pca_std.transform(scaler.transform(X))
        
        if save:
            np.save(name+"_X", X_std)
            np.save(name+"_y", y)
        
    @staticmethod
    def getXYSingle(src="D:/Developement/Datasets/Leaves/", save=False, name="" , pcaComponents=16, des=False):
        # für jedes bild einzeln & für alle Zusammen
        kmeansSurf, kmeansOrb, kmeansSift = Des2.initLocalDescriptors(src)
        X = []
        y = []    
        for nr, folder in enumerate(os.listdir(src)):
            nr+=1 #klasse startet bei 1 nicht 0
            print("[+] processing ", folder)
            path = src + folder
            for imgpath in os.listdir(path):
                d = Des2(path+"/"+imgpath, des, kmeansSurf, kmeansOrb, kmeansSift)
                v = d.getVector()
                X.append(v)
                y.append(nr)  
        X = np.array(X)
        y = np.array(y)
        std_clf = make_pipeline(StandardScaler(), PCA(n_components=16))  #, GaussianNB()
        std_clf.fit(X, y)
        pca_std = std_clf.named_steps['pca']
        scaler = std_clf.named_steps['standardscaler']
        X_std = pca_std.transform(scaler.transform(X))
        if save:
            np.save(name+"_X", X_std)
            np.save(name+"_y", y)
            
        for nr, folder in enumerate(os.listdir(src)):
            nr+=1 #klasse startet bei 1 nicht 0
            print("[+] processing ", folder)
            path = src + folder
            for imgpath in os.listdir(path):
                d = Des2(path+"/"+imgpath, des, kmeansSurf, kmeansOrb, kmeansSift)
                v = d.getVector()
                Xsave = pca_std.transform(scaler.transform(np.array([v])))
                ysave = np.array([nr])
                np.save(path+"/"+imgpath+"_X", Xsave)
                np.save(path+"/"+imgpath+"_y", ysave)
                    
     
    @staticmethod 
    def initLocalDescriptors(src):
        liSurf = []
        liOrb = []
        liSift = []
        for folder in os.listdir(src):
            path = src + folder
            for imgpath in os.listdir(path):
                img = cv2.imread(path + "/"+imgpath)
                img = cv2.resize(img, Deskriptor.IMAGESIZE)
                bwimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                try:
                    #surf
                    surf = cv2.xfeatures2d.SURF_create(400)
                    kp, desSURF = surf.detectAndCompute(bwimg,None)
                    liSurf.extend(desSURF)
                except:
                    pass
                try:
                    #orb detector
                    orb = cv2.ORB_create(edgeThreshold=25)
                    kpORB = orb.detect(bwimg,None)
                    kpORB, desOrb = orb.compute(img, kpORB)
                    liOrb.extend(desOrb)
                except:
                    pass
                
                try:
                    #sift
                    sift = cv2.xfeatures2d.SIFT_create()
                    kpSift, desSift = sift.detectAndCompute(bwimg,None)
                    liSift.extend(desSift)
                except: pass
            
        aSurf = np.array(liSurf)
        aOrb = np.array(liOrb)
        aSift = np.array(liSift)
        kmeansSurf = KMeans(n_clusters = Des2.NCLUSTER)
        kmeansSurf.fit(aSurf)
        kmeansOrb= KMeans(n_clusters = Des2.NCLUSTER)
        kmeansOrb.fit(aOrb)
        kmeansSift = KMeans(n_clusters = Des2.NCLUSTER)
        kmeansSift.fit(aSift)
        return (kmeansSurf, kmeansOrb, kmeansSift)
        
    

            
    def getVector(self):
        if self.vector is not None:
            return self.vector
        else:
            stack = []
            if self.des[0]:
                #Berechnet alle drei Deskriptor Vektoren und fügt sie zusammen
                #hu moments
                hmv = cv2.HuMoments(cv2.moments(self.bwimg)).flatten()
                stack.append(hmv)
            if self.des[1]:
                #haralick
                hav = mahotas.features.haralick(self.bwimg).mean(axis=0)
                stack.append(hav)
            
            if self.des[2]:
                try:
                    #surf
                    surf = cv2.xfeatures2d.SURF_create(Des2.SURFPARAM)
                    kp, desSURF = surf.detectAndCompute(self.bwimg,None)
                    predSurf = self.kmeansSurf.predict(desSURF)
                    vecSurf = []
                    for i in range(Des2.NCLUSTER):
                        vecSurf.append(np.where(predSurf==i, 1, 0).sum())
                    stack.append(vecSurf)
                except Exception as e:
                    vecSurf = []
                    for i in range(Des2.NCLUSTER):
                        vecSurf.append(0)
                    stack.append(vecSurf)
                    #print("Error getVector, Surf: ",e)
                    

            if self.des[3]:
                try:
                    #orb detector
                    orb = cv2.ORB_create()
                    kpORB = orb.detect(self.bwimg,None)
                    kpORB, desOrb = orb.compute(self.img, kpORB)
                    predOrb = self.kmeansOrb.predict(desOrb)
                    vecOrb = []
                    for i in range(Des2.NCLUSTER):
                        vecOrb.append(np.where(predOrb==i, 1, 0).sum())
                    stack.append(vecOrb)
                except Exception as e:
                    vecOrb = []
                    for i in range(Des2.NCLUSTER):
                        vecOrb.append(0)
                    stack.append(vecOrb)
                    #print("Error getVector, ORB: ",e)
                    
                
            if self.des[4]:
                try: 
                    #sift
                    sift = cv2.xfeatures2d.SIFT_create()
                    kpSift, desSift = sift.detectAndCompute(self.bwimg,None)
                    predSift = self.kmeansSift.predict(desSift)
                    vecSift = []
                    for i in range(Des2.NCLUSTER):
                        vecSift.append(np.where(predSift==i, 1, 0).sum())
                    stack.append(vecSift)
                except Exception as e:
                    vecSift = []
                    for i in range(Des2.NCLUSTER):
                        vecSift.append(0)
                    stack.append(vecSift)
                    #print("Error getVector, Sift: ",e)
                    
            if self.des[5]:
                colhistogram = cv2.calcHist([cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)], [0,1,2], None, [Deskriptor.BINS, Deskriptor.BINS, Deskriptor.BINS], [0, 256, 0, 256, 0, 256])
                chv = cv2.normalize(colhistogram, colhistogram).flatten()
                stack.append(chv)
                
            stacked = np.hstack(stack)
            return stacked
        

        
        
        
        
if __name__ == "__main__":
    if False:
        results = []
        for ORB in [True]:
            for n in [24]:
                for clusters in [90]:
                    try:
                        Des2.NCLUSTER = clusters
                        des = [True, True, True, ORB, True, True]
                        name = "test"
                        Des2.getXYSingle(src="D:/Developement/Datasets/Leaves_LocalDescriptors/", save=True, name=name , pcaComponents=n, des=des)
                        
                        #X = np.load(r"Data/Leaves/X_std_n16.npy")
                        #y = np.load(r"Data/Leaves/y_n16.npy")
                        
                        X = np.load(name+"_X.npy")
                        y = np.load(name+"_y.npy")
                        
                        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.30, random_state=1)
                        svm = MultiSVM()
                        svm.fit(X_train, y_train)
                    
                        score = svm.score(X_test, y_test)
                        results.append([score, ORB, n, clusters])
                        print(results[-1])
                    except Exception as e: 
                        print("Error at: ", [score, ORB, n, clusters], "\n" )
                        print(e)
        res = np.array(results)
        try:
            labels = ["Score", "ORB", "PCA-comps","clusters"]
            df = pd.DataFrame.from_records(res, columns=labels)
            df = df.sort_values(by=['Score'])
            print(df)
            df.to_csv(path_or_buf="df2.csv")
        except Exception as e:
            print(e)
            print(res)
            
    name = "LD_n24_c90_noCHist"
    X = np.load(name+"_X.npy")
    y = np.load(name+"_y.npy")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.30, random_state=19)
    svm = MultiSVM()
    svm.fit(X_train, y_train)
    
    np.save(name+"_X_train", X_train)
    np.save(name+"_X_test", X_test)
    np.save(name+"_y_train", y_train)
    np.save(name+"_y_test", y_test)
    svm.save("SVM_"+name)
   
    score = svm.score(X_test, y_test)
    print(score)
    
                  
#Data without PCA:
#[0.28097731239092494, True, 16, 4]
#[0.16230366492146597, True, 16, 8]
#[0.09947643979057597, True, 16, 12]
#[0.16230366492146597, True, 16, 16]
#[0.1169284467713787, True, 16, 32]
#[0.11169284467713791, True, 16, 48]
#[0.054101221640488695, True, 16, 64]              
