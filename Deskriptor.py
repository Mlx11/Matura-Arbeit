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
from Kernels import *
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from multiprocessing import Pool

class Deskriptor:
    # Deskriptor mit nur globalen Deskriptoren
    #Deskriptoren aus https://github.com/Gogul09/image-classification-python
    # In der Arbeit wurde Des2 verwendet
    
    #constants
    IMAGESIZE = tuple((100,100))
    BINS = 8
    
    def __init__(self, imgPath):
        img = cv2.imread(imgPath)
        self.img = cv2.resize(img, Deskriptor.IMAGESIZE)
        self.bwimg = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.vector = None
                
    
    def colhist(self):
        #colorhistogram ausgelagert
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

# Deskriptor mit globalen und lokalen Deskriptoren
class Des2(Deskriptor):
    NCLUSTER = 240
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
    def getXY(src="D:/Developement/Datasets/Leaves/", save=False, name="" , pcaComponents=24, des=False):
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
        return X_std, y
        
    @staticmethod
    def getXYSingle(src="D:/Developement/Datasets/Leaves/", save=False, name="" , pcaComponents=24, des=False):
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
        std_clf = make_pipeline(StandardScaler(), PCA(n_components=pcaComponents))  #, GaussianNB()
        std_clf.fit(X, y)
        pca_std = std_clf.named_steps['pca']
        scaler = std_clf.named_steps['standardscaler']
        X_std = pca_std.transform(scaler.transform(X))
        if save:
            np.save(name+"_X", X_std)
            np.save(name+"_y", y)
            
        if save:
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
        return X_std,y
                        
     
    @staticmethod 
    def initLocalDescriptors(src):
        # initialisiert die lokalen Deskriptoren und kmenas
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
            self.vector = stacked
            return stacked
        

        
#----------------------------------------------------------------------------------
            
def gridSearch():
        filename = "file.txt"
        PCAComps_li = [12,18, 24, 30, 36, 42]
        clusters_li = [90,120,150,180, 210, 240]
        kernel_li = [LinearKernel(), RBFKernel(), PolyKernel(p=2, c=1), LinearKernel()+RBFKernel(), PolyKernel(p=3, c=1)]
        c_li = [None]
        attemptsPerParameter = 1
        testSize = 0.3
        
        f = open(filename, "a")
        for n in PCAComps_li:
            for clusters in clusters_li:
                f.write("----------------------------------------")
                f.write("Data: PCA-Components = {" + str(n) + "{, clusters = {" + str(clusters) + "\n")
                print("Data: PCA-Components = " + str(n) + ", clusters = " + str(clusters) + "\n")
                try:
                    Des2.NCLUSTER = clusters
                    des = [True, True, True, True, True, True]
                    name = "test"
                    X, y = Des2.getXY(src="D:/Developement/py/Final/Data/Leaves_ImagesOnly/", save=False, name=name , pcaComponents=n, des=des)
                except Exception as e: 
                    print("Error at n: ", n,"clusters: " , clusters)
                    print(e)

                    for C in c_li:
                        for K in kernel_li:
                            avgscore = 0 
                            for rs in range(attemptsPerParameter):
                                try:                                     #print("C = " + str(C) + ", Kernel = " + str(K)+ "\n")
                                    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=testSize)
                                    svm = MultiSVM(kernel=K, C=C)
                                    svm.fit(X_train, y_train)
                                    
                                    score = svm.score(X_test, y_test)
                                    avgscore += score
                                except Exception as e:
                                    print("Error at Kernel: ", str(K), ", C: ", str(C))
                                    print(e)
                            avgscore = avgscore / attemptsPerParameter
                            f.write("score = {" + str(avgscore) + "{ C = {" + str(C) + "{, Kernel = {" + str(K)+ "\n")
                            print("score: ", str(avgscore))
        f.close()
        
        # erstellte Datei auslesen und Daten in Pandas DF anzeigen
        f = open(filename, "r")
        dfli = []
        for line in f.readlines():
            li = line.split("{")
            if len(li)==4:
                pcaC = float(li[1])
                clusters = float(li[3][:-1])
            else:
                score = float(li[1])
                score = str(score)[:5]
                try:
                    C = float(li[3])
                except:
                    C = None
                Kernel = li[5][:-1]
                dfli.append([score, pcaC, clusters, C, Kernel])
        dfnp = np.array(dfli)
        df = pd.DataFrame(dfnp, columns=["score","PCA-Comps", "Clusters", "C", "Kernel"])
        df = df.sort_values("score")
        print(df)
        
def averageDetailedScore():
    attemptsPerParameter = 2
    testSize = 0.3
    kernel = LinearKernel()
    C = None    
            
            
    X = np.load("Best_Parameters_PCAComps24_clusters120_correct_X.npy")
    y = np.load("Best_Parameters_PCAComps24_clusters120_correct_y.npy")
    summ = np.zeros(np.unique(y).shape[0])
    for count in range(attemptsPerParameter):
        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=testSize)
        svm = MultiSVM(kernel=kernel, C=C)
        svm.fit(X_train, y_train)
        dscore = svm.detailedScore(X_test, y_test)
        array = np.array(dscore)
        for line in array:
            summ[int(line[0])] += line[1]
        
    print("Average: ", summ/attemptsPerParameter)   
        
def dfToLatex(df):
    # Pandas df zu Latex-Tabelle
    string = ""
    array = np.array(df)
    for line in array: 
        for element in line:
            string += str(element) + " & "
        string = string[:-2]
        string += " \\\\ \n"
    print(string)
        
        
if __name__ == "__main__":
    
    #Dieser Code berchnet X und y, teilt sie in Test- und Trainingsdaten und trainiert eine SVM darauf
    print("running")
    store = False # True falls gespeichert werden soll
    src = "D:/Developement/py/Final/Data/Leaves_ImagesOnly/" # Falls gespeichert wird unbedingt vorher ImagesOnly Ordner kopieren, umbenennen und den neuen Ordner als src angeben!
    name = "name"
    Des2.NCLUSTERS = 120 # Anzahl CLuster des k-means Algorithmus => Anzahl verschiedene von den lokalen Deskriptoren gefundene Punkte
    des = [True, True, True, True, True, True] # Hiermit können einzelne Deskriptoren deaktiviert werden, Form: Hu-Momente, Haralick, SURF, ORB, SIFT, color histogram
    pcaComponents = 24 # Anzahl Hauptkomponenten, auf welche die PCA die Daten reduziert.
    X,y = Des2.getXYSingle(src=src, save=store, name=name , pcaComponents=pcaComponents, des=des) 
    #X = np.load(name + "_X.npy")
    #y = np.load(name + "_y.npy")
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3)
    svm = MultiSVM(C=None, kernel=LinearKernel())
    svm.fit(X_train, y_train)

    if store:
        np.save("X_test.npy", X_test)
        np.save("X_train.npy",X_train)
        np.save("y_test.npy", y_test)
        np.save("y_train.npy", y_train)
        svm.save("SVM.pkl")
    print(svm.score(X_test, y_test))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
