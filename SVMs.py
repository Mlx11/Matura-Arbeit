# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 16:14:16 2018

@author: Arbeiten
"""

# -*- coding: utf-8 -*-
"""
SVM Maturaarbeit
Author: Michael Linder
Date: 23.04.18
"""


try:
    import cvxopt.solvers
except:
    print("cvxopt nicht geladen")
import matplotlib.pyplot as plt
from sklearn import datasets
import  numpy as np
import sklearn.model_selection
from Kernels import *
import operator
import pandas as pd
import pickle

class SVM:
    def __init__(self, kernel, C=None, msg=""):
        # initialisiert die Parameter
        self.kernel = kernel
        try: 
            if C is None:
                self.C = None;
            else: 
                self.C = float(C)
        except Exception as e: 
            print(e)
            return   

    def __str__(self):
        s = "<< Binäre SVM"
        try:
            if self.sv_a:
                s += ", trainiert >>"
        except:
            s += ", nicht trainiert >>"
        return s
        
    def fit(self, X, y):
        # trainiert die SVM auf Trainigsdaten
        n_samples, n_features = X.shape
        # Compute the Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
           for j in range(n_samples):
               K[i,j] = self.kernel.calculate(X[i], X[j])
        # construct P, q, A, b, G, h matrices for CVXOPT
        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1,n_samples), 'd')
        b = cvxopt.matrix(0.0)
        if self.C is None: # hard-margin SVM
           G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
           h = cvxopt.matrix(np.zeros(n_samples))
        else:              # soft-margin SVM
           G = cvxopt.matrix(np.vstack((np.diag(np.ones(n_samples) * -1), np.identity(n_samples))))
           h = cvxopt.matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))
        # solve QP problem
        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        # Lagrange multipliers
        a = np.ravel(solution['x'])
        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5 # some small threshold
        
        
        #if no support vectors found
        if not True in sv:
            sv = a == a.max()
        
        self.X = X
        self.y = y
        self.a = a
        self.sv_a = a[sv]
        self.sv_X = X[sv]
        self.sv_y = y[sv]
        
        #self.compute_w()
        self.compute_b()
        
    def compute_w(self):
        # Berechnet den Parameter w aus den Supportvektoren
        # Wird zur Klassifikation nicht benötigt, da w dort direkt mithilfe der Supportvektoren berechnet wird.
        try:
            self.w = np.sum(self.sv_a[i] * self.sv_y[i] * self.sv_X[i] for i in range(len(self.sv_y)))
        except Exception as e:
            print(e)
            return
        return self.w
        
    def compute_b(self):
        # Berchnet den Parameter b aus den Supportvektoren
        try:
            self.b = self.sv_y[0] - np.sum([self.a[j]*self.y[j]*self.kernel.calculate(self.sv_X[0], self.X[j]) for j in range(len(self.a))])
        except Exception as e:
            print(e)
            return
        
    def predict(self, X):
        #sagt die Klasse/n von einem oder mehrerer Vektore/en voraus
        try:
            predictions = []
            #y = np.where(self.k(X,  + self.b > 0, 1, -1)
            if X.shape[1]:
                for j in range(len(X)):
                    predictions.append(np.sum([self.a[i] * self.y[i] * self.k(self.X[i], X[j]) for i in range(len(self.y))]) + self.b)
                #np.sum(self.sv_multipliers[i] * self.sv_y[i] * self.k(self.sv_X[i], X) for i in range(len(self.sv_y)))
                y = np.asanyarray(predictions)
                y = np.where(y>0, 1, -1)
        except Exception as e:
            try:
                y = np.sum(self.sv_a[i] * self.sv_y[i] * self.k(self.sv_X[i], X) for i in range(len(self.sv_y))) + self.b
            except Exception as e:
                return
        return y

    def k(self, x1, x2):
        #berechnet die Kernelfunktion für Vektoren und Matrizen mit mehreren Vektoren
        case=False
        # x1: nxm matrix, x2: m vector
        try:
            if x1.shape[1]:
                case = "2d 1d arrays"
        # x1: nxm matrix, x2: nxm array
        except:
            pass
        
        try:
            if x1.shape[1] and x2.shape[1]:
                case = "2x2d arrays"
        except:
            pass
        if not case:
            # x1: m vector, x2: m vector
            case = "2x1d array"
            
        if case == "2x2d arrays" :
            products = []
            for x3, x4 in zip(x1,x2):
                products.append(self.k(x3,x4))
            product = np.asanyarray(products)
            print(product)
        elif case == "2d 1d arrays":
            products = []
            for x3 in x1:
                #print("x3:", x3)
                products.append(self.k(x3,x2))
            product = np.asanyarray(products)            
        else:
            product = self.kernel.calculate(x1,x2)
        return product
    
    
    
    
    
    def plotDecisionRegions(self, X,y, title="Decision Regions"):
        # Zeigt die Trennung zweier Klassen eines 2d-Datensatzes
        try:            
            h = .01  # step size in the mesh
            # create a mesh to plot in
            #margin: size of area shown
            margin = 1
            x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
            y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
            h = (x_max - x_min) / 200 # step size in the mesh
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
            fig, ax = plt.subplots()

            Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = np.array(Z)
            Z = Z.reshape(xx.shape)
            plt.contour(xx, yy, Z, colors=("black","green","blue"))
            # Plot the training points
            X1 = X[np.where(y==1)]
            X2 = X[np.where(y==-1)]
            
            plt.scatter(X1[:, 0], X1[:, 1], c="red",edgecolor='k', label="Klasse 1")
            plt.scatter(X2[:, 0], X2[:, 1], c="blue",edgecolor='k', label="Klasse -1")
            #plot sv
            #plt.scatter(self.sv_X[:, 0], self.sv_X[:, 1], c="yellow",edgecolor='k', label="sv")
            
            plt.xlabel('Sepal length')
            plt.ylabel('Sepal width')
            ax.legend()
            if self.C is None:
                plt.title(title + " Hard Margin")
            else: 
                plt.title(title + ", C=" + str(self.C))
            plt.show()
            fig.savefig("fig.png")
        except Exception as e:
            print(e)         
        
        
        
    def score(self, X, y):
        # berechnet den Anteil der richtig Klassifizierten Objekte
        try: 
            y_pred = self.predict(X)
            summ = np.sum(np.where(y_pred == y, 0, 1))
            acc = 1 - (summ / y.shape[0])
        except Exception as e: 
            print(e)
        return acc, summ
        
    
    def load(self, path):
        # lädt eine SVM
        f = open(path, 'rb')
        tmp_dict = pickle.load(f)
        f.close()          
        self.__dict__.clear()
        self.__dict__.update(tmp_dict) 


    def save(self, path):
        #speichert die SVM
        f = open(path, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()
        
        
    
    
    
#----------------------------------------------------------------------------------------------------------------------------------
        
    
class MultiSVM():
    # SVM, welche auf Daten mit beliebig vielen Klassen klassifiziert werden kann
    
    def __init__(self, kernel=LinearKernel(),C=None):
        self.classifiers = []
        self.kernel = kernel
        self.C = C
    
    def fitSVM(self, X, y, c1, c2):
        X1 = X[np.isin(y, [c1,c2])]
        y1 = y[np.isin(y, [c1,c2])]               
        y1 = np.where(y1==c1, 1, -1)
        svm = SVM(kernel=self.kernel, C=self.C, msg=str(c1)+"/"+str(c2))
        svm.fit(X1,y1)
        return svm
        
        
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.classes = np.unique(y)
        self.numberOfClasses = np.unique(y).shape[0]
        
        for c1 in self.classes:
            status = True
            for c2 in self.classes:
                if c1 == c2:
                    status = False
                if status:
                    self.classifiers.append([self.fitSVM(X,y, c1, c2), c1, c2])
                    
    def predictSingle(self, X):
        # sagt die Klasse eines einzelnen Vektors voraus
        vote = {}
        for c in self.classes:
            vote[str(c)] = 0
        for obj in self.classifiers:
            pred = obj[0].predict(X)
            if pred > 0.9: # pred == 1 geht nicht, da Fehler aufgrund Binärsystem
                vote[str(obj[1])] += 1
            else: 
                 vote[str(obj[2])] += 1
        return int(max(vote.items(), key=operator.itemgetter(1))[0])
    
    
    def predict(self, X):
        # sagt die Klasse von einem oder mehreren Vektor/en voraus
        try:
            predictions = []
            #y = np.where(self.k(X,  + self.b > 0, 1, -1)
            if X.shape[1]:
                # falls mehrere Vektoren
                for j in range(len(X)):
                    predictions.append(self.predictSingle(X[j]))
                #np.sum(self.sv_multipliers[i] * self.sv_y[i] * self.k(self.sv_X[i], X) for i in range(len(self.sv_y)))
                y = np.asanyarray(predictions)
        except Exception as e:
            #falls ein einzelner Vektor
            try:
                y = self.predictSingle(X)
            except Exception as e:
                print(e)
                return
        
        return y

    def plotDecisionRegions(self, X,y, title="Decision Regions Multi SVM"): 
        # zeigt die Trennlinier der Klassen für 2d Daten         
        h = .01  # Genauigkeit des meshgrids
        margin = 1 # abstand äusserster Punkt zum Rand des Plots
        x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
        y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
        h = (x_max - x_min) / 200 # step size in the mesh
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        fig, ax = plt.subplots()
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = np.array(Z)
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, colors=("black","green","blue"))
        
        
        # Stellt die Daten im plot dar
        for cls, color in zip(np.unique(np.array(y)), ["red","blue","yellow","green","violet"]):
            Xcls = X[np.where(y==cls)]
            plt.scatter(Xcls[:, 0], Xcls[:, 1],edgecolor='k', label=str(cls))                
        plt.xlabel('Sepal length')
        plt.ylabel('Sepal width')
        ax.legend()
        plt.title(title)
        plt.show()
        fig.savefig("fig.png")       
        
    def score(self, X, y):
        # berechnet den Anteil der richtig klassifizierten Elemente
        y_pred = self.predict(X)
        summ = np.sum(np.where(y_pred == y, 0, 1))
        acc = 1 - (summ / y.shape[0])
        return acc
    
    def detailedScore(self, X, y):
        # berechnet den Anteil der richtig klassifizierten Elemente pro Klasse
        dScores = []
        y_pred = self.predict(X)
        uniquenr = np.unique(y, return_counts=True)
        for label, count in zip(uniquenr[0], uniquenr[1]):
            y_spec = np.where(y == label, label, -1)
            summ = np.sum(np.where(y_pred == y_spec, 1, 0))
            acc = summ / count
            dScores.append([label, acc])
        dScores = np.array(dScores)
        labels = ["label", "Score"]
        df = pd.DataFrame.from_records(dScores, columns=labels)
        df = df.sort_values(by=['Score'])
        return df
    
        
    def load(self, path):
        #lädt die svm
        f = open(path, 'rb')
        tmp_dict = pickle.load(f)
        f.close()          
        self.__dict__.clear()
        self.__dict__.update(tmp_dict) 


    def save(self, path):
        #speichert die SVM
        f = open(path, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()
        
    
if __name__ == "__main__":
    print("running")



    
    
    
    
    
    
    
    
    
    