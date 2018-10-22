# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 22:23:52 2018

@author: Arbeiten
"""


import tkinter as tk 
from tkinter.filedialog import askopenfilename, askopenfilenames
import numpy as np
from SVMs import SVM, MultiSVM
import pandas as pd
from Kernels import LinearKernel, PolyKernel, RBFKernel

class Window:

    WIDTHBUTTON = 20
    BACKGROUNDLABEL = "#2277FF"
    BACKGROUNDBUTTON = "#DDDDDD"
    BACKGROUND = "#BBBBBB"
    
    
    def __init__(self):
        self.window = tk.Tk()
        self.window.geometry("650x900")
        
        
        # Frame laden
        self.frameLoad = tk.Frame(self.window, bg=Window.BACKGROUND, width = 1000, padx=10, pady=20)
        self.frameLoad.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        self.labelLoad = tk.Label(self.frameLoad, text="SVM & Testdaten laden", bg=Window.BACKGROUNDLABEL)
        self.labelLoad.grid(row=1, column=1, columnspan=4, sticky='we', pady=(0, 10))
        
        self.LoadButtonSVM = tk.Button(self.frameLoad, text="SVM laden", command=self.loadSVM, width=Window.WIDTHBUTTON, bg=Window.BACKGROUNDBUTTON)
        self.LoadButtonSVM.grid(row=2, column=1)
        
        self.LoadButtonMultiSVM = tk.Button(self.frameLoad, text="Multi SVM Laden", command=self.loadMultiSVM,width=Window.WIDTHBUTTON, bg=Window.BACKGROUNDBUTTON)
        self.LoadButtonMultiSVM.grid(row=2, column=2)
        
        self.LoadButtonX = tk.Button(self.frameLoad, text="X laden", command=self.loadX,width=Window.WIDTHBUTTON, bg=Window.BACKGROUNDBUTTON)
        self.LoadButtonX.grid(row=2, column=3)
        
        self.LoadButtony = tk.Button(self.frameLoad, text="y laden", command=self.loady,width=Window.WIDTHBUTTON, bg=Window.BACKGROUNDBUTTON)
        self.LoadButtony.grid(row=2, column=4)
        
        self.labelButtonSVM = tk.Label(self.frameLoad, text="None", bg=Window.BACKGROUND)
        self.labelButtonSVM.grid(row=3, column=1, sticky="we")
        
        self.labelButtonMultiSVM = tk.Label(self.frameLoad, text="None", bg=Window.BACKGROUND)
        self.labelButtonMultiSVM.grid(row=3, column=2, sticky="we")
        
        self.labelButtonX = tk.Label(self.frameLoad, text="None", bg=Window.BACKGROUND)
        self.labelButtonX.grid(row=3, column=3, sticky="we")
        
        self.labelButtony = tk.Label(self.frameLoad, text="None", bg=Window.BACKGROUND)
        self.labelButtony.grid(row=3, column=4, sticky="we")
        
        self.ButtonClassifyImage = tk.Button(self.frameLoad, text="Bild klassifizieren", command=self.classifyImg, width=Window.WIDTHBUTTON, bg=Window.BACKGROUNDBUTTON)
        self.ButtonClassifyImage.grid(row=4, column=1)
        
        # Frame erstellen
#        self.frameCreate = tk.Frame(self.window, bg="#888888", width = 550,)
#        self.frameCreate.pack()
        self.frameCreate = self.frameLoad
        
        self.labelCreate = tk.Label(self.frameCreate, text="SVM erstellen", bg=Window.BACKGROUNDLABEL )
        self.labelCreate.grid(row=13, column=1, columnspan=4, sticky='we', pady=(20, 10))
        
        self.labelKernel = tk.Label(self.frameCreate, text="Kernel auswählen:", bg=Window.BACKGROUND)
        self.labelKernel.grid(row=14, column=1, sticky="w")
        
        choices = { 'Linear','RBF','Poly'}
        self.varKernel = tk.StringVar(self.window)
        self.varKernel.set('Linear') # 
        self.DropdownKernel = tk.OptionMenu(self.frameCreate, self.varKernel, *choices)
        self.DropdownKernel.grid(row=14, column=2, sticky="w")
        
        self.varHardMargin = tk.IntVar(value=1)
        self.CheckHardMargin = tk.Checkbutton(self.frameCreate, text="Hard-Margin SVM", variable=self.varHardMargin, bg=Window.BACKGROUND)
        self.CheckHardMargin.grid(row=15, column=1, sticky="w")
        
        self.labelC = tk.Label(self.frameCreate,text="C", bg=Window.BACKGROUND)
        self.labelC.grid(row=16, column=1, sticky="w")
        
        self.varC = tk.StringVar()
        self.EntryC = tk.Entry(self.frameCreate, textvariable=self.varC)
        self.EntryC.grid(row=16, column=1, sticky="e")
        
        self.labelPotenzC = tk.Label(self.frameCreate,text="* 10^", bg=Window.BACKGROUND)
        self.labelPotenzC.grid(row=16, column=2, sticky="w")
        
        self.varPotenzC = tk.StringVar()
        self.EntryPotenzC = tk.Entry(self.frameCreate, textvariable=self.varPotenzC, width=18)
        self.EntryPotenzC.grid(row=16, column=2, sticky="e")
        
        self.labelTrainingsdaten = tk.Label(self.frameCreate, text="Trainingsdaten laden", bg=Window.BACKGROUNDLABEL )
        self.labelTrainingsdaten.grid(row=17, column=1, columnspan=4, sticky='we', pady=(20, 10))
        
        self.LoadButtonX_train = tk.Button(self.frameCreate, text="X laden", command=self.loadX_train, width=Window.WIDTHBUTTON, bg=Window.BACKGROUNDBUTTON)
        self.LoadButtonX_train.grid(row=18, column=1, columnspan=2)
        
        self.LoadButtony_train = tk.Button(self.frameCreate, text="y laden", command=self.loady_train, width=Window.WIDTHBUTTON, bg=Window.BACKGROUNDBUTTON)
        self.LoadButtony_train.grid(row=18, column=3, columnspan=2)
        
        self.labelButtonX_train = tk.Label(self.frameLoad, text="None", bg=Window.BACKGROUND)
        self.labelButtonX_train.grid(row=19, column=1, sticky="we", columnspan=2)
        
        self.labelButtony_train = tk.Label(self.frameLoad, text="None", bg=Window.BACKGROUND)
        self.labelButtony_train.grid(row=19, column=3, sticky="we", columnspan = 2)
        
        self.ButtonTrainSVM = tk.Button(self.frameCreate, text="SVM trainieren", command=self.trainSVM, bg=Window.BACKGROUNDBUTTON, width=Window.WIDTHBUTTON)
        self.ButtonTrainSVM.grid(row=20, column=1, columnspan=2, pady=20)
        
        self.Output = tk.Label(self.frameCreate,text="", bg=Window.BACKGROUND)
        self.Output.grid(row=21, column=1, columnspan=4, pady=50)
        
        #SVM objects
        
        self.svm = None
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.kernel = None
        self.C = None

        self.window.mainloop()
        
        
    
    def loadSVM(self):
        try:
            path = askopenfilename()
            self.svm = SVM(kernel=None)
            self.svm.load(path)
            self.labelButtonSVM["text"] = "loaded"
            self.labelButtonMultiSVM["text"] = "None"
            self.Output["text"] = ""
        except:
            self.Output["text"] = "SVM konnte nicht geladen werden!"
        
    def loadMultiSVM(self):
        try:
            path = askopenfilename()
            self.svm = MultiSVM(kernel=None)
            self.svm.load(path)
            self.labelButtonMultiSVM["text"] = "loaded"
            self.labelButtonSVM["text"] = "None"
            self.Output["text"] = ""
        except Exception as e:
            self.Output["text"] = "MultiSVM konnte nicht geladen werden!"
            print(e)
        
    def loadX(self):
        try:
            path = askopenfilenames()
            Xtuple = [np.load(file) for file in path]
            self.X = np.vstack(Xtuple)
            print(path)
        except:
            self.Output["text"] = "X konnte nicht geladen werden!"  
            self.X = None
        if self.X is not None:
            self.labelButtonX["text"] = "loaded"
            self.Output["text"] = ""
        
    def loady(self):
        try:
            path = askopenfilenames()
            ytuple = [np.load(file) for file in path]
            self.y = np.hstack(ytuple).flatten()
        except:
            self.Output["text"] = "y konnte nicht geladen werden!"   
            self.y = None
        if self.y is not None:
            self.labelButtony["text"] = "loaded"
            self.Output["text"] = ""

    def classifyImg(self):
        if self.svm is not None:
            if self.X is not None:
                pred = self.svm.predict(self.X)
                if self.y is not None:
                    percent = str(100 * np.where(pred ==  self.y, 1, 0).sum() / self.y.shape[0])[:5] + "%"
                    self.Output["text"] = str(np.where(pred == self.y, 1, 0).sum()) + " von " + str(self.y.shape[0]) + " korrekt! Dies sind " + percent
                else:
                    new = []
                    for i in range(0, len(pred), 5):
                        new.append(pred[i : i+5])
                    self.Output["text"] = pd.DataFrame(new).to_string(header=False, na_rep="", index_names = False, col_space=10, index=False, justify="left")

# -------------------------------------- SVM erstellen ------------------------------------
     
    def trainSVM(self):
        print()
        kernels = { 'Linear':LinearKernel(),'RBF':RBFKernel(),'Poly':PolyKernel()}
        if self.varHardMargin.get() == 1:
            C = None
        else:
            C = float(self.varC.get())*(10**float(self.varPotenzC.get()))
            
        if np.unique(self.y_train).shape[0] == 2:
            self.Output["text"] = "trainiere SVM, dies kann einige Sekunden dauern!"
            self.svm = SVM(kernels[self.varKernel.get()], C=C)
        else:
            self.Output["text"] = "trainiere MultiSVM, dies kann einige Sekunden dauern!"
            self.svm = MultiSVM(kernel=kernels[self.varKernel.get()], C=C)
        self.svm.fit(self.X_train, self.y_train)
        self.Output["text"] = "SVM trainiert!"
        self.labelButtonSVM["text"] = "loaded"
        
        
    def loadX_train(self):
        try:
            path = askopenfilenames()
            Xtuple = [np.load(file) for file in path]
            self.X_train = np.vstack(Xtuple)
        except:
            self.Output["text"] = "X konnte nicht geladen werden!"  
            self.X_train = None
        if self.X_train is not None:
            self.labelButtonX_train["text"] = "loaded"
            self.Output["text"] = ""
        
    def loady_train(self):
        try:
            path = askopenfilenames()
            ytuple = [np.load(file) for file in path]
            self.y_train = np.hstack(ytuple).flatten()
        except:
            self.Output["text"] = "y konnte nicht geladen werden!"   
            self.y_train = None
        if self.y_train is not None:
            self.labelButtony_train["text"] = "loaded"
            self.Output["text"] = ""
            

if __name__ == '__main__':
  app = Window() 