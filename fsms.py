import pandas as pd
import pandas_profiling
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import StandardScaler
import numpy as np
from imblearn.over_sampling import ADASYN  # Importing ADASYN for oversampling



class Fsms:

    cols=["fLength","fWidth","fSize","fConc","fConc1","fAsym","fM3Long","fM3Trans","fAlpha","fDist","class"]
    
    
    def get_data(self):
        self.df=pd.read_csv(r'C:\Users\ramgo\OneDrive\Desktop\Learn\MagicGammaTelescope\magic04.data', names=Fsms.cols)
        print(self.df.head())
    
    def generate_report(self):
        rpt=ProfileReport(self.df)
        rpt.to_file("TelescopeData.html")

    def binary_enocoding(self):
        self.df["class"]=self.df["class"].replace({"g":0, "h":1})
        #print(self.df.head())
        #print(self.df.tail())

    def visualizations(self):
        for label in Fsms.cols[:-1]:
            plt.hist(self.df[self.df["class"]==0] [label], color="blue", label="gamma", density=True, alpha=0.7)
            plt.hist(self.df[self.df["class"]==1] [label], color="red", label="hedron", density=True, alpha=0.7)
            plt.title(label)
            plt.xlabel(label)
            plt.ylabel("Probability")
            plt.legend()
            plt.show()

    def standard_scalar(self):
        self.x=self.df[Fsms.cols[:-1]].values
        self.y=self.df[Fsms.cols[-1]].values
        Scalar=StandardScaler()
        self.x=Scalar.fit_transform(self.x)
        self.df[Fsms.cols[:-1]] = self.x  #assigning scaled data to dataframe
    
    def train_test_valid(self):
        self.train,self.valid,self.test=np.split(self.df.sample(frac=1), [int(0.6*len(self.df)), int(0.8*len(self.df))])
        '''cross checking percentages'''
        #print(len(train)/len(self.df)*100)
        #print(len(valid)/len(self.df)*100)
        #print(len(test)/len(self.df)*100)
        
    def focus_train(self):
        print(self.train["class"].value_counts())
        print(self.train["class"].value_counts()[0]) 
        print(self.train["class"].value_counts()[1]) #we have more 0's i.e., gamma values in training set which might imbalance and model may biased towards gamma

    def OverSampling(self):
        adasyn=ADASYN(sampling_strategy=1.0, random_state=42) #random state=42 put the data same doesn't matter how many time you run/execute
        x_train, y_train=self.train[Fsms.cols[:-1]], self.train["class"]
        x_train,y_train=adasyn.fit_resample(x_train,y_train)
        print(y_train.value_counts())
            

    def Execution_Order(self):
        self.get_data()
        #self.generate_report()
        self.binary_enocoding()
        #self.visualizations()
        self.standard_scalar()
        self.train_test_valid()
        #self.focus_train()
        self.OverSampling()
        


obj=Fsms()
obj.Execution_Order()
