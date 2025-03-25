import pandas as pd
import pandas_profiling 
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler



cols=["fLength","fWidth","fSize","fConc","fConc1","fAsym","fM3Long","fM3Trans","fAlpha","fDist","class"]
df=pd.read_csv("magic04.data", names=cols)
print(df["class"].value_counts())


#print(df.tail())

print(df["class"].unique())
df["class"]=df["class"].replace({'g':0, 'h':1})


#rpt=ProfileReport(df)
#rpt.to_file("Data.html")

'''
for label in cols[:-1]:
    plt.hist(df[df["class"]==0] [label], color="blue", label="gamma", density=True, alpha=0.7)
    plt.hist(df[df["class"]==1] [label], color="red", label="hedron", density=True, alpha=0.7)
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    plt.show()
'''
scaler=StandardScaler()
df[cols[:-1]]=scaler.fit_transform(df[cols[:-1]])

train,valid,test=np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])








print(len(train[train["class"]==0]))
print(len(train[train["class"]==1])) 
#identified more gamma values than hedera in training dataset. so applying oversampling techniques


ros=RandomOverSampler(random_state=42)
