import pandas as pd
import pandas_profiling 
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler


cols=["fLength","fWidth","fSize","fConc","fConc1","fAsym","fM3Long","fM3Trans","fAlpha","fDist","class"]
df=pd.read_csv("magic04.data", names=cols)
print(df["class"].value_counts())


#print(df.tail())

print(df["class"].unique())
df["class"]=df["class"].replace({'g':0, 'h':1})
print(df.head())
print(df["class"].value_counts())

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

train,valid,test=np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

x=df[cols[:-1]].values
y=df[cols[-1]].values
scaler=StandardScaler()
X=scaler.fit_transform(x)
print(X)