import pandas as pd
import pandas_profiling 
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
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


for label in cols[:-1]:
    plt.hist(df[df["class"]==0][label], color='blue', label='gamma',  alpha=0.7, density=True)
    plt.hist(df[df["class"]==1][label], color='red', label='hadron', alpha=0.7, density=True)
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    plt.show()


