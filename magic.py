import pandas as pd


cols=["fLength","fWidth","fSize","fConc","fConc1","fAsym","fM3Long","fM3Trans","fAlpha","fDist","class"]
df=pd.read_csv("magic04.data", names=cols)
print(df["class"].value_counts())

#print(df.tail())
print(df["class"].unique())
df["class"]=df["class"].replace({'g':0, 'h':1})
print(df.head())
print(df["class"].value_counts())
