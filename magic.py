import pandas as pd


cols=["fLength","fWidth","fSize","fConc","fConc1","fAsym","fM3Long","fM3Trans","fAlpha","fDist"]
df=pd.read_csv("magic04.data", names=cols)

print(df.head())
