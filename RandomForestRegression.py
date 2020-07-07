
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("RandomForest.csv",sep=";",header=None)

x=df.iloc[:,0].values.reshape(-1,1)
y=df.iloc[:,1].values.reshape(-1,1)

#%%
from sklearn.ensemble import RandomForestRegressor

#Decision tree de döngü sayısı kısıtlı olduğu için çok verimli bir algoritma değildi
#Ancak randomForest algoritamasında döngü sayısını ayarlayabildiğimiz için daha verimli
#sonuclar elde edebiliriz
rf=RandomForestRegressor(n_estimators=100,random_state=42)
rf.fit(x,y)
print("7.8 seviyesinde fiyatın ne kadar olduğu:",rf.predict([[7.8]]))

x_=np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head=rf.predict(x_)

#visualize
plt.scatter(x,y,color="red")
plt.plot(x_,y_head,color="green")
plt.xlabel("Tribün level")
plt.ylabel("Ucret")
plt.show()