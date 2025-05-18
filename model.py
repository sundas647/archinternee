import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
data = pd.read_csv("Crop_recommendation.csv")


data.head()
data.shape

data.isnull().sum()

x=data.iloc[:,:-1] #features
y=data.iloc[:,-1] #labels



x_test,x_train,y_test,y_train=train_test_split(x,y,test_size=0.2,random_state=42)
model=RandomForestClassifier()
model.fit(x_train,y_train)
pickle.dump(model,open('model.pkl','wb'))
