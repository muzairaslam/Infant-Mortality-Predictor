import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

dfx =pd.read_csv("infants_final.csv")
dfx = dfx.drop(["Unnamed: 0"],axis =1)
dfy =pd.read_csv("target.csv")
dfy = dfy.drop(["Unnamed: 0"],axis =1)


X_train, X_test, Y_train, Y_test = train_test_split(dfx,dfy, test_size=0.2)


forest = RandomForestClassifier(n_estimators=100, random_state=0,oob_score = True,bootstrap = True)
forest.fit(X_train, Y_train)


pickle.dump(forest,open("model.pkl","wb"))