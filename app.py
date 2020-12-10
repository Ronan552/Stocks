from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

from google.colab import files
files.upload()

#store data from csv
T = pd.read_csv('AAPL-3.csv')

#set date as index
T = T.set_index(pd.DatetimeIndex(T['Date'].values))


T.index.name ='Date'


#manipulate data
#create targe column
T['Price_Up'] = np.where(T['Close'].shift(-1)>T['Close'],1,0)

#remove date column
T = T.drop(columns=['Date'])

#Split data into feature and targe
X =T.iloc[:,0:T.shape[1]-1].values
Y =T.iloc[:,T.shape[1]-1].values

#Split data into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)

#create and train model(Decision Tee Classifier)
tree = DecisionTreeClassifier().fit(X_train, Y_train)

#Show model accuray
print(tree.score(X_test,Y_test))


#show models predictions
tree_predictions = tree.predict(X_test)
print(tree_predictions)


#Show actual value
Y_test
