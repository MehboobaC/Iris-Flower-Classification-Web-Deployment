import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
data=pd.read_excel('iris.xls')
X = data.drop(['Classification'], axis=1)
y = data['Classification']
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.2)
from sklearn.svm import SVC
sm = SVC(kernel='linear')
m=sm.fit(X_train, y_train)
pickle.dump(sm,open('model.pkl','wb'))
