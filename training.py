import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_blobs

url ="diabetes.csv"
data = pd.read_csv(url)
print(data)

print('x')
x = data.drop(['Outcome'],axis = 1)
x.head()
print(x)

print('y')
y = data['Outcome']
print(y)

from sklearn.preprocessing import MinMaxScaler

print('x scaler')
scaler = MinMaxScaler()
x = scaler.fit_transform(x)
print(x)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3, random_state=1)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(xtrain,ytrain)

print('ypred')
ypred = knn.predict(xtest)
print(ypred)

print('ytest')
print(ytest)

from sklearn.metrics import confusion_matrix,classification_report
print('confusion_matrix(ytest,ypred)')
print(confusion_matrix(ytest,ypred))
print('classification_report(ytest,ypred)')
print(classification_report(ytest,ypred))

error_rate = []
k_error = []
def error(e):
    return e['error']

for i in range(1,40):
  knn = KNeighborsClassifier(n_neighbors=i)
  knn.fit(xtrain,ytrain)
  print(knn)
  pred_i = knn.predict(xtest)
  error_rate.append(np.mean(pred_i != ytest))
  k_error.append({'k':i,"error":np.mean(pred_i != ytest)})
print(k_error)
k_error.sort(key= error)
print(k_error[0])


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue',linestyle='--',markersize=10,markerfacecolor='red',marker='o')
plt.title('K versus Error rate')
plt.xlabel('k')
plt.ylabel('Error rate')

print('k= 11')
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(xtrain,ytrain)
predictions = knn.predict(xtest)
print('confusion_matrix(ytest,ypred)')
print(confusion_matrix(ytest,ypred))
print('classification_report(ytest,ypred)')
print(classification_report(ytest,ypred))

