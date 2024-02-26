# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
Developed by: MONISH R
RegisterNumber:  212223220061

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
Dataset

![image](https://github.com/monishr288/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147474049/d4685b9d-e498-43e0-bd4a-ad6d955b4490)

Head Values

![head](https://github.com/monishr288/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147474049/72d36cae-5e8f-475e-9bd0-f201b9fb230b)
Tail Values
![image](https://github.com/monishr288/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147474049/813504b9-7f6e-4ac2-bec9-cdddeef3dec6)

X and Y values

![image](https://github.com/monishr288/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147474049/ff8e3af4-3919-41f2-a38e-54201d654992)

Predication values of X and Y

![image](https://github.com/monishr288/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147474049/4d36c7a2-169f-48b0-9af7-a8d6cf7124f9)

MSE,MAE and RMSE

![values](https://github.com/monishr288/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147474049/c62fdb65-910f-479e-aa8a-d172292d63db)

Training Set

![WhatsApp Image 2024-02-26 at 20 23 22_fb9f82ba](https://github.com/monishr288/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147474049/edec5014-4016-490c-b5b5-c5ef8e7f2d64)

Testing Set

![WhatsApp Image 2024-02-26 at 20 24 55_387bb824](https://github.com/monishr288/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147474049/b0bb7ae0-8995-4a05-b6c0-d39bf0a2a6e3)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
