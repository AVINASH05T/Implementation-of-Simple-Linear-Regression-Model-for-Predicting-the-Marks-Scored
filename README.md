# EXPERIMENT: 2
### NAME: AVINASH T
### REG NO: 212223230026
## Implementation of Simple Linear Regression Model for Predicting the Marks Scored:
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
#Program to implement the simple linear regression model for predicting the marks scored.
#Developed by: AVINASH T
#RegisterNumber:  212223230026
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()

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
y_pred
y_test

plt.scatter(x_train,y_train,color="green")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Training set(H vs S)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color="blue")
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("Testing set(H vs S)")
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
![Screenshot 2024-08-30 114254](https://github.com/user-attachments/assets/ae333482-d5bf-4527-9182-2c8690b740f6)
![Screenshot 2024-08-30 113717](https://github.com/user-attachments/assets/170488a7-449a-4a4c-8cba-a33def2d2c5d)

![Screenshot 2024-08-30 113748](https://github.com/user-attachments/assets/a38c3460-17aa-4fb4-aad2-ab1349e61920)
![Screenshot 2024-08-30 113810](https://github.com/user-attachments/assets/58458612-339a-4251-a00c-7e0f56f596bb)
![Screenshot 2024-08-30 113837](https://github.com/user-attachments/assets/dbdba6c5-868c-4c4c-8a67-a5352ce03ac2)
![Screenshot 2024-08-30 113911](https://github.com/user-attachments/assets/702a67c3-3730-4f82-be06-35a2bb6de7b2)
![image](https://github.com/user-attachments/assets/a300e1aa-2db0-456c-aadf-2bc627f8b5df)
![image](https://github.com/user-attachments/assets/0006c21d-afea-444a-abed-423d4cd8eba0)


![Screenshot 2024-08-30 140735](https://github.com/user-attachments/assets/1f07520b-d4c7-4681-a04c-d6133f20952e)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
