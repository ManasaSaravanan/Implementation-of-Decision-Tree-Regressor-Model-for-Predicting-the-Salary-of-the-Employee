# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee
## MANASA S
## 212224220059
## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.

2.Upload the dataset and check for any null values using .isnull() function.

3.Import LabelEncoder and encode the dataset.

4.Import DecisionTreeRegressor from sklearn and apply the model on the dataset.

5.Predict the values of arrays.

6.Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.

7.Predict the values of array.

8.Apply to new unknown values.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: LOKESHWARAN S
RegisterNumber:212224240080
import pandas as pd


data = pd.read_csv("Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()

x = data[["Position", "Level"]]
y = data["Salary"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)
y_pred = df.predict(x_test)

from sklearn import metrics
mse = metrics.mean_squared_error(y_test, y_pred)
mse

r2 = metrics.r2_score(y_test, y_pred)
r2

dt.predict([[5,6]])
*/
```

## Output:
<img width="331" height="301" alt="image" src="https://github.com/user-attachments/assets/d9a641b6-dd9d-46a9-ae28-f9f0e32bb89a" />

<img width="370" height="281" alt="image" src="https://github.com/user-attachments/assets/229c84ef-ebe2-4b5e-a735-ae3b0ad45416" />

<img width="199" height="259" alt="image" src="https://github.com/user-attachments/assets/5ed07a64-cccc-4752-8772-313b136004b1" />

<img width="127" height="74" alt="image" src="https://github.com/user-attachments/assets/17818de2-f7f5-4dd5-825e-2955b429852c" />

<img width="182" height="69" alt="image" src="https://github.com/user-attachments/assets/0b5bf2f6-39be-4a67-ad1d-8ed9119b51f6" />

<img width="1609" height="126" alt="image" src="https://github.com/user-attachments/assets/4a17359d-c257-44e0-9b64-5004b79bfef7" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
