
# =>=>=>=> importing libraries <= <= <= <=
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# =>=>=>=> getting data <= <= <= <=
data = pd.read_csv("student_info.csv")
# print(data.head())
# print(data.tail())
# print(data.shape)

# data.info()

des = data.describe()
# =>=>=>=> scatter diagram for data <= <= <= <=

# plt.scatter(x=data.study_hours, y=data.student_marks,color = 'hotpink')
# plt.xlabel("Students study Hours")
# plt.ylabel("Students marks")
# plt.title("Scatter plot of Students study hours and students marks ")
# plt.show()

# =>=>=>=> histogram for data <= <= <= <=

# plt.hist(x=data.study_hours)
# plt.show()



# =>=>=>=> data cleaning <= <= <= <=

# print(data.isnull().sum()) #checking how many columns are null

# =>=>=>=> Replace null value with mean of that column<= <= <= <=
mean = data["study_hours"].mean()
# print(mean)
clean_data = data.fillna(mean)

# clean_data.info()


# =>=>=>=> split dataset <= <= <= <=
X = clean_data.drop("student_marks", axis="columns")
y= clean_data.drop("study_hours", axis="columns")
# print(X.shape)
# print(y.shape)


# =>=>=>=> test and train data<= <= <= <=

X_train,X_test, y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=51)


# =>=>=>=> use linear regression model to test data <= <= <= <=
# ( y=mx+c)

lr = LinearRegression()
lr.fit(X_train,y_train)

# =>=>=>=> prdic data by yourself <= <= <= <=

# study_hours = int(input("enter study hours : "))
# predicted_marks = lr.predict([[study_hours]])[0][0]
# print("you will get : ", int(predicted_marks))


# =>=>=>=> check predicted data with actual data <= <= <= <=
y_pred = lr.predict(X_test)
new_dataframe = pd.DataFrame(np.c_[X_test, y_test, y_pred], columns=["Study_hours", "Student_marks_Original","Student_marks_predicted"])
# print(new_dataframe)


# =>=>=>=> test accuracy <= <= <= <=
score = lr.score(X_test,y_test)
score = score*100
print(f"Accuracy is :  {score:.2f}")


# =>=>=>=>scatter diagram for train data <= <= <= <=
# plt.scatter(X_train, y_train)
# plt.show()

# =>=>=>=>scatter diagram and plot diagram for test data and predicted data together <= <= <= <=
plt.scatter(X_test,y_test)
plt.plot(X_test,y_pred, color="r")
plt.show()