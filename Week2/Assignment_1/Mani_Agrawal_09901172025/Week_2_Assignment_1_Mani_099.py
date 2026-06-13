# Q1. Load the dataset and display the first five records.
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
df=pd.read_csv("Dataset 2.csv")
print(df.head())

# Q2. Determine the number of rows and columns in the dataset. 
print(f"Number of rows and column in the data is {df.shape}")

# Q3. Display all column names. 
print(f"Column in the data are {df.columns}")

# Q4. Identify numerical and categorical features.
print("Datatype of the column are:")
print(df.dtypes)

# Q5. Check whether the dataset contains missing values.
print(df.isna().any())

# Q6. Calculate the average age of users. 
print(f"Average age of users is {round(df['Age'].mean())}")

# Q7. Determine the average watch hours per week. 
print(f"Average watch hours per week of users are {round(df['WatchHoursPerWeek'].mean())} hours.")

# Q8. Find the average monthly spending of users.
print(f"Average monthly spending of the users is {round(df['MonthlySpend'].mean(),2)} (₹)")

# Q9. Count the number of users in each subscription category.
print(df['SubscriptionType'].value_counts())

# Q10. Determine the percentage of users who renewed their subscriptions. 
print((df['SubscriptionRenewed']=='Yes').mean()*100)

# Q11. Convert categorical features into numerical form.
le= LabelEncoder()
df['Gender']=le.fit_transform(df['Gender'])
df['SubscriptionType']=le.fit_transform(df['SubscriptionType'])
df['FavoriteGenre']=le.fit_transform(df['FavoriteGenre'])
df['SubscriptionRenewed']=le.fit_transform(df['SubscriptionRenewed'])

# Q12. Define the feature set (X) and target variable (y) for subscription renewal prediction.
# Feature set
X=df.drop(['UserID','MonthlySpend','SubscriptionRenewed'],axis=1)
# target Variable
Y=df['SubscriptionRenewed']

# Q13. Split the dataset into training and testing sets.
X_train,X_test,Y_train,Y_test = train_test_split(
    X,Y,test_size=0.3,random_state=42
)

# Q14. Train a Decision Tree model to predict whether a user will renew their subscription.
dtModel = DecisionTreeClassifier(random_state=42)
# Train model on the training data
dtModel.fit(X_train,Y_train)
# Predit labels for the test data
predictions = dtModel.predict(X_test)

# Q15. Evaluate the model using accuracy. 
# Accuracy of the Decision Tree
print(f'Decision Tree Accuracy: {accuracy_score(Y_test,predictions)}')

# Q16. Generate and interpret the confusion matrix. 
print("Confusion Matrix:")
print(confusion_matrix(Y_test,predictions))

# Q17. Train a KNN classifier with K = 5. 
# Creating a KNN model
knnModel = KNeighborsClassifier(n_neighbors=5)
# Train model on the training data
knnModel.fit(X_train,Y_train)
#Predictions
predictions_knn=knnModel.predict(X_test)
# Accuracy of the knn model
print(f"knn model accuracy: {accuracy_score(Y_test,predictions_knn)}")

# Q18. Compare the accuracy of KNN with the Decision Tree model.
dtAccuracy=accuracy_score(Y_test,predictions)
knnAccuracy=accuracy_score(Y_test,predictions_knn)
if(dtAccuracy>knnAccuracy):
    print(f"Decision Tree model has better accuracy or perform better than knn model by {(dtAccuracy-knnAccuracy)*100}")
elif(dtAccuracy<knnAccuracy):
    print(f"knn model has better accuracy or preform better than decison tree model by {(knnAccuracy-dtAccuracy)*100}")
else:
    print("Both model has same accuracy")

# Q19.  Train a Linear Regression model to predict monthly spending. 
df=pd.read_csv("Dataset 2.csv")
le= LabelEncoder()
df['Gender']=le.fit_transform(df['Gender'])
df['SubscriptionType']=le.fit_transform(df['SubscriptionType'])
df['FavoriteGenre']=le.fit_transform(df['FavoriteGenre'])
df['SubscriptionRenewed']=le.fit_transform(df['SubscriptionRenewed'])

XReg=df.drop(['UserID','MonthlySpend','SubscriptionRenewed'],axis=1)
YReg=df['MonthlySpend']
xTrainReg,xTestReg,yTrainReg,yTestReg = train_test_split(
    XReg,YReg,test_size=0.2,random_state=42
)
lr = LinearRegression()
lr.fit(xTrainReg,yTrainReg)

# Q20. Predict the monthly spending for a new user and interpret the result.
newUser = pd.DataFrame({
    'Age': [25],
    'Gender': [1],
    'SubscriptionType': [1],
    'WatchHoursPerWeek': [20],
    'DevicesUsed': [2],
    'FavoriteGenre': [0],
    'AdClicks': [15],
})
predictedSpending=lr.predict(newUser)
print(f"Monthly spending of the user is {round(predictedSpending[0],2)}")

# Business Reflection Questions 
 
# 1. Which factors appear to influence subscription renewal the most?  
# The factors appear to influence subscription renewal the most are monthly spending,watch hours per week.

# 2. Why is subscription renewal a classification problem?  
# because it has 2 categories either no or yes. Here the target vatiable is discrete categories not continuous number

# 3. Why is monthly spending a regression problem? 
# because the target variable is continuous number not discrete categories.
 
# 4. Which algorithm performed better for renewal prediction? 
#  Decision tree algorithm performed better for renewal prediction than knn.

# 5. How could the platform use these predictions to improve customer retention? 
# It help platform understand customer behaviour, optimize subscription plan, make data driven business decision
