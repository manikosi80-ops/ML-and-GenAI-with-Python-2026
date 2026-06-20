# Part A: Understanding the Dataset
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("agriculture_yield_dataset.csv")

# Q1 Dataset Overview
print(f"Rows and Columns : {df.shape}")
print(f"Column Names : {df.columns.tolist()}")
print("First 10 Records :")
print(df.head(10))

# Q2. Data Types and Missing Values
print("Data type of the columns is:")
print(df.dtypes)
missingValues=df.isnull().sum()
if(missingValues.sum()>0):
    isMissing=True
    print("Some data is missing in the given dataset.")
    print("Columns with missing values are:")
    print(missingValues[missingValues>0])
else:
    print("NO data is missing in the dataset.")

# Q3. Descriptive Statistics
print("Statistical Analysis of the data : ")
print(df.describe()) 
print(f"{df.describe().loc['mean'].idxmax()} has the highest mean.")
print(f"{df.describe().loc['std'].idxmax()} has the highest standard deviation")

# Part B: Exploratory Data Analysis (EDA) 

# Q4. Distribution Analysis

# Create histograms for rainfall_mm 
plt.figure()
sns.histplot(data=df,x='rainfall_mm',color='red',kde=True)
plt.xlabel('Rainfall in mm')
plt.ylabel("Frequency")
plt.title("Rainfall Distribution")
plt.savefig("Rainfall distribution")
plt.show()
# Rainfall values range from about 300 mm to 1200 mm.
# The highest frequency of rainfall observations occurs around 800–900 mm.

# Create histograms for temperature_c
plt.figure()
sns.histplot(data=df,x='temperature_c',color='blue',kde=True)
plt.title("Temperature Distribution")
plt.xlabel("Temperature in c")
plt.ylabel("Frequency")
plt.savefig("Temperature distribution")
plt.show()
# Temperature values range from approximately 18°C to 38°C.
# The highest frequency of temperature observations occurs around 21.5-23 C.

# Create histograms for fertilizer_kg 
plt.figure()
sns.histplot(data=df,x='fertilizer_kg',color='pink',kde=True)
plt.title("Fertilizer Distribution")
plt.xlabel("Fertilizer in kg")
plt.ylabel("Frequency")
plt.savefig("Fertilizer distribution")
plt.show()
# Fertilizer application ranges from 50 kg to 250 kg.

# Create histograms for yield_ton_per_hectare  
plt.figure()
sns.histplot(data=df,x='yield_ton_per_hectare',color='green',kde=True)
plt.title("Yield Distribution")
plt.xlabel("Yield ton per hectare")
plt.ylabel("Frequency")
plt.savefig("Yield distribution")
plt.show()
# Yield values range from approximately 2 to 8 tons per hectare.
# Most observations are concentrated around 5 tons per hectare.

# Q5. Crop Type Analysis 
print("Number of records for each crop type is:")
print(df['crop_type'].value_counts())

# Create a count plot (bar chart) for crop_type.  
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='crop_type')
plt.title("Frequency of Each Crop Type")
plt.xlabel("Crop Type")
plt.ylabel("Count")
plt.savefig("Crop type")
plt.show()

# Which crop appears most frequently?
print(f"Crop with the highest frequency is {df['crop_type'].value_counts().idxmax()}")

# Q6. Soil Type Analysis
print(" the frequency of each soil type:")
print(df['soil_type'].value_counts())

# Create a count plot for soil_type. 
plt.figure()
sns.countplot(data=df,x='soil_type',color='red')
plt.title("Frequency of each soil type")
plt.xlabel("Soil Type")
plt.savefig("Soil Type")
plt.show()

# Which soil type is most common? 
print(f"Most common soil type is : {df['soil_type'].value_counts().idxmax()}")

# Q7. Yield Distribution 
# Create a histogram of yield_ton_per_hectare. 
plt.figure()
sns.histplot(data=df,x='yield_ton_per_hectare',color='green',kde=True)
plt.title("Yield")
plt.xlabel("Yield in ton per hectare")
plt.savefig("Yield  Distribution")
plt.show()

skewness=df['yield_ton_per_hectare'].skew()
if(skewness>0.5 or skewness<-0.5 ):
    print("Distribution is not symmetric")
    print(f"The skewness in the graph is {skewness}")
else:
    print("Distribution is normal")

# Q8. Scatter Plot Analysis
# Create scatter plots of rainfall_mm vs yield_ton_per_hectare  
plt.figure()
sns.scatterplot(data=df,x='rainfall_mm',y='yield_ton_per_hectare')
plt.xlabel("Rainfall in mm")
plt.title("Rainfall vs Yield")
plt.savefig("Rainfall vs Yield")
plt.show()

# Create scatter plots of fertilizer_kg vs yield_ton_per_hectare 
plt.figure()
sns.scatterplot(data=df,x='fertilizer_kg',y='yield_ton_per_hectare')
plt.title("Fertilizer vs Yield")
plt.xlabel("Fertilizer in kg")
plt.savefig("Fertilizer vs Yield")
plt.show()

# Rainfall has a stronger relationship with yield than fertilizer.

# Q9. Correlation Analysis 
numerical_df=df.drop(['crop_type','soil_type'],axis=1)
print(numerical_df.corr())
plt.figure()
sns.heatmap(numerical_df.corr(),cmap="Reds",annot=True)
plt.title("Correlation Matrix")
plt.savefig("Correlation matrix")
plt.show()

# Irrigation hours , fertilizer in kg , rainfall mm are most correlated with yield

# Q10. Group-Based Analysis 
print(df.groupby('crop_type')['yield_ton_per_hectare'].mean())
print(df.groupby('soil_type')['yield_ton_per_hectare'].mean())
print(f'Crop type with highest average yield is {df.groupby('crop_type')['yield_ton_per_hectare'].mean().idxmax()}')
print(f'Soil type with highest average yield is {df.groupby('soil_type')['yield_ton_per_hectare'].mean().idxmax()}')

# Part C: Data Preparation

# Q11. Feature Encoding 
categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
print(categorical_cols)
df_encoded=pd.get_dummies(df,columns=categorical_cols,drop_first=True)
df_encoded.info()
print(df_encoded.head())

# Q12. Feature Selection
from sklearn.model_selection import train_test_split
X = df.drop(['yield_ton_per_hectare','crop_type','soil_type'], axis=1)
Y = df['yield_ton_per_hectare']

# Q13. Train-Test Split
X_train,X_test,y_train,y_test = train_test_split(
    X,Y,test_size=0.2,random_state=42
)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Q14. Linear Regression Model
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
# Training the model
lr.fit(X_train,y_train)
amount = lr.predict(X_test)
print(amount)
print("Intercept:", lr.intercept_)
print('Coefficient :',lr.coef_)
print("Highest Positive Feature:", X.columns[lr.coef_.argmax()])