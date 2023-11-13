# Mini-Project
# Road Accident Severity Analysis:
# Mini Project by
```
NAME : SABITHA P
REG_NO : 212222040137
```
# AIM
To analyse the Road Accident dataset and to understand about severity of the accidents.
# ALGORITHM
### STEP 1
Read the given Dataset
### STEP 2
Perform Data Cleaning operations and Outlier Detection and Removal
### STEP 3
Perform Univariate Analysis and Multivariate Analysis.
### STEP 4
Apply ,Feature Encoding, Feature Scaling and Feature transformation and selection techniques to all the features of the data set.
### STEP 5
Apply data visualization techniques to identify the patterns of the data.

# CODE
```
Developed By: Sabitha P
Reg No: 212222040137
```
# Import necessary libraries and read the dataset
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("/content/RTA Dataset.csv.zip")
df
```

# Output
![image](https://github.com/sabithapaulraj/Mini-Project/assets/118343379/5b02864f-3db1-4713-9c53-07c848e2bef1)

```python
df.head()
df.tail()
df.shape
df.info()
df.describe()
```
# Output
![image](https://github.com/sabithapaulraj/Mini-Project/assets/118343379/d649fdec-ae8a-4f8b-b99d-6c77fb21ba38)
![image](https://github.com/sabithapaulraj/Mini-Project/assets/118343379/bf96931a-36a5-4af5-87cf-d5a026f771f6)
![image](https://github.com/sabithapaulraj/Mini-Project/assets/118343379/fc466391-a974-41c8-9c7a-9e99387a174b)
![image](https://github.com/sabithapaulraj/Mini-Project/assets/118343379/5a36a81c-0183-402d-ad95-297f4ee5ca28)
![image](https://github.com/sabithapaulraj/Mini-Project/assets/118343379/b74834a6-2160-4200-ae4b-9012be75ae7a)

# Data Cleaning
```python
# Check for missing values
print(df.isnull().sum())

# Handle missing values 
df.dropna(inplace=True)
df

# Check for duplicate rows
print("Number of duplicate rows:", df.duplicated().sum())

```

# Output
![image](https://github.com/sabithapaulraj/Mini-Project/assets/118343379/df652056-650f-4745-879c-f7b05f3dbcaa)
![image](https://github.com/sabithapaulraj/Mini-Project/assets/118343379/78548c06-ee17-4b78-aec7-54f58c91553b)
![image](https://github.com/sabithapaulraj/Mini-Project/assets/118343379/616676cd-770a-463f-ad6c-e3e62728a73b)

# Outlier Detection and Removal:

```python
# Detect outliers using boxplots
plt.figure(figsize=(9, 6))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.show()

# Handling Outliers:
# Calculate IQR for each column
Q1 = df.quantile(0.25, numeric_only=True)
Q3 = df.quantile(0.75, numeric_only=True)
IQR = Q3 - Q1

# Use the align method to align DataFrames before the comparison
df_no_outliers = df[~((df < (Q1 - 1.5 * IQR)).align(df)[0] | (df > (Q3 + 1.5 * IQR)).align(df)[0]).any(axis=1)]

# Detect outliers using boxplots
plt.figure(figsize=(9, 6))
sns.boxplot(data=df_no_outliers)
plt.xticks(rotation=90)
plt.title('Boxplot without Outliers')
plt.show()
```

# Output
![image](https://github.com/sabithapaulraj/Mini-Project/assets/118343379/95e7be2f-7672-45e6-9008-21f562e6ec8a)
![image](https://github.com/sabithapaulraj/Mini-Project/assets/118343379/9336d411-94cf-471a-9f3c-32980db22c8f)
![image](https://github.com/sabithapaulraj/Mini-Project/assets/118343379/fad62dcb-99c3-448d-bcb9-c7eecafb0a03)

# Univariate Analysis:
# Univariate analysis for 'Age_band_of_driver'
```python
plt.figure(figsize=(10, 6))
sns.countplot(x='Age_band_of_driver', data=df)
plt.title('Distribution of Age Bands of Drivers')
plt.show()
```
# Output
![image](https://github.com/sabithapaulraj/Mini-Project/assets/118343379/dafc6c24-f8cd-4def-b270-a8db6efc8cde)

# Distribution of Age_band_of_casualty
```python
plt.figure(figsize=(10, 6))
sns.histplot(df['Age_band_of_casualty'], bins=20, kde=True)
plt.title('Distribution of Age_band_of_casualty')
plt.xlabel('Age_band_of_casualty')
plt.ylabel('Count')
plt.show()
```
# Output
![image](https://github.com/sabithapaulraj/Mini-Project/assets/118343379/89139744-4042-4466-92b9-a8afe8d6abee)

# Count of Accidents by Road Surface Type:
```python
plt.figure(figsize=(10, 6))
sns.countplot(x='Road_surface_type', data=df)
plt.title('Count of Accidents by Road Surface Type')
plt.xlabel('Road Surface Type')
plt.ylabel('Count')
plt.show()
```
# Output
![image](https://github.com/sabithapaulraj/Mini-Project/assets/118343379/4f0a0bd7-4e8d-4923-bd0c-81572069570f)

# Multivariate Analysis:
# Accident Severity by Weather Conditions and Light Conditions:
```python
plt.figure(figsize=(12, 8))
sns.countplot(x='Weather_conditions', hue='Light_conditions', data=df, hue_order=df['Light_conditions'].value_counts().index)
plt.title('Accident Severity by Weather and Light Conditions')
plt.xlabel('Weather Conditions')
plt.ylabel('Count')
plt.show()
```
# Output
![image](https://github.com/sabithapaulraj/Mini-Project/assets/118343379/06679bd1-a76c-489f-8275-30419d8ee4ff)

# Vehicle Movement during Accidents by Age Band:
```python
plt.figure(figsize=(14, 8))
sns.countplot(x='Vehicle_movement', data=df, hue='Age_band_of_casualty')
plt.title('Vehicle Movement during Accidents by Age Band')
plt.xlabel('Vehicle Movement')
plt.ylabel('Count')
plt.show()
```
# Output

![image](https://github.com/sabithapaulraj/Mini-Project/assets/118343379/e4a680ae-c55a-44ad-b627-02159eef9d23)

# Multivariate analysis using a heatmap for correlation matrix
```python
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
```
# Output
![image](https://github.com/sabithapaulraj/Mini-Project/assets/118343379/6f166dcd-ca78-4152-8b70-515a34612eb0)

# Feature Encoding and Scaling:
# One-Hot Encoding for categorical variables
```python
# Identify non-numeric columns
non_numeric_cols = df_encoded.select_dtypes(exclude=['float64', 'int64']).columns

# Example: Min-Max Scaling for numerical variables
from sklearn.preprocessing import MinMaxScaler

# Exclude non-numeric columns from scaling
numeric_cols = df_encoded.columns.difference(non_numeric_cols)
df_scaled = df_encoded.copy()
scaler = MinMaxScaler()
df_scaled[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])
print(df_scaled)
```
# Output
![image](https://github.com/sabithapaulraj/Mini-Project/assets/118343379/159de8f7-589b-40e6-bd46-3b9f1130858e)
![image](https://github.com/sabithapaulraj/Mini-Project/assets/118343379/8f68aed4-c6d4-44ef-b035-620dee963889)
![image](https://github.com/sabithapaulraj/Mini-Project/assets/118343379/128bb31b-4e57-446c-865f-71ad3cf0fa89)

# Feature Transformation:
# Log transformation for a skewed feature
```python
import numpy as np

# Convert 'Age_band_of_driver' to numeric, coercing errors to NaN
df['Age_band_of_driver'] = pd.to_numeric(df['Age_band_of_driver'], errors='coerce')

# Apply log transformation
df['Age_band_of_driver_log'] = np.log(df['Age_band_of_driver'] + 1)
# Print the first few rows of the DataFrame after log transformation
print(df[['Age_band_of_driver', 'Age_band_of_driver_log']])
```
# Output
![image](https://github.com/sabithapaulraj/Mini-Project/assets/118343379/fc3cc47e-1dba-4820-b34e-8559fb8a5148)

# Data Visualization and Analysis:
# Visualization of accidents severity
```python
plt.figure(figsize=(8, 6))
sns.countplot(x='Accident_severity', data=df)
plt.title('Distribution of Accident Severity')
plt.show()
```
# Output
![image](https://github.com/sabithapaulraj/Mini-Project/assets/118343379/e00b0bfc-4551-4d56-80a9-999eb7f2fee3)

# Correlation Heatmap:
```python
plt.figure(figsize=(9, 6))
sns.heatmap(df.corr(), cmap='coolwarm', annot=True, fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()
```
# Output
![image](https://github.com/sabithapaulraj/Mini-Project/assets/118343379/b1b261da-0b8f-412f-9468-0dac885388a2)

# Accident Severity by Day of the Week:
```python
plt.figure(figsize=(10, 6))
sns.countplot(x='Day_of_week', hue='Accident_severity', data=df)
plt.title('Accident Severity by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Count')
plt.show()
```
# Output
![image](https://github.com/sabithapaulraj/Mini-Project/assets/118343379/933681ea-5468-4a07-8d33-596a89f6a437)

# Casualty Severity by Gender:
```python
plt.figure(figsize=(8, 6))
sns.countplot(x='Sex_of_casualty', hue='Casualty_severity', data=df)
plt.title('Casualty Severity by Gender')
plt.xlabel('Gender of Casualty')
plt.ylabel('Count')
plt.show()
```
# Output
![image](https://github.com/sabithapaulraj/Mini-Project/assets/118343379/63a57eda-6765-48ba-b538-bc0dc594e1c3)
