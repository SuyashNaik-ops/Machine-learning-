import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"C:\Users\PRAJWAL\Downloads\archive (9)\Titanic-Dataset.csv")

print("Original Dataset")

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

print("\nCategorical Columns")
print(df.select_dtypes(include=['object', 'string']).columns)

print("\nNumerical Columns")
print(df.select_dtypes(include=['int64', 'float64']).columns)

print("\n----- Handling Missing Values -----")

age_imputer = SimpleImputer(strategy='mean')
df['Age'] = age_imputer.fit_transform(df[['Age']])

embarked_imputer = SimpleImputer(strategy='most_frequent')
df['Embarked'] = embarked_imputer.fit_transform(df[['Embarked']]).ravel()

df.drop('Cabin', axis=1, inplace=True)


print("Missing values handled successfully.")

print("\nMissing Values After Handling")
print(df.isnull().sum())

print("\n----- Encoding Categorical Variables -----")

df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

print("Sex and Embarked columns encoded successfully.")

print("\n----- Scaling Numerical Features -----")

numerical_columns = ['Age', 'Fare', 'SibSp', 'Parch']

scaler = StandardScaler()

df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

print("Scaled Columns:", numerical_columns)

print("\nFinal Model-Ready Dataset")
print(df.head())

df.to_csv(r"C:\Users\PRAJWAL\Downloads\archive (9)\Titanic_Cleaned.csv", index=False)

print("\nTitanic_Cleaned.csv saved successfully.")