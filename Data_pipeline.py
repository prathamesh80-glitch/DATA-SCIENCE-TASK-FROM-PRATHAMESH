import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load Data
data = {
    'Name': ['A', 'B', 'C', 'D', 'E'],
    'Age': [25, 30, None, 35, 40],
    'Salary': [50000, 60000, 55000, None, 65000],
    'Department': ['HR', 'IT', 'HR', 'Finance', 'IT']
}

df = pd.DataFrame(data)

# Handle Missing Values
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Salary'].fillna(df['Salary'].mean(), inplace=True)

# Encode Categorical Data
le = LabelEncoder()
df['Department'] = le.fit_transform(df['Department'])

# Feature Scaling
scaler = StandardScaler()
df[['Age', 'Salary']] = scaler.fit_transform(df[['Age', 'Salary']])

# Save Data
df.to_csv('processed_data.csv', index=False)

print(df)
