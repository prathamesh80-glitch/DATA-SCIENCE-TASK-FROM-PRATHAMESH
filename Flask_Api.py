import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

data = {
    'Area': [1000, 1500, 2000, 2500, 3000],
    'Bedrooms': [2, 3, 3, 4, 5],
    'Price': [3000000, 4500000, 5000000, 6500000, 8000000]
}

df = pd.DataFrame(data)

X = df[['Area', 'Bedrooms']]
y = df['Price']

model = LinearRegression()
model.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved")
