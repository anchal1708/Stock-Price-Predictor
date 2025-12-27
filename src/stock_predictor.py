import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_csv("data/stock_data.csv")

# Drop unnecessary column if exists
if 'Price' in df.columns:
    df.drop(columns=['Price'], inplace=True)

# Select required columns
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

# Convert to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.dropna(inplace=True)

# Features and target
X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Close']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
print("Mean Squared Error:", mean_squared_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))
