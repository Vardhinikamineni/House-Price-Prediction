import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

print("======================================")
print("   HOUSE PRICE PREDICTION SYSTEM")
print("======================================")

# Load dataset
data = pd.read_csv("house_data.csv")

print("\nDataset loaded successfully!")

# Input features
X = data[["area", "bedrooms", "bathrooms", "floors", "age"]]

# Target variable
y = data["price"]

# Split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("Model training completed!")

# Test the model
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("-----------------")
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))

# Get new house details
print("\nEnter New House Details")
print("-----------------------")

area = float(input("Enter area in square feet: "))
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = int(input("Enter number of bathrooms: "))
floors = int(input("Enter number of floors: "))
age = int(input("Enter house age in years: "))

# Create data for the new house
new_house = pd.DataFrame({
    "area": [area],
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms],
    "floors": [floors],
    "age": [age]
})

# Predict house price
predicted_price = model.predict(new_house)

print("\n======================================")
print("       HOUSE PRICE PREDICTION")
print("======================================")

print("Predicted House Price: ₹", round(predicted_price[0], 2))

print("\nPrediction completed successfully!")