import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("car data.csv")

print("DATASET LOADED SUCCESSFULLY!")
print("Dataset shape:", df.shape)

# Remove unnecessary column
if "Car_Name" in df.columns:
    df = df.drop("Car_Name", axis=1)

# Features and target
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# Categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(exclude=["object"]).columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="passthrough"
)

# Machine Learning model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", model)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
pipeline.fit(X_train, y_train)

# Prediction
y_pred = pipeline.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nMODEL EVALUATION")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 2))

# Example prediction
sample = X_test.iloc[[0]]
predicted_price = pipeline.predict(sample)[0]

print("\nEXAMPLE CAR PRICE PREDICTION")
print("Predicted Selling Price:", round(predicted_price, 2), "lakhs")

# Actual vs predicted graph
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.savefig("car_price_prediction_graph.png")

print("\nGraph saved as car_price_prediction_graph.png")
print("CAR PRICE PREDICTION PROJECT COMPLETED!")
