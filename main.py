import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load dataset
data = pd.read_csv("Agricultural_Crop_Yield.csv")

# Convert categorical columns into numeric using one-hot encoding
data = pd.get_dummies(data, columns=['Crop', 'Season', 'State'])

# Select features (remove target)
X = data.drop(['Yield', 'Production'], axis=1)
y = data['Yield']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

import matplotlib.pyplot as plt

# Feature importance
importances = model.feature_importances_

# Create dataframe
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
})

# Sort
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance_df.head(10))
import matplotlib.pyplot as plt

# Sort feature importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Take top 10
top_features = feature_importance_df.head(10)

# Plot
plt.figure()
plt.barh(top_features['Feature'], top_features['Importance'])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Feature Importance for Crop Yield Prediction")
plt.gca().invert_yaxis()
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt

# Select only numerical columns
numeric_data = data[['Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Yield']]

# Compute correlation
correlation = numeric_data.corr()

# Plot heatmap
plt.figure()
sns.heatmap(correlation, annot=True)
plt.title("Correlation Heatmap")
plt.show()
print("\n---- Crop Yield Prediction ----")

# Example manual input
crop_year = int(input("Enter Crop Year: "))
area = float(input("Enter Area: "))
rainfall = float(input("Enter Annual Rainfall: "))
fertilizer = float(input("Enter Fertilizer amount: "))
pesticide = float(input("Enter Pesticide amount: "))

# Create input dataframe (without categorical features for now)
input_data = pd.DataFrame({
    'Crop_Year': [crop_year],
    'Area': [area],
    'Annual_Rainfall': [rainfall],
    'Fertilizer': [fertilizer],
    'Pesticide': [pesticide]
})

# IMPORTANT:
# Because we used one-hot encoding earlier,
# prediction input must match training columns.

# Recreate training structure
input_data = pd.get_dummies(input_data)

# Align columns
input_data = input_data.reindex(columns=X.columns, fill_value=0)

# Predict
predicted_yield = model.predict(input_data)

print("Predicted Yield:", predicted_yield[0])

import joblib

# Save model
joblib.dump(model, "crop_model.pkl")

# Save training column names
joblib.dump(X.columns, "model_columns.pkl")