#!/usr/bin/env python
# coding: utf-8

# In[7]:


#Importing the Data

import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("/Users/tanmaydhiman/ml_project/ml_project")

# Convert date column to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Check for missing values
print(df.isnull().sum())

#Data Analysis

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 5))
sns.lineplot(x="Date", y="Tickets_Booked", data=df, hue="Destination")
plt.title("Ticket Bookings Over Time")
plt.xticks(rotation=45)
plt.show()

#Converting languistic values to boolean values

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["Weekday"] = df["Date"].dt.weekday
df = df.drop(columns=["Date"])
df["Is_Weekend"] = df["Weekday"].apply(lambda x: 1 if x >= 5 else 0)
df["Is_Festival_Week"] = df["Festival_Impact"]  # Already encoded


# One-hot encode categorical features
df = pd.get_dummies(df, columns=["Destination", "Festival_Name"])

# Normalize Tickets Booked (Min-Max Scaling)
df["Tickets_Booked_Norm"] = (df["Tickets_Booked"] - df["Tickets_Booked"].min()) / (df["Tickets_Booked"].max() - df["Tickets_Booked"].min())

# Normalize Weekdays (since range is 1 to 7)
df["Weekdays_Norm"] = (df["Weekday"] - 1) / 6  # Min = 1, Max = 7

# Compute Congestion Score
df["Congestion"] = df["Tickets_Booked_Norm"] + df["Festival_Impact"] + df["Is_Festival_Week"] + df["Weekdays_Norm"]

# Normalize Congestion between 0 and 1
df["Congestion"] = (df["Congestion"] - df["Congestion"].min()) / (df["Congestion"].max() - df["Congestion"].min())

# Save the updated DataFrame
df.to_csv("mldata_updated1.csv", index=False)
df.head()
del df


# In[8]:


# Load the dataset
df = pd.read_csv("mldata_updated1.csv")
# Display first few rows
df.head()


# In[9]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix , ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import StandardScaler

selected_features = ["Weekday", "Weekdays_Norm", "Festival_Impact", "Tickets_Booked", "Is_Festival_Week"]
X = df[selected_features]
y = df["Congestion"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestRegressor(
    n_estimators=300,  # Number of trees
    max_depth=3,  # Maximum depth of each tree
    random_state=42,
    n_jobs=-1  # Use all CPU cores for faster training
)

# Train the model
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluation metrics
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.2f}")

# Convert continuous congestion values into discrete categories
thresholds = np.percentile(y, [33, 66])  # Create thresholds for categorization

def categorize(values, thresholds):
    return np.digitize(values, bins=thresholds)  # Assigns 0 (Low), 1 (Medium), 2 (High)

# Convert actual and predicted values into categories
y_test_classes = categorize(y_test, thresholds)
y_pred_classes = categorize(y_pred, thresholds)

# Compute Confusion Matrix
cm = confusion_matrix(y_test_classes, y_pred_classes)

# Alternative Confusion Matrix with Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Low", "Medium", "High"], yticklabels=["Low", "Medium", "High"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
# Compute classification metrics
print("\nClassification Report:\n")
print(classification_report(y_test_classes, y_pred_classes, target_names=["Low", "Medium", "High"]))
print(f"Expected features: {X.shape[1]}")


# In[10]:


# Extract Feature Importance
feature_importance = model.feature_importances_
feature_names = X.columns

# Create a DataFrame for feature importance
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Display top 5 important features
print("\nTop 5 Important Features:")
print(importance_df.head(5))

# Plot Feature Importance
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(x=importance_df["Importance"][:10], y=importance_df["Feature"][:10], palette="Blues_r")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Top 5 Important Features")
plt.show()


# In[11]:


#Saving the model
import joblib
joblib.dump(model, "model.pkl")  # Save model with joblib


# In[12]:


print(list(X.columns))


# In[ ]:


import joblib
from flask import Flask, request, jsonify
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for Android requests

# Load the trained model
try:
    model = joblib.load("model.pkl")  # Use joblib instead of pickle
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.json
        features = np.array(data["features"]).reshape(1, -1)

        # Ensure correct number of features
        expected_features = 5  # Based on your model
        if features.shape[1] != expected_features:
            return jsonify({"error": f"Expected {expected_features} features, but got {features.shape[1]}"}), 400

        # Make prediction
        prediction = model.predict(features)[0]  

        # Convert continuous value to categorical congestion level
        if prediction < 0.33:
            congestion_level = "Low"
        elif prediction < 0.66:
            congestion_level = "Medium"
        else:
            congestion_level = "High"

        return jsonify({
            "prediction": round(prediction, 3),
            "congestion_level": congestion_level
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5010, debug=False)  


# In[ ]:




