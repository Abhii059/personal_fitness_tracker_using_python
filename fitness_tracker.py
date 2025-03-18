import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import time

import warnings
warnings.filterwarnings('ignore')

st.title("Comprehensive Fitness Tracker")
st.markdown("This WebApp allows you to predict the calories burned by entering parameters such as `Age`, `Gender`, `BMI`, and more.")

st.sidebar.header("Enter Your Parameters: ")

def get_user_inputs():
    age = st.sidebar.slider("Age (years): ", 10, 100, 30)
    weight = st.sidebar.slider("Weight (kg): ", 30.0, 150.0, 70.0)
    height = st.sidebar.slider("Height (cm): ", 120, 220, 170)
    bmi = weight / ((height / 100) ** 2)
    duration = st.sidebar.slider("Exercise Duration (minutes): ", 0, 120, 30)
    heart_rate = st.sidebar.slider("Heart Rate (bpm): ", 60, 200, 100)
    body_temp = st.sidebar.slider("Body Temperature (°C): ", 36.0, 42.0, 38.0)
    steps_count = st.sidebar.number_input("Steps Count: ", min_value=0, value=5000)
    sleep_duration = st.sidebar.number_input("Sleep Duration (hours): ", min_value=0.0, max_value=24.0, value=7.0)
    water_intake = st.sidebar.number_input("Water Intake (liters): ", min_value=0.0, max_value=10.0, value=2.0)
    exercise_type = st.sidebar.selectbox("Exercise Type: ", ["Cardio", "Strength Training", "Mixed"])
    gender_option = st.sidebar.radio("Gender: ", ["Male", "Female"])

    gender = 1 if gender_option == "Male" else 0

    # Create a dictionary for the input data
    user_data = {
        "Age": age,
        "Weight": weight,
        "Height": height,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Steps_Count": steps_count,
        "Sleep_Duration": sleep_duration,
        "Water_Intake": water_intake,
        "Exercise_Type_Cardio": 1 if exercise_type == "Cardio" else 0,
        "Exercise_Type_Strength": 1 if exercise_type == "Strength Training" else 0,
        "Gender_male": gender  # Encode gender as 1 for male, 0 for female
    }

    input_df = pd.DataFrame(user_data, index=[0])
    return input_df

# Fetch user input
df = get_user_inputs()

# Display user input
st.write("---")
st.header("Input Parameters")
progress_message = st.empty()
progress_bar = st.progress(0)
for i in range(100):
    progress_bar.progress(i + 1)
    time.sleep(0.01)
st.write(df)

# Load and preprocess the data
calories_data = pd.read_csv("calories1.csv")
exercise_data = pd.read_csv("exercises1.csv")

merged_data = exercise_data.merge(calories_data, on="User_ID")
merged_data.drop(columns="User_ID", inplace=True)

train_data, test_data = train_test_split(merged_data, test_size=0.2, random_state=1)

# Compute BMI for training and testing data
for dataset in [train_data, test_data]:
    dataset["BMI"] = dataset["Weight"] / ((dataset["Height"] / 100) ** 2)
    dataset["BMI"] = round(dataset["BMI"], 2)

# Prepare data for model training and testing
train_data = train_data[["Gender", "Age", "Weight", "Height", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Steps_Count", "Sleep_Duration", "Water_Intake", "Exercise_Type", "Calories"]]
test_data = test_data[["Gender", "Age", "Weight", "Height", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Steps_Count", "Sleep_Duration", "Water_Intake", "Exercise_Type", "Calories"]]
train_data = pd.get_dummies(train_data, drop_first=True)
test_data = pd.get_dummies(test_data, drop_first=True)

# Separate features and labels
X_train = train_data.drop("Calories", axis=1)
y_train = train_data["Calories"]

X_test = test_data.drop("Calories", axis=1)
y_test = test_data["Calories"]

# Train the model
model = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
model.fit(X_train, y_train)

# Align prediction data columns with training data
df = df.reindex(columns=X_train.columns, fill_value=0)

# Make prediction
predicted_calories = model.predict(df)

# Display prediction result
st.write("---")
st.header("Predicted Calories Burned")
progress_message = st.empty()
progress_bar = st.progress(0)
for i in range(100):
    progress_bar.progress(i + 1)
    time.sleep(0.01)

st.write(f"{round(predicted_calories[0], 2)} kilocalories")

# Display similar results
st.write("---")
st.header("Similar Results")
progress_message = st.empty()
progress_bar = st.progress(0)
for i in range(100):
    progress_bar.progress(i + 1)
    time.sleep(0.01)

# Find similar results based on predicted calories
calorie_range = [predicted_calories[0] - 10, predicted_calories[0] + 10]
similar_results = merged_data[(merged_data["Calories"] >= calorie_range[0]) & (merged_data["Calories"] <= calorie_range[1])]
st.write(similar_results.sample(5))

# Display general information and insights
st.write("---")
st.header("General Information and Insights")

# Compare user's input with the dataset
age_comparison = (merged_data["Age"] < df["Age"].values[0]).tolist()
duration_comparison = (merged_data["Duration"] < df["Duration"].values[0]).tolist()
temp_comparison = (merged_data["Body_Temp"] < df["Body_Temp"].values[0]).tolist()
hr_comparison = (merged_data["Heart_Rate"] < df["Heart_Rate"].values[0]).tolist()

st.write(f"You are older than {round(sum(age_comparison) / len(age_comparison), 2) * 100}% of other individuals.")
st.write(f"Your exercise duration is longer than {round(sum(duration_comparison) / len(duration_comparison), 2) * 100}% of others.")
st.write(f"Your heart rate is higher than {round(sum(hr_comparison) / len(hr_comparison), 2) * 100}% of others during exercise.")
st.write(f"Your body temperature is higher than {round(sum(temp_comparison) / len(temp_comparison), 2) * 100}% of others during exercise.")
