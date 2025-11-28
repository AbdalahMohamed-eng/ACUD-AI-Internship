import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set page configuration
st.set_page_config(page_title="Smart Lighting Prediction App", layout="wide")

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("smart_lighting_dataset_2024.csv")
    numerical_features = ['ambient_light_lux', 'motion_detected', 'temperature_celsius', 'occupancy_count',
                          'energy_price_per_kwh', 'prev_hour_energy_usage_kwh', 'traffic_density',
                          'avg_pedestrian_speed', 'adjusted_light_intensity']
    categorical_features = ['day_of_week', 'time_of_day', 'weather_condition', 'special_event_flag']
    
    # Encode categorical features
    label_encoders = {}
    for feature in categorical_features:
        label_encoders[feature] = LabelEncoder()
        df[feature] = label_encoders[feature].fit_transform(df[feature])
    
    return df, numerical_features, categorical_features, label_encoders

# Train models
@st.cache_resource
def train_models(df, numerical_features, categorical_features):
    X = df[numerical_features + categorical_features]
    y_reg = df['energy_consumption_kwh']
    y_clf = df['lighting_action_class']

    # Split data
    X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42)

    # Scale numerical features
    scaler = StandardScaler()
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])

    # Train regression model
    regressor = RandomForestRegressor(random_state=42)
    regressor.fit(X_train, y_train_reg)
    
    # Train classification model
    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X_train_clf, y_train_clf)

    # Evaluate models
    y_pred_reg = regressor.predict(X_test)
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    r2 = r2_score(y_test_reg, y_pred_reg)
    y_pred_clf = classifier.predict(X_test_clf)
    accuracy = accuracy_score(y_test_clf, y_pred_clf)
    clf_report = classification_report(y_test_clf, y_pred_clf, output_dict=True)
    cm = confusion_matrix(y_test_clf, y_pred_clf)

    return regressor, classifier, scaler, X_test, y_test_reg, y_test_clf, y_pred_reg, y_pred_clf, mse, r2, accuracy, clf_report, cm

# Load data and train models
df, numerical_features, categorical_features, label_encoders = load_data()
regressor, classifier, scaler, X_test, y_test_reg, y_test_clf, y_pred_reg, y_pred_clf, mse, r2, accuracy, clf_report, cm = train_models(df, numerical_features, categorical_features)

# Streamlit UI
st.title("Smart Lighting Prediction App")
st.write("This app predicts energy consumption and lighting action class based on input features.")

# Model evaluation section
st.header("Model Performance Metrics")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Regression Metrics")
    st.write(f"Mean Squared Error: {mse:.3f}")
    st.write(f"R2 Score: {r2:.3f}")

with col2:
    st.subheader("Classification Metrics")
    st.write(f"Accuracy: {accuracy:.3f}")
    st.write("Classification Report:")
    clf_report_df = pd.DataFrame(clf_report).transpose()
    st.dataframe(clf_report_df)

# Confusion matrix visualization
st.subheader("Confusion Matrix")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted Class')
ax.set_ylabel('Actual Class')
ax.set_title('Confusion Matrix for Classification')
st.pyplot(fig)

# User input section
st.header("Make a Prediction")
st.write("Enter the feature values below to predict energy consumption and lighting action class.")

user_input = {}
with st.form("prediction_form"):
    col3, col4 = st.columns(2)

    with col3:
        for feature in numerical_features:
            user_input[feature] = st.number_input(f"{feature.replace('_', ' ').title()}", value=0.0, step=0.1)

    with col4:
        user_input['day_of_week'] = st.selectbox("Day of Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        user_input['time_of_day'] = st.selectbox("Time of Day", ['Morning', 'Afternoon', 'Evening', 'Night'])
        user_input['weather_condition'] = st.selectbox("Weather Condition", ['Clear', 'Cloudy', 'Rainy', 'Foggy'])
        user_input['special_event_flag'] = st.selectbox("Special Event Flag", ['0', '1'])

    submitted = st.form_submit_button("Predict")

# Process predictions
if submitted:
    # Create DataFrame from user input
    input_df = pd.DataFrame([user_input])

    # Encode categorical features
    for feature in categorical_features:
        input_df[feature] = label_encoders[feature].transform(input_df[feature])

    # Combine features
    X_input = input_df[numerical_features + categorical_features]

    # Scale numerical features
    X_input[numerical_features] = scaler.transform(X_input[numerical_features])

    # Make predictions
    regression_prediction = regressor.predict(X_input)[0]
    classification_prediction = classifier.predict(X_input)[0]

    # Display predictions
    st.header("Predictions")
    st.write(f"**Predicted Energy Consumption (kWh):** {regression_prediction:.3f}")
    st.write(f"**Predicted Lighting Action Class:** {classification_prediction}")