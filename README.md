# ACUD Smart City AI Solutions 🏙️⚡🤖

**Role:** AI Engineer Intern  
**Organization:** Administrative Capital for Urban Development (ACUD)  
**Location:** Egypt (New Administrative Capital)

## 📖 Overview
This repository contains two major **AI projects** developed during my internship at ACUD. These projects align with ACUD's vision of creating a fully digital, sustainable, and safe smart city.

The solutions focus on **infrastructure optimization (Smart Lighting)** and **security (Face Recognition)** using Machine Learning and Computer Vision techniques.

---

## 🏗️ Project 1: AI-Powered Smart Lighting & Energy Management

### 🎯 Objective
To optimize energy consumption in smart cities by predicting energy usage and automating lighting controls based on environmental factors.

### 🧠 Technical Approach
* **Data Inputs:** Ambient light (Lux), Motion detection, Temperature, Traffic density, Pedestrian speed, and Time of day.
* **Models Used:** * `RandomForestRegressor`: To predict energy consumption (kWh).
    * `RandomForestClassifier`: To classify the required lighting action (Dim, Off, High Intensity).
* **Interface:** A Streamlit web application for real-time inference and parameter adjustment.

### 🚀 Key Features
* **Dynamic Adaptation:** Streetlights adjust based on live traffic and weather conditions.
* **Energy Efficiency:** significantly reduces waste by dimming lights in low-activity zones (e.g., parks after midnight).
* **Decision Support:** Provides analytics for city planners regarding energy tariffs and usage patterns.

---

## 👁️ Project 2: Real-Time Face Recognition Security System

### 🎯 Objective
To enhance security in government buildings and residential areas within the New Capital by implementing a robust access control system.

### 🧠 Technical Approach
* **Pipeline:** 
    1.  **Face Detection:** Uses `MTCNN` and `Haar Cascades` for accurate face localization.
    2.  **Feature Extraction:** Uses `FaceNet (Inception ResNet v1)` to generate 128-D embeddings.
    3.  **Classification:** Uses `SVM` (Support Vector Machine) and `Random Forest` for identity verification.
* **Live Detection:** Integrated with `YOLO` for object detection and `Streamlit WebRTC` for live webcam processing.

### 🚀 Key Features
* **Real-Time Processing:** Detects and recognizes faces instantly via webcam feed.
* **High Accuracy:** Robust against changes in lighting and facial angles due to FaceNet embeddings.
* **User Interface:** Interactive dashboard to upload images or stream video for verification.

---

## 🛠️ Technologies Used
* **Languages:** Python 3.x
* **Machine Learning:** Scikit-Learn, TensorFlow/Keras
* **Computer Vision:** OpenCV, MTCNN, FaceNet, YOLO
* **Web Framework:** Streamlit, Streamlit-WebRTC
* **Data Processing:** Pandas, NumPy

## 💻 How to Run
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AbdalahMohamed-eng/ACUD-AI-Internship.git
    ```

2.  **Run Smart Lighting App:**
    ```bash
    cd Smart Lighting & Energy Management
    streamlit run Streamlit.py
    ```

3.  **Run Face Recognition App:**
    ```bash
    cd Face Recognition System
    streamlit run Streamlit.py
    ```

## 📈 Impact
These projects contribute to the "Smart City" initiative by ensuring:
* **Sustainability:** Reduced carbon footprint through intelligent energy usage.
* **Safety:** Enhanced surveillance and secure access control for critical infrastructure.

---
*Developed by Abdalah Saad as part of the ACUD Internship Program.*