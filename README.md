# Human Joint Orientation Tracking (IMU Sensor-Based)

> Project Date: Summer 2025

A real-time, end-to-end motion tracking system that captures 3D human arm joint orientation using an MPU9250 IMU, Arduino, and Python. This project implements a **Kalman filter** to effectively denoise raw sensor data and minimize the integration drift, significantly improving tracking accuracy.

---

## 1. About The Project

This project was developed to create a complete, end-to-end system for real-time motion capture. The primary challenge with IMU-based tracking is managing the inherent noise from the accelerometer and the significant drift that occurs when integrating gyroscope data over time.

To solve this, a **Kalman filter** was implemented as the critical signal processing step. This filter effectively fuses the data from the accelerometer and gyroscope, correcting for noise and drift. The final, filtered data is then processed using kinematic principles to calculate the 3D orientation, which is then visualized in Python.

## 2. Key Features

* **End-to-End System:** Developed the entire pipeline, from raw sensor integration (MPU9250 with Arduino) to the final 3D visualization (Python).
* **Advanced Signal Processing:** Implemented a Kalman filter to effectively denoise raw accelerometer and gyroscope data.
* **High-Accuracy Tracking:** Significantly improved tracking accuracy by minimizing the drift caused by data integration.
* **Real-Time 3D Visualization:** Applied kinematic principles to visualize the arm's 3D joint orientation in real-time.

## 3. How It Works

1.  **Data Acquisition:** The `MPU9250 IMU` sensor captures 9-axis (accelerometer, gyroscope, magnetometer) raw data.
2.  **Hardware Interface:** An `Arduino` board reads the data from the IMU sensor and transmits it to a computer via serial communication.
3.  **Data Processing:** A `Python` script receives the serial data. The raw data is passed through the **Kalman filter** to compute a stable and accurate 3D orientation.
4.  **Visualization:** The resulting orientation data is used to render a 3D model of the human arm, updating its position in real-time.

## 4. Tech Stack

* **Hardware:**
    * MPU9250 (Inertial Measurement Unit)
    * Arduino
* **Software & Libraries:**
    * Python
    * (Add any specific Python libraries used, e.g., `PySerial`, `VPython`, `PyOpenGL`)
* **Core Concepts:**
    * Kalman Filter
    * Signal Processing
    * Sensor Fusion
    * Kinematics
