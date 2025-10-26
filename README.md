# Flight Price Prediction Application ✈️

## Project Overview
This project predicts flight ticket prices based on multiple features such as airline, source, destination, number of stops, duration, and date of journey. The prediction model is built using **Random Forest Regressor** and a complete preprocessing pipeline for feature extraction and transformation.

## Features
- Airline
- Source (Departure City)
- Destination (Arrival City)
- Total Stops
- Duration of Flight (in minutes)
- Days Left until Departure
- Date of Journey (automatically transformed into day, month, year)
- Departure Time (hour & minute)

## Project Structure
- `flight_price_model.ipynb` - Notebook containing data cleaning, feature engineering, EDA, model building, hyperparameter tuning, and pipeline creation.
- `best_model.pkl` - Saved trained pipeline (preprocessing + Random Forest model) for predictions.
- `requirements.txt` - Python dependencies needed to run the project.
- `app.py` - Streamlit web application for interactive predictions.
- `README.md` - Project documentation and instructions (this file).

## Installation
1. Clone the repository:
```bash
git clone https://github.com/abdelrahman-beep/flight-price-prediction.git
cd flight-price-prediction
