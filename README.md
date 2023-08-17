# Flight-Fare-Prediction

This project involves predicting the fare of a flight based on various attributes using machine learning techniques. Flight ticket prices can be unpredictable and vary widely, making it challenging for travelers to estimate costs accurately. This project aims to build a machine learning model that can predict flight ticket prices based on attributes such as airline, source, destination, date of journey, duration, and more.

## Table of Contents

- [Overview](#flight-fare-prediction)
- [Data](#data)
- [Features](#features)
- [Steps](#steps)
- [Feature Engineering](#feature-engineering)
- [Feature Selection](#feature-selection)
- [Model Building](#model-building)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results](#results)
- [Getting Started](#getting-started)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [License](#license)

## Data

The dataset used for this project includes information about flights, such as the airline, source, destination, date of journey, duration, total stops, and additional information. The dataset is split into training and testing sets, allowing for model development and evaluation.

## Features

The following features are considered for predicting flight ticket prices:

- Airline
- Source
- Destination
- Date of Journey
- Duration
- Total Stops
- Additional Information

## Steps

1. Data Analysis: Exploring and understanding the dataset's structure and characteristics.
2. Feature Engineering: Preprocessing and transforming data to create meaningful features.
3. Feature Selection: Identifying important features that contribute to predicting flight fares.
4. Model Building: Constructing a machine learning model to predict flight ticket prices.
5. Hyperparameter Tuning: Optimizing the model's hyperparameters to improve performance.
6. Results: Evaluating the model's performance using metrics such as Mean Absolute Error, Mean Squared Error, Root Mean Squared Error, and R-squared.

## Feature Engineering

The data preprocessing steps include:

- Extracting day, month, and year from the date of journey.
- Converting arrival time and departure time to hours and minutes.
- Splitting the route into different columns.
- Handling missing values in the total stops column.
- Label encoding categorical variables.

## Feature Selection

Feature selection techniques are applied to select the most relevant features for the model. Lasso regression and feature importance from an Extra Trees Regressor are used for feature selection.

## Model Building

A RandomForestRegressor is used for predicting flight fares. The model is trained on the training data and evaluated on the test data.

## Hyperparameter Tuning

RandomizedSearchCV is employed to search for optimal hyperparameters for the RandomForestRegressor. The best hyperparameters are used to fine-tune the model's performance.

## Results

The model's performance is evaluated using various metrics, including Mean Absolute Error, Mean Squared Error, Root Mean Squared Error, and R-squared. Visualizations are provided to illustrate the model's predictions and the differences between predicted and actual values.

## Getting Started

To run the code, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies (see Dependencies section below).
3. Run the Jupyter Notebook containing the code.

## Dependencies

Ensure you have the following dependencies installed:

- Python 3.x
- Jupyter Notebook
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

You can install the required packages using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn

Usage
Clone the repository to your local machine:
git clone https://github.com/22m0052-Rajat/Flight-Fare-Prediction.git

Navigate to the project directory:
cd flight-fare-prediction

Install the required dependencies:
pip install numpy pandas matplotlib seaborn scikit-learn

Launch Jupyter Notebook:
Open the "Flight_Fare_Prediction.ipynb" notebook and run the code cells.

