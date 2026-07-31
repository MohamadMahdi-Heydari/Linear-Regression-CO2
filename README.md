# Linear-Regression-CO2
Multiple Linear Regression in Python (scikit-learn) to predict CO₂ emissions from fuel consumption, engine size, and cylinders, with R² evaluation, prediction plot, and correlation heatmap.


#  CO₂ Emission Prediction using Multiple Linear Regression

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Linear%20Regression-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

##  Overview

This project predicts **vehicle CO₂ emissions** using **Multiple Linear Regression**.

The model is trained using three vehicle features:

- Fuel Consumption (`fuelcomb`)
- Engine Size (`engine`)
- Number of Cylinders (`cylandr`)

The target variable is:

- CO₂ Emissions (`out1`)

The project demonstrates the complete workflow of a machine learning regression problem using Python and Scikit-Learn.

---

# 📷 Project Preview

## Correlation Heatmap

![Heatmap]([images/heatmap.png](https://github.com/MohamadMahdi-Heydari/Linear-Regression-CO2/blob/main/Figure_1.png))

---

## Model Prediction

![Prediction]([images/prediction.png](https://github.com/MohamadMahdi-Heydari/Linear-Regression-CO2/blob/main/Heatmap.png))

---

#  Dataset

The dataset contains information about different vehicles.

| Feature | Description |
|----------|-------------|
| fuelcomb | Combined fuel consumption |
| engine | Engine size (Liters) |
| cylandr | Number of cylinders |
| out1 | CO₂ emissions (g/km) |

---

# ⚙️ Libraries Used

- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

Install them using:

```bash
pip install -r requirements.txt
```

---

#  How the Code Works

## 1. Import Libraries

The required Python libraries are imported.

```python
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
```

---

## 2. Load Dataset

The CSV file is loaded into a Pandas DataFrame.

```python
df = pd.read_csv("co2.csv")
```

---

## 3. Select Features

Three input variables are selected.

```python
x = df[['fuelcomb', 'engine', 'cylandr']]
```

The target variable:

```python
y = df['out1']
```

---

## 4. Split the Dataset

The data is divided into training and testing sets.

```python
train_test_split(...)
```

80% is used for training.

20% is used for testing.

---

## 5. Train the Model

A Multiple Linear Regression model is created.

```python
model = linear_model.LinearRegression()
model.fit(x_train, y_train)
```

The model learns the coefficients:

\[
CO₂ =
B_0
+
B_1(FuelConsumption)
+
B_2(EngineSize)
+
B_3(Cylinders)
\]

where:

- **B₀** is the intercept.
- **B₁, B₂, B₃** are learned from the training data using the Least Squares method.

---

## 6. Prediction

The model predicts CO₂ emissions for unseen vehicles.

Example:

```python
p206 = np.array([[7.24, 1.7, 4]])
```

Prediction:

```python
model.predict(p206)
```

---

## 7. Model Parameters

The learned coefficients are displayed.

```python
print(model.intercept_)
print(model.coef_)
```

---

## 8. Evaluation

The model can be evaluated using:

- MAE
- MSE
- RMSE
- R² Score

Example:

```python
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
```

---

#  Correlation Analysis

Before training, the relationships between variables can be visualized using a Seaborn heatmap.

```python
sns.heatmap(df.corr(), annot=True)
```

This helps identify which features have the strongest relationship with CO₂ emissions.

---

#  Results

The model predicts vehicle CO₂ emissions based on:

- Fuel Consumption
- Engine Size
- Number of Cylinders

Multiple Linear Regression performs well because these variables have strong correlations with CO₂ emissions.

---

#  Project Structure

```
Linear-Regression-CO2/
│
├── co2.csv
├── main.py
├── requirements.txt
├── README.md
└── images
    ├── heatmap.png
    └── prediction.png
```

---

#  Future Improvements

- Polynomial Regression
- Feature Scaling
- Cross Validation
- Hyperparameter Optimization
- Manual Linear Regression (without Scikit-Learn)

---

#  Author

**Mohamad Mahdi Heydari**

GitHub:

https://github.com/MohamadMahdi-Heydari

---

#  If you found this project useful, consider giving it a Star!
