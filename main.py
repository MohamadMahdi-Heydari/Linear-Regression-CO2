import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("co2.csv")

x = df[['fuelcomb', 'engine', 'cylandr']]
y = df['out1']

x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.2,  random_state=42)

model = linear_model.LinearRegression()
model.fit(x_train,y_train)

y_test_pred = model.predict(x_test)

p206 = np.array([[7.24, 1.7, 4]])
co2 = model.predict(p206)
print(co2[0])

# r^2 
r2 = r2_score(y_test,y_test_pred)
print(r2)

# plot show
plt.scatter(y_test, y_test_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r')
plt.xlabel("Actual (y_test)")
plt.ylabel("Predicted (y_pred)")
plt.title("Predicted vs Actual")
plt.show()


# heatmap
corr = df[['fuelcomb', 'engine', 'cylandr', 'out1']].corr()

plt.figure(figsize=(6,5))

sns.heatmap(corr,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5)

plt.show()