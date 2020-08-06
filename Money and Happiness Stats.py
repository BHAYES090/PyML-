import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model
import numpy as np

A = pd.read_csv(r'C:\Users\bohayes\AppData\Local\Programs\Python\Python38\LifeSatandGDP.csv')
df = pd.DataFrame(A)
print(df)

X = np.c_[df['GDP_Per_Capita']]
y = np.c_[df['Life_Sat']]

plt.scatter(X, y)

model = sklearn.linear_model.LinearRegression()
model.fit(X, y)
plt.show()
