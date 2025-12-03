data = {'area':[800,1000,800,1400,1600,1200],
        'age':[20,25,21,26,35,32],
        'incomej':[100000, 250000,75000, 200000,400000,300000],
        'raint':[2200,6000,1800,7000,10000,8000]
        }

import pandas as pd
from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import train_test_split
df = pd.DataFrame(data)

X = df.drop('raint',axis=1)
y = df['raint']

model = LinearRegression()
model.fit(X, y)
area = 700
age = 27
income = 350000
pred_raint = model.predict([[[area,age,income]]])
print(pred_raint[0])