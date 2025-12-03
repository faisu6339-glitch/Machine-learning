import pandas as pd
from sklearn.linear_model import LinearRegression
df=pd.read_csv('rent.csv')
area=df[['area']]
rent=df['rent']
print('Area',area)
print('Rent',rent)
model=LinearRegression()
model.fit(area,rent)
inputarea=int(input('Enter area:'))
rent=model.predict([[inputarea]])
print("Price:",rent[0])
