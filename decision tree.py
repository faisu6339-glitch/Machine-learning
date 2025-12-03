import pandas as pd
from sklearn import tree
df = pd.read_csv("C:/Users/Hp/Documents/cc13_2.csv")

df_encoded = df.apply(lambda col: col.astype('category').cat.codes)

model = tree.DecisionTreeClassifier()

X = df_encoded.drop('rain',axis=1)
y = df_encoded['rain']

sky = input('enter sky').upper()
humidity = input('enter humidity').upper()
temperature = input('enter temperature').upper()

feature = []

if sky == 'CLEAN':
    feature.append(0)
elif sky == 'CLOUD':
    feature.append(1)
else:
    feature.append(0)

if humidity == 'NO':
    feature.append(0)
elif humidity == 'YES':
    feature.append(1)
else:
    feature.append(0)

if temperature == 'MEDIUM':
    feature.append(2)
elif temperature == 'HIGH':
    feature.append(0)
elif temperature == 'LOW':
    feature.append(1)
else:
    feature.append(1)


model.fit(X,y)
pred = model.predict([feature])

if pred == 1:
    print("It will rain.")
else:
    print('IT will not rain')


