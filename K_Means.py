import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = {
    'customer_id':[x for x in range(1,11)],
    'age':[19,21,20,23,31,22,35,23,64,30],
    'annual_income':[15,15,16,16,17,17,18,18,19,19],
    'spending_score':[39,81,6,77,40,76,6,94,3,72]
}

df = pd.DataFrame(data) # DataFrame

X = df[['annual_income','spending_score']]

model = KMeans(n_clusters=3,random_state=0)
model.fit(X)

df['cluster'] = model.labels_

print(df)

plt.scatter(X['annual_income'], X['spending_score'], c=df['cluster'])

plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()