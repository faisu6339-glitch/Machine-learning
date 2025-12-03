import matplotlib.pyplot as plt

from sklearn import svm
x = [[2,3],[3,4],[1,1],[5,6],[6,7],[4,3],[6,8]]
y = [0,0,0,1,1,0,1]

model = svm.SVC(kernel='linear')
model.fit(x,y)
x1 = int(input("Enter x cordinate : "))
y1 = int(input("Enter the y coordinate : "))
feature =[]
feature.append(x1)
feature.append(y1)
pred = model.predict([feature])
print("class purpose by model : ",pred[0])
for i, point in enumerate(x):
    if y[i] == 0:
        plt.scatter(point[0], point[1],marker ='o')
    else:
        plt.scatter(point[0],point[1], marker='s')

plt.title("Linear Scatter ")
plt.xlabel("x")
plt.ylabel('class')
plt.show()