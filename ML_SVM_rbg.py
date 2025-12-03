import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

student_attendance = {
    'hours':[3,4,7,8,6,6.5,6,9,2],
    'attendance_per':[40, 45, 75, 80, 70, 65,50, 75,30],
    'pass':[0, 0, 1, 0, 1, 1, 0, 1, 0]
}

df = pd.DataFrame(student_attendance)

X = df[['hours','attendance_per']]
y = df['pass']

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3,random_state=42)

model = svm.SVC(kernel='rbf')
model.fit(X,y)

h = float(input('enter hours'))
a_per = float(input('enter your attendance'))

feature = []
feature.append(h)
feature.append(a_per)

pred = model.predict([feature])
if pred[0] == 1:
    print('pass')
else:
    print('fail')