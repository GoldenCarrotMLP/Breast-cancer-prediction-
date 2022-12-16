import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
colnames = ['CodeNumber','Thickness', 'Size', 'Shape', 'Adhesion', 'Epithetialsize', 'nuclei', 'chromatin', 'nucleoi',
'mitoses', 'Class']


df = pd.read_csv('pre-Midterm/breast-cancer-wisconsin.data', names=colnames)

print('Dataframe CodeNumber')
print(df.CodeNumber)
print('Dataframe Thickness')
print(df.Thickness)
print('Dataframe Description')
print(df.describe())
print('Dataframe Info')
print(df.info)
print('Dataframe Head')
print(df.head())

#----------------------- Pre-processing------------------------------------------------------------------------


df['nuclei'] = (df['nuclei'] == '?').astype(int)

X = df.values[:, 0:9]
y = df.values[:, 10]

print('Any columns with null values')
print(df.isna().sum())
print(df.dtypes)
print('Column \'Class\' Unique Values: ', df['nuclei'].unique())
print('Column \'Class\' Values Count:')
print(df['nuclei'].value_counts())

print('X')
print(X)
print('Data (X) Size:',X.shape)

print('Label (y) Size:',y.shape)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
    

#------------------------------ Model: kNN------------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('y_pred: ', y_pred)
print('y_test')
print(y_test)

from sklearn.metrics import classification_report
print('Classification Report')
print(classification_report(y_test, y_pred))

tes = np.array([-0.01575291 -1.2146669  -0.69999505 -0.06984988 -0.28278021 -0.54956136
 -0.15305575 -0.17966213 -0.61182504], dtype=np.int32).reshape(1,-1)
print(str(model.predict(tes)).replace("2", "benign").replace("4","malignant"))
