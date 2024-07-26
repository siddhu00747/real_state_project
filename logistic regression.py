from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
iris = datasets.load_iris()
print(list(iris.keys()))
# f=iris.data
f=iris['data'][:,3:]
# print(iris['data'])
# print(iris['data'].shape)
y = (iris.target == 2).astype(int)
# y=iris.target
# print(f)
# # print(iris['target'])
# print(y)

clf=LogisticRegression()
clf.fit(f,y)
ex=clf.predict(([[2.6]]))
print("yhe hai classfire",ex)


f_new=np.linspace(0,3,1000).reshape(-1,1)
# print( f_new)
y_pro= clf.predict_proba(f_new)

# n=np.mean(y)
# a=np.mean(f)
plt.plot(f_new,y_pro[:,1],"g-", label="virginica")
# plt.axhline(y=n,color="r",linestyle="--",label="mean")
# plt.axhline(y=a,color="g",linestyle="--")
plt.show()
