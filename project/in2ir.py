from cProfile import label
from turtle import color
import pandas as pb
import matplotlib.pyplot as m
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
dat=pb.read_excel(r"C:\Users\user\Desktop\7th\project2022\smp.xlsx", names=['year','articial'])
y=np.array(dat.articial)
x=np.array(dat.year)
x1,x2,y1,y2= train_test_split(x,y ,test_size=0.33,shuffle=True)
lr=LinearRegression()
X1=(np.array(x2)).reshape(-1,1)
lr.fit(X1,y2)
pre=lr.predict(np.array(x1).reshape(-1,1))
z=list(map(int,input().split()))
z=np.array(z).reshape(-1,1)
print(lr.predict(z))
m.plot(x1,pre,label='linear regression',color='b')
m.scatter(x1,y1,label='Actual data',color='g')
m.scatter(x2,y2,label='training data',color='r')
m.legend()
m.show()

