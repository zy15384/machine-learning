import numpy as np
from mpl_toolkits import mplot3d
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import metrics
from keras import optimizers
from scipy.stats import multivariate_normal
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import tensorflow
import math
import random

# section 1
# parameter
n = 3
N = 10000
prior = [0.1, 0.2, 0.3, 0.4]
mu0 = np.array([1, 4, 6])
mu1 = np.array([2, -1, 3])
mu2 = np.array([-3, 3, 1])
mu3 = np.array([-2, -1, -3])
sigma0 = np.mat([[ 4.7709,-1.2340,1.9591],
                [ -1.2340,6.2264,1.5383],
                [1.9591,1.5383,3.0027]])
sigma1 = np.mat([[4.82347,-0.31032,-0.51968],
                [ -0.31032,4.67813,-1.39300],
                [ -0.51968,-1.39300,4.49840]])
sigma2 = np.mat([[4.1613791,0.0020674,-0.8451883],
                [0.0020674,3.9438725,-0.5152832],
                [-0.8451883,-0.5152832,3.8947484]])
sigma3 = np.mat([[ 2.8364083,-0.0095313,-0.4147533],
                [-0.0095313,1.3531993,-0.5009474],
                [-0.4147533,-0.5009474,1.8103923]])

choose = np.zeros([N,1])
choose = np.random.rand(N)
x = np.zeros([N,n])
truedec = np.zeros([N,2])

for i in range(N):
    if(choose[i] > 0 and choose[i] <= prior[0]):
        x[i] = np.random.multivariate_normal(mu0, sigma0)
        truedec[i,0] = 0
    if(choose[i] > prior[0] and choose[i] <= prior[1]+prior[0]):
        x[i] = np.random.multivariate_normal(mu1, sigma1)
        truedec[i,0] = 1
    if(choose[i] > prior[1]+prior[0] and choose[i] <=prior[2]+prior[1]+prior[0]):
        x[i] = np.random.multivariate_normal(mu2, sigma2)
        truedec[i,0] = 2
    if(choose[i] > prior[2]+prior[1]+prior[0] and choose[i] <=1):
        x[i] = np.random.multivariate_normal(mu3, sigma3)
        truedec[i,0] = 3

dis = plt.axes(projection='3d')
plt.figure(1)
for i in range(N):
    if(truedec[i,0]==0):
        dis.scatter3D(x[i,0], x[i,1], x[i,2], color='r',marker='d')
    elif(truedec[i,0]==1):
        dis.scatter3D(x[i,0], x[i,1], x[i,2], color='y',marker='o')
    elif(truedec[i,0]==2):
        dis.scatter3D(x[i,0], x[i,1], x[i,2], color='b',marker='^')
    elif(truedec[i,0]==3):
        dis.scatter3D(x[i,0], x[i,1], x[i,2], color='g',marker='.')
plt.show()

# section 2

# eva gaussian PDF
evalgau = np.zeros([N,4])
for i in range(N):
    nor1 = multivariate_normal(mean=mu0, cov=sigma0)
    re1 = nor1.pdf(x[i])*prior[0]
    evalgau[i,0] = re1
    nor2 = multivariate_normal(mean=mu1, cov=sigma1)
    re2 = nor2.pdf(x[i])*prior[1]
    evalgau[i,1] = re2
    nor3 = multivariate_normal(mean=mu2, cov=sigma2)
    re3 = nor3.pdf(x[i])*prior[2]
    evalgau[i,2] = re3
    nor4 = multivariate_normal(mean=mu3, cov=sigma3)
    re4 = nor4.pdf(x[i])*prior[3]
    evalgau[i,3] = re4

# decision
for i in range(N):
    maxd = evalgau[i,0]
    dec = 0
    if(evalgau[i,1] > maxd):
        maxd = evalgau[i,1]
        dec = 1
    if(evalgau[i,2] > maxd):
        maxd = evalgau[i,2]
        dec = 2
    if(evalgau[i,3] > maxd):
        maxd = evalgau[i,3]
        dec = 3
    truedec[i,1] = dec

# calculate P error
e = 0
for i in range(N):
    if(truedec[i,0] != truedec[i,1]):
        e = e+1

print(e/N)

# section 3
model1 = Sequential()
# training 100
xTrain100 = np.zeros([100,n])
xTrainlabel100 = np.zeros([100,2])
choose1 = np.zeros([100,1])
choose1 = np.random.rand(100)
for i in range(100):
    if(choose1[i] > 0 and choose1[i] <= prior[0]):
        xTrain100[i] = np.random.multivariate_normal(mu0, sigma0)
        xTrainlabel100[i,0] = 0
    if(choose1[i] > prior[0] and choose1[i] <= prior[1]+prior[0]):
        xTrain100[i] = np.random.multivariate_normal(mu1, sigma1)
        xTrainlabel100[i,0] = 1
    if(choose1[i] > prior[1]+prior[0] and choose1[i] <=prior[2]+prior[1]+prior[0]):
        xTrain100[i] = np.random.multivariate_normal(mu2, sigma2)
        xTrainlabel100[i,0] = 2
    if(choose1[i] > prior[2]+prior[1]+prior[0] and choose1[i] <=1):
        xTrain100[i] = np.random.multivariate_normal(mu3, sigma3)
        xTrainlabel100[i,0] = 3
one_hot1 = to_categorical(xTrainlabel100.T[0])

a100 = np.zeros([10,1])
for p in range(10):
    print(p+1)
    model1.add(Dense(units=p+1, input_shape=(3,)))
    model1.add(Activation('softplus'))
    model1.add(Dense(units=4))
    model1.add(Activation('softmax'))
    model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    acc1 = np.zeros([10,1])
    i = 0
    k1 = KFold(n_splits=10, shuffle=False)
    for train_index , test_index in k1.split(xTrain100):
        train, test = xTrain100[train_index], xTrain100[test_index]
        trainL, testL = one_hot1[train_index], one_hot1[test_index]
        model1.fit(train, trainL, batch_size=1, epochs=30, verbose=0)
        score, acc = model1.evaluate(test, testL, verbose=0)
        acc1[i] = acc
        i = i+1
    a100[p] =  np.sum(acc1)/10

model2 = Sequential()
# training 1000
xTrain1000 = np.zeros([1000,n])
xTrainlabel1000 = np.zeros([1000,2])
choose2 = np.zeros([1000,1])
choose2 = np.random.rand(1000)
for i in range(1000):
    if(choose2[i] > 0 and choose2[i] <= prior[0]):
        xTrain1000[i] = np.random.multivariate_normal(mu0, sigma0)
        xTrainlabel1000[i,0] = 0
    if(choose2[i] > prior[0] and choose2[i] <= prior[1]+prior[0]):
        xTrain1000[i] = np.random.multivariate_normal(mu1, sigma1)
        xTrainlabel1000[i,0] = 1
    if(choose2[i] > prior[1]+prior[0] and choose2[i] <=prior[2]+prior[1]+prior[0]):
        xTrain1000[i] = np.random.multivariate_normal(mu2, sigma2)
        xTrainlabel1000[i,0] = 2
    if(choose2[i] > prior[2]+prior[1]+prior[0] and choose2[i] <=1):
        xTrain1000[i] = np.random.multivariate_normal(mu3, sigma3)
        xTrainlabel1000[i,0] = 3
one_hot2 = to_categorical(xTrainlabel1000.T[0])

a1000 = np.zeros([10,1])
for p in range(10):
    print(p+1)
    model2.add(Dense(units=p+1, input_shape=(3,)))
    model2.add(Activation('softplus'))
    model2.add(Dense(units=4))
    model2.add(Activation('softmax'))
    model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    acc2 = np.zeros([10,1])
    i = 0
    k1 = KFold(n_splits=10, shuffle=False)
    for train_index , test_index in k1.split(xTrain1000):
        train, test = xTrain1000[train_index], xTrain1000[test_index]
        trainL, testL = one_hot2[train_index], one_hot2[test_index]
        model2.fit(train, trainL, batch_size=10, epochs=30, verbose=0)
        score, acc = model2.evaluate(test, testL, verbose=0)
        acc2[i] = acc
        i = i+1
    a1000[p] =  np.sum(acc2)/10

model3 = Sequential()
# training 10000
xTrain10000 = np.zeros([10000,n])
xTrainlabel10000 = np.zeros([10000,2])
choose3 = np.zeros([10000,1])
choose3 = np.random.rand(10000)
for i in range(10000):
    if(choose3[i] > 0 and choose3[i] <= prior[0]):
        xTrain10000[i] = np.random.multivariate_normal(mu0, sigma0)
        xTrainlabel10000[i,0] = 0
    if(choose3[i] > prior[0] and choose3[i] <= prior[1]+prior[0]):
        xTrain10000[i] = np.random.multivariate_normal(mu1, sigma1)
        xTrainlabel10000[i,0] = 1
    if(choose3[i] > prior[1]+prior[0] and choose3[i] <=prior[2]+prior[1]+prior[0]):
        xTrain10000[i] = np.random.multivariate_normal(mu2, sigma2)
        xTrainlabel10000[i,0] = 2
    if(choose3[i] > prior[2]+prior[1]+prior[0] and choose3[i] <=1):
        xTrain10000[i] = np.random.multivariate_normal(mu3, sigma3)
        xTrainlabel10000[i,0] = 3
one_hot3 = to_categorical(xTrainlabel10000.T[0])

a10000 = np.zeros([10,1])
for p in range(10):
    print(p+1)
    model3.add(Dense(units=p+1, input_shape=(3,)))
    model3.add(Activation('softplus'))
    model3.add(Dense(units=4))
    model3.add(Activation('softmax'))
    model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    acc3 = np.zeros([10,1])
    i = 0
    k1 = KFold(n_splits=10, shuffle=False)
    for train_index , test_index in k1.split(xTrain10000):
        train, test = xTrain10000[train_index], xTrain10000[test_index]
        trainL, testL = one_hot3[train_index], one_hot3[test_index]
        model3.fit(train, trainL, batch_size=100, epochs=30, verbose=0)
        score, acc = model3.evaluate(test, testL, verbose=0)
        acc3[i] = acc
        i = i+1
    a10000[p] =  np.sum(acc3)/10

plt.figure(2)
plt.plot([1,2,3,4,5,6,7,8,9,10], a100[:], 'o', label = 'training 100')
plt.plot([1,2,3,4,5,6,7,8,9,10], a1000[:], 'o', label = 'training 1000')
plt.plot([1,2,3,4,5,6,7,8,9,10], a10000[:], 'o', label = 'training 10000')
plt.title('training set 100-1000-10000')
plt.xlabel('perceptrons')
plt.ylabel('accuracy')
plt.legend()
plt.show()

p1 = np.argmax(a100)
p2 = np.argmax(a1000)
p3 = np.argmax(a10000)
print("model with training set 100: ", p1, "units")
print("model with training set 1000: ", p2, "units")
print("model with training set 10000: ", p3, "units")

one_hot_test = to_categorical(truedec.T[0])

# 100
model3 = Sequential()
model3.add(Dense(units=p1, input_shape=(3,)))
model3.add(Activation('softplus'))
model3.add(Dense(units=4))
model3.add(Activation('softmax'))
model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model3.fit(xTrain100, one_hot1, batch_size=1, epochs=30, verbose=0)
score1, acc1 = model3.evaluate(x, one_hot_test, verbose=0)
print("training set: 100, accuracy: ", acc1)

# 1000
model3 = Sequential()
model3.add(Dense(units=p2, input_shape=(3,)))
model3.add(Activation('softplus'))
model3.add(Dense(units=4))
model3.add(Activation('softmax'))
model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model3.fit(xTrain1000, one_hot2, batch_size=10, epochs=30, verbose=0)
score2, acc2 = model3.evaluate(x, one_hot_test, verbose=0)
print("training set: 1000, accuracy: ", acc2)

# 10000
model3 = Sequential()
model3.add(Dense(units=p3, input_shape=(3,)))
model3.add(Activation('softplus'))
model3.add(Dense(units=4))
model3.add(Activation('softmax'))
model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model3.fit(xTrain10000, one_hot3, batch_size=100, epochs=30, verbose=0)
score3, acc3 = model3.evaluate(x, one_hot_test, verbose=0)
print("training set: 10000, accuracy: ", acc3)