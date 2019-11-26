from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import metrics
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
import math
from sklearn.model_selection import KFold

# data
n = 2
Ntrain = 1000
Ntest = 10000
alpha = [0.33, 0.34, 0.33]
meanVec = np.mat([[-18,0,18], [-8,0,8]])
covV = np.mat([[3.2,0],[0,0.6]])
covE1 = np.mat([[1,-1],[1,1]])/math.sqrt(2)
covE2 = np.mat([[1,0],[0,1]])
covE3 = np.mat([[1,-1],[1,1]])/math.sqrt(2)

# training
t = np.zeros([Ntrain,1])
t = np.random.rand(Ntrain)
xTrain = np.zeros([Ntrain,n])

for i in range(Ntrain):
    if(t[i] > 0 and t[i] <= alpha[0]):
        xTrain[i] = np.transpose(np.dot(covE1,np.dot(covV,np.random.randn(2,1)))+meanVec[:,0])
    if(t[i] > alpha[0] and t[i] <= alpha[1]+alpha[0]):
        xTrain[i] = np.transpose(np.dot(covE2,np.dot(covV,np.random.randn(2,1)))+meanVec[:,1])
    if(t[i] > alpha[1]+alpha[0] and t[i] <=1):
        xTrain[i] = np.transpose(np.dot(covE3,np.dot(covV,np.random.randn(2,1)))+meanVec[:,2])

# test
t = np.zeros([Ntest,1])
t = np.random.rand(Ntest)
xTest = np.zeros([Ntest,n])

for i in range(Ntest):
    if(t[i] > 0 and t[i] <= alpha[0]):
        xTest[i] = np.transpose(np.dot(covE1,np.dot(covV,np.random.randn(2,1)))+meanVec[:,0])
    if(t[i] > alpha[0] and t[i] <= alpha[1]+alpha[0]):
        xTest[i] = np.transpose(np.dot(covE2,np.dot(covV,np.random.randn(2,1)))+meanVec[:,1])
    if(t[i] > alpha[1]+alpha[0] and t[i] <=1):
        xTest[i] = np.transpose(np.dot(covE3,np.dot(covV,np.random.randn(2,1)))+meanVec[:,2])

plt.figure(1)
plt.subplot(1,2,1)
plt.plot(xTrain[:,0],xTrain[:,1],'.')
plt.subplot(1,2,2)
plt.plot(xTest[:,0],xTest[:,1],'.')
plt.show()

# sigmoid
sigmoid = np.zeros([10,1])
for p in range(10):
    print(p+1)
    model1 = Sequential()
    model1.add(Dense(units=p+1, input_shape=(1,), activation='sigmoid'))
    model1.add(Dense(units=1))
    model1.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    scesigmoid = np.zeros([10,1])
    i = 0
    k1 = KFold(n_splits=10, shuffle=False)
    for train_index , test_index in k1.split(xTrain):
        train = xTrain[train_index] 
        test = xTrain[test_index]
        model1.fit(train.T[0].T, train.T[1].T, batch_size=10, epochs=25, verbose=0)
        score, acc = model1.evaluate(test.T[0].T, test.T[1].T, verbose=0)
        scesigmoid[i] = score
        i = i+1
    sigmoid[p] = np.sum(scesigmoid)/10
    
# softplus
softplus = np.zeros([10,1])
for p in range(10):
    print(p+1)
    model2 = Sequential()
    model2.add(Dense(units=p+1, input_shape=(1,), activation='softplus'))
    model2.add(Dense(units=1))
    model2.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    scesoftplus = np.zeros([10,1])
    i = 0
    k1 = KFold(n_splits=10, shuffle=False)
    for train_index , test_index in k1.split(xTrain):
        train = xTrain[train_index]
        test = xTrain[test_index]
        model2.fit(train.T[0].T, train.T[1].T, batch_size=10, epochs=25, verbose=0)
        score, acc = model2.evaluate(test.T[0].T, test.T[1].T, verbose=0)
        scesoftplus[i] = score
        i = i+1
    softplus[p] = np.sum(scesoftplus)/10

plt.figure(2)
plt.plot([1,2,3,4,5,6,7,8,9,10], sigmoid[:], 'o', label = 'sigmoid')
plt.plot([1,2,3,4,5,6,7,8,9,10], softplus[:], 'x', label = 'softplus')
plt.xlabel('perceptrons')
plt.ylabel('loss')
plt.legend()
plt.show()

listaa = softplus.tolist()
print(listaa)
listaa = listaa.index(min(listaa))+1
print(listaa)

y = np.zeros([Ntest,1])
model2 = Sequential()
model2.add(Dense(units=listaa, input_shape=(1,)))
model2.add(Activation('softplus'))
model2.add(Dense(units=1))
model2.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model2.fit(xTrain.T[0].T, xTrain.T[1].T, batch_size=1, epochs=35, verbose=0)
y = model2.predict(xTest.T[0].T)

plt.figure(3)
plt.plot(xTest[:,0],xTest[:,1],'.')
plt.plot(xTest[:,0], y[:],'.')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()