from sklearn import datasets
import numpy as np
from matplotlib import pyplot as pl
import random
import math

# here we are loading the iris dataset and then separating the data and the target we have only taken 0 and 1 as our target value
iris = datasets.load_iris()

x = iris.data[:, :2]
y = (iris.target != 0) * 1

# here we are zipping our x and y value and make a list after that we are shuffling the list
z = list(zip(x, y))
random.shuffle(z)

# In this part we are spliting train,validation and test set
train_set = []
val_set = []
test_set = []

for a in z:
    num = random.random()
    if 0 < num <= 0.7:
        train_set.append(a)
    elif 0.7 < num <= 0.85:
        val_set.append(a)
    else:
        test_set.append(a)

# here we are creating the theta array
theta = np.array([])
for a in range(3):
    theta = np.append(theta, random.random())
# print(theta)


# it will done the work of sigmoid


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

#In this part we are training the dataset


lr = 0.01  # initialing learning rate
train_loss = []
iteration = list(range(1000))
for i in range(1000):
    TJ = 0
    for a in train_set:
        # print(a[1])
        x = np.array(a[0])
        x = np.insert(x, 0, 1)
        # print(x)
        Z = np.dot(x, theta)
        h = sigmoid(Z)
        j = (-a[1]*np.log(h)) - ((1-a[1])*np.log(1-h))
        TJ += j
        dv = np.dot(x, (h-a[1]))
        theta = theta - dv * lr
    TJ = TJ / len(train_set)
    train_loss.append(TJ)

pl.plot(iteration, train_loss)
pl.show()

#this part is for validation set
correct = 0
for a in val_set:
    x = np.array(a[0])
    x = np.insert(x, 0, 1)
    Z = np.dot(x, theta)
    h = sigmoid(Z)
    if h >= 0.5:
        h = 1
    else:
        h = 0
    if h == a[1]:
        correct += 1
val_set_accuracy = correct*100/len(val_set)
print("Validation set accuracy for learning rate = {} is {}%".format(lr, val_set_accuracy))

#this part is for test set
correct = 0
for a in test_set:
    x = np.array(a[0])
    x = np.insert(x, 0, 1)
    Z = np.dot(x, theta)
    h = sigmoid(Z)
    if h >= 0.5:
        h = 1
    else:
        h = 0
    if h == a[1]:
        correct += 1
test_set_accuracy = correct*100/len(test_set)
print("Test set accuracy for learning rate = {} is {}%".format(lr, test_set_accuracy))