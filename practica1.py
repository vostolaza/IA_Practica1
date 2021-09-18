import numpy as np
import matplotlib.pyplot as plt

num = 1000
epochs = 3500
k = 10
alfa = 0.001

v = []
for i in range(k):
    v.append(0)

x = np.linspace(0, 1, num=num)
y = [np.sin(i*2*3.14) + np.random.normal(0, 0.1) for i in x]
plt.plot(x, y, '*')
# plt.show()

def h(x, w, b):
    sum = 0
    for i in range(k):
        sum += (x ** i) * w[i]
    return sum + b

def derivada(y, x, b, w):
    db = 0
    dw = np.array([0 for i in range(k)], dtype=float)
    m = len(x)
    for i in range(m):
        db += (y[i]-h(x[i], w, b)) * (-1)
    db /= m
    for j in range(k):
        for i in range(m):
            dw[j] += (y[i]-h(x[i], w, b)) * (- (x[i] ** j))
        dw[j] /= m
    return dw, db

def calc_error(y, x, b, w):
    m = len(x)
    e = 0
    for i in range(m):
        e += (y[i]-h(x[i], w, b)) ** 2
    return e/(2*m)

def calc_error_reg(l, w, x):
    sum = 0
    for i in range(k):
        sum += w[i] ** 2
    return (l*sum)/k

def update(w, b, dw, db):
    w -= dw*alfa
    b -= db*alfa
    return w, b

def train(x, y, x_val):
    b = np.random.rand()
    w = np.random.random(k)
    l_error = []
    l_error_val = []
    l = 100
    for e in range(epochs):
        dw, db = derivada(y, x, b, w)
        #w, b = update(w, b, dw, db)
        error = calc_error(y, x, b, w) #+ calc_error_reg(l, w, x)
        l_error.append(error)
        l_error_val.append(calc_error(y, x_val, b, w))

        for i in range(k):
            v[i] = 0.9*v[i] + alfa*dw[i]
            w[i] = w[i] - v[i]

        #print(error)
    return w, b, l_error, error, l_error_val


splits = np.split(x, [600, 800])
y_splits = np.split(y, [600, 800])

wT, bT, l_t_error, trainingError, l_val_error = train(x, y, x)
# wV, bV, l__v_error, validationError = validate(splits[1], y_splits[1])
#wTest, bTest, l__test_error, testError = train(splits[2], y_splits[1])

Ytrain = [h(x[i], wT, bT) for i in range(len(x))]
#Yval = [h(x[i], wV, bV) for i in range(len(x))]
#Ytest = [h(x[i], wTest, bTest) for i in range(len(x))]
plt.plot(x, Ytrain, '*')


##plt.plot([i for i in range(epochs)], l_t_error, '*')
#plt.plot([i for i in range(epochs)], l_val_error, '*')

#plt.plot(x, Yval, '*')
#plt.plot(x, Ytest, '*')
plt.show()