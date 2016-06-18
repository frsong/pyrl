import numpy as np

x = np.random.normal(size=(2,3,4))
print(x)
print("")
print(x.max(axis=-1, keepdims=True))
print("")
print(x-x.max(axis=-1, keepdims=True))

print("\n\n")

def softmax(x):
    y = np.exp(x)

    return y/y.sum(-1, keepdims=True)

def log_softmax(x):
    y  = x/1
    y -= y.max(axis=-1, keepdims=True)

    return y - np.log(np.exp(y).sum(axis=-1, keepdims=True))

print(log_softmax(x))
print(np.log(softmax(x)))
