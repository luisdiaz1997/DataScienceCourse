#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf




# In[3]:


data_url = "https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv"


# In[4]:


df = pd.read_csv(data_url, sep=",")
df.head()


# In[5]:


x = df.households.values.reshape(-1, 1)
y = df.total_rooms.values.reshape(-1,1)


# In[6]:


plt.figure(1, figsize=(8, 6))
plt.scatter(x[::10], y[::10])
plt.xlabel('Number of Households')
plt.ylabel('Total Rooms');
plt.show()


# $$h(x) = x w + b$$

# In[7]:


def get_params(shape):
    np.random.seed(7)
    params = {
    'W':tf.Variable(np.random.randn(*shape), dtype = tf.float32),
    'b':tf.Variable(np.zeros((1, shape[1])), dtype = tf.float32)
    }
    return params


# In[8]:


params = get_params([x.shape[1], y.shape[1]])


# In[9]:


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    result = sess.run(params)
    print(result)


# In[10]:


def linear(x, params):
    h = tf.add(tf.matmul(x, params['W']),params['b'])
    return h


# In[11]:


def get_cost(x, y, model, params, lambd = 0):
    h= model(x, params)
    J = tf.math.reduce_mean((h-y)**2) + ((lambd/2)*tf.math.reduce_sum(params['W']**2))
    return J


# In[12]:


def train(inp, out, model, epochs = 100, lr= 1e-3, lambd = 0):
    x = tf.placeholder(tf.float32, shape = (None, inp.shape[1]))
    y = tf.placeholder(tf.float32, shape = (None, out.shape[1]))
    params = get_params([x.shape[1], y.shape[1]])
    costs = []
    cost = get_cost(x, y, model, params, lambd)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = lr).minimize(cost)
    init=tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epochs):
            c, _ = sess.run([cost, optimizer], feed_dict={x: inp, y:out})

            if epoch%(epochs//10) == 0:
                print('Cost at epoch ' + str(epoch), c)
            if epoch%5 == 0:
                costs.append(c)

        params = sess.run(params)

    return costs, params




# In[ ]:


costs, params = train(x, y, model = linear, lr= 1e-6)


# In[ ]:

plt.figure(2, figsize = (8, 6))
plt.plot(costs)
plt.show()


# In[ ]:


with tf.Session() as sess:
    inp = tf.constant(x, dtype = tf.float32)
    prediction = sess.run(linear(inp, params))


# In[ ]:


plt.figure(3, figsize = (8, 6))
plt.scatter(x[::10], y[::10], label = 'Actual Data')
plt.scatter(x[::10], prediction[::10], label = 'Predicted Data')
plt.legend()
plt.xlabel('Number of Households')
plt.ylabel('Total Rooms');
plt.show()


# In[ ]:
