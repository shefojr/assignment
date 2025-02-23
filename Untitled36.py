#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random


# In[2]:


def tanh(x):
    return (2 / (1 + (2.71828 ** (-2 * x)))) - 1


# In[3]:


def tanh_derivative(x):
    return 1 - tanh(x) ** 2


# In[4]:


w = [random.uniform(-0.5, 0.5) for _ in range(8)]
i1, i2 = 0.05, 0.1
target_o1, target_o2 = 0.1, 0.99
b1, b2 = 0.5, 0.7
lr = 0.1


# In[5]:


net_h1 = w[0] * i1 + w[1] * i2 + b1
t_h1 = tanh(net_h1)

net_h2 = w[2] * i1 + w[3] * i2 + b1
t_h2 = tanh(net_h2)

net_o1 = w[4] * t_h1 + w[5] * t_h2 + b2
t_o1 = tanh(net_o1)

net_o2 = w[6] * t_h1 + w[7] * t_h2 + b2
t_o2 = tanh(net_o2)


# In[6]:


e_o1 = 0.5 * (target_o1 - t_o1) ** 2
e_o2 = 0.5 * (target_o2 - t_o2) ** 2
total_error = e_o1 + e_o2


# In[7]:


delta_o1 = (target_o1 - t_o1) * tanh_derivative(net_o1)
delta_o2 = (target_o2 - t_o2) * tanh_derivative(net_o2)

delta_h1 = (delta_o1 * w[4] + delta_o2 * w[6]) * tanh_derivative(net_h1)
delta_h2 = (delta_o1 * w[5] + delta_o2 * w[7]) * tanh_derivative(net_h2)


# In[8]:


w[4] += lr * delta_o1 * t_h1
w[5] += lr * delta_o1 * t_h2
w[6] += lr * delta_o2 * t_h1
w[7] += lr * delta_o2 * t_h2

w[0] += lr * delta_h1 * i1
w[1] += lr * delta_h1 * i2
w[2] += lr * delta_h2 * i1
w[3] += lr * delta_h2 * i2


# In[9]:


b1 += lr * (delta_h1 + delta_h2)
b2 += lr * (delta_o1 + delta_o2)


# In[10]:


print("Output of neuron o1:", t_o1)
print("Output of neuron o2:", t_o2)
print("Total Error:", total_error)


# In[ ]:




