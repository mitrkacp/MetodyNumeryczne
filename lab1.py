
# coding: utf-8

# In[1]:


import math
import numpy as np
import matplotlib.pyplot as plt

#zad1
print("zad1")
k = 1240* math.sqrt(7)
m = 4467
l = 2j
d = k + m
c = d + l


# In[2]:


#zad2
print("zad2")
print(round(d,3))
print(round(d,20))


# In[3]:


#zad3/4
print("zad3/4")
r = 17
h = 33

#wzór na pole powierzchni walca 
"""
    pPodstawy = pi * r^2
    pBoczne = 2 * pi *r * h
    pCalkowite = 2 * pPodst + pBoczne
"""

s = 2 * math.pi * r * r + 2 * math.pi * r * h
print(s)


# In[4]:


#zad5
print("zad5")
def wyrazenie (x1, t, r) :
    b = x1**(t*r) * (x1 + r) / (r * math.sin(2 * x1) + 3.3456)
    print (b)

wyrazenie (1, 2, 3)


# In[5]:


#zad6
print("zad6")
a = math.sqrt(2)
m = np.array([[a,1,-a],[0,1,1],[-a,a,1]])
print("Macierz: ")
print(m)

mO = np.linalg.inv(m)
print("Macierz odwrotna:")
print(mO)
mT = np.transpose(m)
print("Macierz transponowana: ")
print(mT)
mW = np.linalg.det(m)
print("Wyznacznik macierzy: ")
print(mW)


# In[6]:


#zad7
print("zad7")
print()
print(m[0,0])
print(m[2,2])
print(m[2,1])

w1 = m[:, 2]
w2 = m[1, :]
print(w1)
print(w2)


# In[8]:


#zad8
print("zad8")
pierwiastki = np.roots([1, -7, 3, 43, -28, -60])
print(pierwiastki)

def check(x):
    if (x**5 - 7*x**4 + 3*x**3 + 43*x**2 - 28*x - 60) == 0:
        return 1
    else:
        return 0

def podstawienie(x):
    return (x**5) - 7*(x**4) + 3*(x**3) + 43*(x**2) - 28*x - 60
    #return ((((((x)-7)*x+3)*x+43)*x-28)*x)-60

for i in pierwiastki:
    print("dla ",i," : ",podstawienie(i))
    if check(i):
        print(i," jest pierwiaskiem wielomianu")
    else:
        print(i," nie jest pierwiastkiem wielomianu")
        


# In[ ]:


#zad9
print("zad9")

ca1 = np.arange(0,10,0.1)
ca2 = np.linspace(0,10,5)
print(ca1)
print(ca2)


# In[ ]:


#zad10
print("zad10")
def funkcja(x):
    return x**3-3*x

x = np.linspace(-1,1)
y = []
for i in x:
    # dodawanie wyników wykonania funkcji do listy
    y.append(funkcja(i))
 
plt.plot(x, y)
plt.title('Wykres 1')
plt.xlabel('x')
plt.ylabel('y')
plt.legend('Wykres 1')
plt.show()

x = np.linspace(-5,5)
y = []
for i in x:
    # dodawanie wyników wykonania funkcji do listy
    y.append(funkcja(i))

plt.plot(x, y)
plt.title('Wykres 2')
plt.xlabel('x')
plt.ylabel('y')
plt.legend('Wykres 2')
plt.show()

x = np.linspace(0,5)
y = []
for i in x:
    # dodawanie wyników wykonania funkcji do listy
    y.append(funkcja(i))

plt.plot(x, y)
plt.title('Wykres 3')
plt.xlabel('x')
plt.ylabel('y')
plt.legend('Wykres 3')
plt.show()


# In[ ]:


#zad 11
print("zad11")
 
m=2500
v=60
 
def q(v,m):
    return m * v**2 / 2

print(q(60,2500))
print(q(60,2500)*(1/4.1868))

x = np.linspace(200,0)
y = []
for i in x:
    y.append(funkcja(i))

plt.plot(x, y)
plt.title('Wykres 4')
plt.xlabel('x')
plt.ylabel('y')
plt.legend('Wykres 4')
plt.show()

plt.semilogy(x, y)
plt.title('Wykres 5')
plt.xlabel('v')
plt.ylabel('Q')
plt.legend('Wykres 5')
plt.show()

