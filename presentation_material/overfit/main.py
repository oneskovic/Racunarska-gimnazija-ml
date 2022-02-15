from scipy.interpolate import lagrange
import numpy as np
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt

price_small = 650
price_big = 980
area_small = 800
area_big = 1400
goal_area = 2000

for cnt_small in range(1,4):
    for cnt_large in range(1,4):
        total_price = price_small * cnt_small + price_big * cnt_large
        total_area = area_small * cnt_small + area_big * cnt_large
        if total_area >= goal_area:
            print(f'{cnt_small} small, {cnt_large} large, price {total_price}')

# x = np.array([-3.2,-2.1,-1.3,0.4,0.9,1.3,2.4,2.9], dtype=np.float32)
minx = -10
maxx = 10
x = np.random.uniform(minx,maxx,8)
a = 5
b = 3
y = a*x+b
noise = np.random.randn(len(x))*5
y += noise
poly = lagrange(x, y)

ax = plt.gca()
ax.set_ylim([-100, 100])
x_new = np.arange(minx - 1, maxx+1, 0.01)
plt.scatter(x, y, label='Podaci')
plt.plot(x_new, a*x_new+b, label='Prava funkcija')
plt.plot(x_new, Polynomial(poly.coef[::-1])(x_new), label='Fitovana funkcija')

plt.legend()
plt.show()