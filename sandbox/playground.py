import numpy as np
import matplotlib.pyplot as plt
#
#
# a = np.array([1,2,3])
# b = np.array([1,2,3])
#
# c = a*b
# d = np.sum(c)
#
#
# print(c)
# print(d)

#
# for i in range(len(b)):
#     low_values_indices = b < 0
#
# print(low_values_indices)
#
# print(len(b))
#
# print(b[0])

# for i in range(self.options):
        #     print("option " + str(i) + " has area under the curve as:")
        #     print("using composite trapezoidal rule " + '{:18.5f}'.format(trapz(soln[:, i], range(0, len(soln)))))
        #     print("using composite Simpson's rule " + '{:18.5f}'.format(simps(soln[:, i], range(0, len(soln)))))
        #
        #     print("using composite trapezoidal rule " + '{:18.5f}'.format(trapz(self.orbits_for_pi_ei[i], range(0, len(self.orbits_for_pi_ei[i])))))
        #     print("using composite Simpson's rule " + '{:18.5f}'.format(simps(self.orbits_for_pi_ei[i], range(0, len(self.orbits_for_pi_ei[i])))))
        #     print("")

#
# time_vector = np.linspace(0, 100, 1)
# print(time_vector)
#
# x = [0.1,0.2,0.3]
# for i in x:
#     print(i)

# x =[1,2,3,4]
# print(x[0])
# y = np.array([(x[0] + x[1]), (x[2] + x[3])])
# print(y)

# a = [1,2]
# b = [3,4]
#
# a,b = b,a
#
# print(a)
# print(b)

#print(ord('A'))

# a = np.array([[0,0,0],[0,0,0]])
# b = np.array([[1,2,3],[4,5,6]])
#
# a[0] += b[0]
# a[0] += b[1]
#
# print(a)
#
# print(a/4)


Fs = 8000
f = 5
sample = 8000
x = np.arange(sample)
y = np.sin(2 * np.pi * f * x / Fs + 180)
plt.plot(x, y)
plt.xlabel('voltage(V)')
plt.ylabel('sample(n)')
plt.show()
