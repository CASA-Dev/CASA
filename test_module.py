import CASA_Alpha
import numpy as np


# This file is used to create test cases and test individual functions
def jacobian_test():
    g = np.array([.5, 7, 10], dtype='f')
    # return CASA_Alpha.numerical_jacobian(test_fun, g)
    return CASA_Alpha.numerical_jacobian(test_fun2, g)
    # dtype needed, otherwise later on down the line python could typecast everything to an integer :(


def test_fun(x):
    # this function is used as a test function for testing the newton solver and numerical jacobian sub routine
    # used with np array of 3
    # test newton solver with parabolic elipse
    # z = x^2 + y^2 -4= x_hat dot x_hat -4
    return np.array([np.dot(x, x) - 4], dtype='f')
    # dtype needed, otherwise later on down the line python could typecast everything to an integer :(


def test_fun2(x):
    # this function is used as a test function for testing the newton solver and numerical jacobian sub routine
    # used with np array of 3
    # construct a function that maps R3 to R2, with f(2,1,2) = [0,0]
    f1 = np.power((x[0] - 2), 3) + x[2] - 2
    f2 = np.power((x[1] - 1), 2)
    return np.array([f1, f2], dtype='f')


def test_fun3(x):
    # Testcase found http://fourier.eng.hmc.edu/e176/lectures/NM/node21.html
    f1 = 3 * x[0] - np.cos(x[1] * x[2]) - 3 / 2
    f2 = 4 * np.power(x[0], 2) - 625 * np.power(x[1], 2) + x[2] * 2 - 1
    f3 = 20 * x[2] + np.exp(-x[0] * x[1]) + 9
    return np.array([f1, f2, f3], dtype='f')


def newton_test(testfunction, guess):
    return CASA_Alpha.newton_solver(testfunction, guess)


# Run tests from here-------------------------------------------
# print(jacobian_test())
guess = np.array([1, 1, 1], dtype='f')  # important to not guess integers !!!!!!!
print(newton_test(test_fun3, guess))
