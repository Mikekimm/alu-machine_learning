#!/usr/bin/env python3
"""
Creates class that performs Bayesian optimization
on a noiseless 1D Gaussian process
"""

from scipy.stats import norm
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Performs Bayesian optimization on a noiseless 1D Gaussian process
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """
        Class constructor
        """
        if type(X_init) is not np.ndarray or len(X_init.shape) != 2:
            raise TypeError("X_init must be numpy.ndarray of shape (t, 1)")
        if X_init.shape[1] != 1:
            raise TypeError("X_init must be numpy.ndarray of shape (t, 1)")
        if type(Y_init) is not np.ndarray or len(Y_init.shape) != 2:
            raise TypeError("Y_init must be numpy.ndarray of shape (t, 1)")
        if Y_init.shape != X_init.shape:
            raise TypeError("Y_init must be numpy.ndarray of shape (t, 1)")
        if type(bounds) is not tuple or len(bounds) != 2:
            raise TypeError("bounds must be a tuple of (min, max)")
        min_, max_ = bounds
        if not isinstance(min_, (int, float)):
            raise TypeError("min in bounds must be int or float")
        if not isinstance(max_, (int, float)):
            raise TypeError("max in bounds must be int or float")
        if min_ >= max_:
            raise ValueError("min from bounds must be less than max")
        if not isinstance(ac_samples, int):
            raise TypeError("ac_samples must be an int")
        if not isinstance(l, (int, float)):
            raise TypeError("l must be int or float")
        if not isinstance(sigma_f, (int, float)):
            raise TypeError("sigma_f must be int or float")
        if not isinstance(xsi, (int, float)):
            raise TypeError("xsi must be int or float")
        if type(minimize) is not bool:
            raise TypeError("minimize must be a boolean")

        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(min_, max_, ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize
        self.X_init = X_init
        self.Y_init = Y_init

    def acquisition(self):
        """
        Calculates the next best sample location using Expected Improvement

        Returns:
            X_next: numpy.ndarray of shape (1,) - next best sample point
            EI: numpy.ndarray of shape (ac_samples,) - expected improvement
        """
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            best = np.min(self.gp.Y)
            imp = best - mu - self.xsi
        else:
            best = np.max(self.gp.Y)
            imp = mu - best - self.xsi

        with np.errstate(divide='warn'):
            Z = np.zeros_like(mu)
            Z[sigma != 0] = imp[sigma != 0] / sigma[sigma != 0]
            EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            EI[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI.flatten()

    def optimize(self, iterations=100):
        """
        Optimizes the black-box function

        Parameters:
            iterations [int]: max number of iterations to perform

        Returns:
            X_opt, Y_opt
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive number")

        X_all = self.gp.X.tolist()

        for _ in range(iterations):
            X_next, _ = self.acquisition()

            # Stop if X_next was already sampled
            if X_next.tolist() in X_all:
                break

            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)
            X_all.append(X_next.tolist())

        if self.minimize:
            idx_opt = np.argmin(self.gp.Y)
        else:
            idx_opt = np.argmax(self.gp.Y)

        X_opt = self.gp.X[idx_opt]
        Y_opt = self.gp.Y[idx_opt]

        return X_opt, Y_opt

