#!/usr/bin/env python3


"""Binomial class module for Binomial distribution calculations """


class Binomial:
    """Binomial class"""

    def __init__(self, data=None, n=1, p=0.5):
        """_summary_

        Args:
            data (_type_, optional): _description_. Defaults to None.
            n (int, optional): _description_. Defaults to 1.
            p (float, optional): _description_. Defaults to 0.5.

        Raises:
            ValueError: _description_
            ValueError: _description_
            TypeError: _description_
            ValueError: _description_
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if type(data) != list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = float(sum(data) / len(data))
            var = float((sum(map(lambda n: pow(n - mean,
                        2), data)) / len(data)))
            self.p = - (var / mean) + 1
            self.n = round(mean / self.p)
            self.p = mean / self.n

    def factorial(self, k):
        """ Find factorial of a number """
        result = 1
        for i in range(1, k+1):
            result *= i
        return result

    def pmf(self, k):
        """ Calculates the value of the PMF for a given number of "successes."

        Args:
            k (int): number of "successes"
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0

        n_fact = self.factorial(self.n)
        k_fact = self.factorial(k)
        n_k_fact = self.factorial(self.n - k)
        return (n_fact / (k_fact * n_k_fact)) * \
            (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of "successes."

        Args:
            k (int): is the number of successes.
        """

        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        return sum([self.pmf(i) for i in range(0, k+1)])
