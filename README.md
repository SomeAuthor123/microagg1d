[![build](https://github.com/Feelx234/microagg1d/actions/workflows/pythonapp.yml/badge.svg)](https://github.com/Feelx234/microagg1d/actions)

microagg1d
========

A Python library which implements a dynamic program for optimal univariate microaggregation. For an input array of length n and minimal class size k it has runtime max(O(n log(n)), O(kn)). It has space requirements of O(n).

The code is written in Python and relies on the [numba](https://numba.pydata.org/) compiler for speed.

Requirements
------------

*microagg1d* relies on `numpy` and `numba` which currently support python 3.8-3.10.

Installation
------------

[microagg1d](https://pypi.python.org/pypi/microagg1d) is available on PyPI, the Python Package Index.

```sh
$ pip3 install microagg1d
```

Example Usage
-------------

```python
import microagg1d

x = [5, 1, 1, 1.1, 5, 1, 5]
k = 3

clusters = microagg1d.optimal_univariate_microaggregation_1d(x, k)

print(clusters)   # [1 0 0 0 1 0 1]
```

Important notice: On first usage the the code is compiled once which may take about 30s. On subsequent usages this is no longer necessary and execution is much faster.

Tests
-----

Tests are in [tests/](https://github.com/Feelx234/microagg1d/blob/master/tests).

```sh
# Run tests
$ python3 -m pytest .
```

License
-------

The code in this repository has an BSD 2-Clause "Simplified" License.

See [LICENSE](https://github.com/Feelx234/microagg1d/blob/master/LICENSE).

