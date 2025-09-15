[![SM Logo](https://github.com/balouf/stochastic_matching/raw/master/docs/sm_logo.png)](https://balouf.github.io/stochastic_matching/)

# Stochastic Matching

[![PyPI Status](https://img.shields.io/pypi/v/stochastic_matching.svg)](https://pypi.python.org/pypi/stochastic_matching)
[![Build Status](https://github.com/balouf/stochastic_matching/actions/workflows/build.yml/badge.svg?branch=master)](https://github.com/balouf/stochastic_matching/actions?query=workflow%3Abuild)
[![Documentation Status](https://github.com/balouf/stochastic_matching/actions/workflows/docs.yml/badge.svg?branch=master)](https://github.com/balouf/stochastic_matching/actions?query=workflow%3Adocs)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Coverage](https://codecov.io/gh/balouf/stochastic_matching/branch/master/graphs/badge.svg)](https://codecov.io/gh/balouf/stochastic_matching/tree/master/stochastic_matching)

Stochastic Matching provides tools to analyze the behavior of stochastic matching problems.


* Free software: MIT License
* Documentation: https://balouf.github.io/stochastic_matching/.

## Features

* Compatibility graph creation (from scratch, from one of the provided generator, or by some combination).
* Theoretical analysis:
    * Injectivity/surjectivity of the graph, kernel description.
    * Polytope description of positive solutions.
* Fast simulator.
    * Provided with a large set of greedy / non-greedy policies.
    * Adding new policies is feasible out-of-the-box.
* Lot of display features, including a [Vis JS Network][VIS] export.


## Installation

To install Stochastic Matching, run this command in your terminal:

```console
$ pip install stochastic_matching
```

This is the preferred method to install Stochastic Matching, as it will always install the most recent stable release.

## Acknowledging package

If you publish results based on [Stochastic Matching][SM], **please acknowledge** the usage of the package by quoting the following paper.

* Céline Comte, Fabien Mathieu, Ana Bušić. [Online Stochastic Matching: A Polytope Perspective](https://hal.archives-ouvertes.fr/hal-03502084). 2024.

## Credits

This package was created with [Cookiecutter][CC] and the [Package Helper][PH3] project template.

[CC]: <https://github.com/audreyr/cookiecutter>
[PH3]: <https://balouf.github.io/package-helper-3/>
[VIS]: <https://visjs.github.io/vis-network/docs/network/>
[SM]: <https://balouf.github.io/stochastic_matching/>
