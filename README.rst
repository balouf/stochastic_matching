.. image:: https://github.com/balouf/stochastic_matching/raw/master/docs/sm_logo.png
    :alt: SMUP logo
    :target: https://balouf.github.io/stochastic_matching/


===================
Stochastic Matching
===================


.. image:: https://img.shields.io/pypi/v/stochastic_matching.svg
        :target: https://pypi.python.org/pypi/stochastic_matching
        :alt: PyPI Status

.. image:: https://github.com/balouf/stochastic_matching/workflows/build/badge.svg?branch=master
        :target: https://github.com/balouf/stochastic_matching/actions?query=workflow%3Abuild
        :alt: Build Status

.. image:: https://github.com/balouf/stochastic_matching/workflows/docs/badge.svg?branch=master
        :target: https://github.com/balouf/stochastic_matching/actions?query=workflow%3Adocs
        :alt: Documentation Status


.. image:: https://codecov.io/gh/balouf/stochastic_matching/branch/master/graphs/badge.svg
        :target: https://codecov.io/gh/balouf/stochastic_matching/tree/master/stochastic_matching
        :alt: Code Coverage


Stochastic Matching provides tools to analyze the behavior of stochastic matching problems.


* Free software: GNU General Public License v3
* Documentation: https://balouf.github.io/stochastic_matching/.


--------
Features
--------

* Compatibility graph creation (from scratch, from one of the provided generator, or by some combination).
* Theoretical analysis:
    * Injectivity/surjectivity of the graph, kernel description.
    * Polytope description of positive solutions.
* Fast simulator.
    * Provided with a large set of greedy / non-greedy policies.
    * Adding new policies is feasible out-of-the-box.
* Lot of display features, including `Vis JS Network`_.


---------------------
Installation
---------------------

To install Stochastic Matching, run this command in your terminal:

.. code-block:: console

    $ pip install stochastic_matching

This is the preferred method to install Stochastic Matching, as it will always install the most recent stable release.


---------------------------
Acknowledging package
---------------------------

If you publish results based on `Stochastic Matching`_, **please acknowledge** the usage of the package by quoting the following paper.

* Céline Comte, Fabien Mathieu, Ana Bušić. `Stochastic dynamic matching: A mixed graph-theory and linear-algebra approach <https://hal.archives-ouvertes.fr/hal-03502084>`_. 2022.

-------
Credits
-------

This package was created with Cookiecutter_ and the `francois-durand/package_helper_2`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`francois-durand/package_helper_2`: https://github.com/francois-durand/package_helper_2
.. _`Vis JS Network`: https://visjs.github.io/vis-network/docs/network/
.. _`Stochastic Matching`: https://balouf.github.io/stochastic_matching/

