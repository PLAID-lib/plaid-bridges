.. plaid documentation master file, created by
   sphinx-quickstart
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. raw:: html

    <br>

.. image:: https://plaid-lib.github.io/assets/images/plaid-bridges-logo.png
   :align: center
   :width: 300px

+-------------+-----------------------------------------------------------------------------------------------+
| **Testing** | |CI Status| |Docs| |Coverage| |Last Commit|                                                   |
+-------------+-----------------------------------------------------------------------------------------------+
| **Package** | |PyPI Version| |PyPi Downloads| |Platform| |Python Version|                                   |
+-------------+-----------------------------------------------------------------------------------------------+
| **Meta**    | |License| |GitHub Stars|                                                                      |
+-------------+-----------------------------------------------------------------------------------------------+


.. |CI Status| image:: https://github.com/PLAID-lib/plaid-bridges/actions/workflows/testing.yml/badge.svg
   :target: https://github.com/PLAID-lib/plaid-bridges/actions/workflows/testing.yml

.. |Docs| image:: https://readthedocs.org/projects/plaid-bridges/badge/?version=latest
   :target: https://plaid-bridges.readthedocs.io/en/latest/?badge=latest

.. |Coverage| image:: https://codecov.io/gh/plaid-lib/plaid-bridges/branch/main/graph/badge.svg
   :target: https://app.codecov.io/gh/plaid-lib/plaid-bridges/tree/main?search=&displayType=list

.. |Last Commit| image:: https://img.shields.io/github/last-commit/PLAID-lib/plaid-bridges/main
   :target: https://github.com/PLAID-lib/plaid-bridges/commits/main

.. |PyPI Version| image:: https://img.shields.io/pypi/v/plaid-bridges.svg
   :target: https://pypi.org/project/plaid-bridges/

.. |Platform| image:: https://img.shields.io/badge/platform-any-blue
   :target: https://github.com/PLAID-lib/plaid-bridges

.. |Python Version| image:: https://img.shields.io/pypi/pyversions/plaid-bridges
   :target: https://github.com/PLAID-lib/plaid-bridges

.. |PyPi Downloads| image:: https://static.pepy.tech/badge/plaid-bridges
   :target: https://pepy.tech/projects/plaid-bridges

.. |License| image:: https://anaconda.org/conda-forge/plaid/badges/license.svg
   :target: https://github.com/PLAID-lib/plaid-bridges/blob/main/LICENSE.txt

.. |GitHub Stars| image:: https://img.shields.io/github/stars/PLAID-lib/plaid-bridges?style=social
   :target: https://github.com/PLAID-lib/plaid-bridges


.. warning::

   The code is still in its initial configuration stages; interfaces may change. Use with care.

Plaid-bridges offers high-level adapters that connect PLAID datasets with popular machine learning ecosystems.
It has been developed at SafranTech, the research center of `Safran group <https://www.safran-group.com/>`_.

The code is hosted on `GitHub <https://github.com/PLAID-lib/plaid-bridges>`_ and the Python package is published as ``plaid-bridges``.

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Overview

   source/quickstart.md

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Advanced

   source/contributing.md

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Documentation

   API Reference <autoapi/plaid_bridges/index>
   Examples & Tutorials <source/notebooks.rst>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
