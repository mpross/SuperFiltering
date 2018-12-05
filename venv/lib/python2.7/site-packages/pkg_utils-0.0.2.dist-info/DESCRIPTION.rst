|PyPI package| |Documentation| |Test results| |Test coverage| |Code
analysis| |License| |Analytics|

pkg\_utils
==========

Utilities for linking setuptools with package version metadata, GitHub
README.md files, requirements.txt files, and restoring overridden entry
points during for editable installations. Includes support for optional
dependencies.

Installation
------------

1. Install Python and pip:

   ::

       apt-get install python python-pip

2. Install this package:

   ::

       pip install git+git://github.com/KarrLab/pkg_utils#egg=pkg_utils

3. Optionally, install support for pandoc to link setuptools with GitHub
   markdown formatted README.md files:

   ::

       apt-get install pandoc
       pip install git+git://github.com/KarrLab/pkg_utils#egg=pkg_utils[pandoc]

Documentation
-------------

Please see the `API documentation <http://pkg_utils.readthedocs.io>`__.

License
-------

The build utilities are released under the `MIT license <LICENSE>`__.

Development team
----------------

This package was developed by the `Karr Lab <http://www.karrlab.org>`__
at the Icahn School of Medicine at Mount Sinai in New York, USA.

Questions and comments
----------------------

Please contact the `Karr Lab <http://www.karrlab.org>`__ with any
questions or comments.

.. |PyPI package| image:: https://img.shields.io/pypi/v/pkg_utils.svg
   :target: https://pypi.python.org/pypi/pkg_utils
.. |Documentation| image:: https://readthedocs.org/projects/pkg_utils/badge/?version=latest
   :target: http://pkg_utils.readthedocs.org
.. |Test results| image:: https://circleci.com/gh/KarrLab/pkg_utils.svg?style=shield
   :target: https://circleci.com/gh/KarrLab/pkg_utils
.. |Test coverage| image:: https://coveralls.io/repos/github/KarrLab/pkg_utils/badge.svg
   :target: https://coveralls.io/github/KarrLab/pkg_utils
.. |Code analysis| image:: https://api.codeclimate.com/v1/badges/719d7a9027bcdf6a63bc/maintainability
   :target: https://codeclimate.com/github/KarrLab/pkg_utils
.. |License| image:: https://img.shields.io/github/license/KarrLab/pkg_utils.svg
   :target: LICENSE
.. |Analytics| image:: https://ga-beacon.appspot.com/UA-86759801-1/pkg_utils/README.md?pixel



