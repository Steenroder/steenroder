Installation
------------

Dependencies
~~~~~~~~~~~~

The latest stable version of ``steenroder`` requires:

-  python (>= 3.8)
-  numpy (>= 1.19.1)
-  numba (>= 0.53.0)
-  psutils (>= 5.8.0)
-  gudhi (>= 3.5.0)
-  plotly (>= 5.3.1)

To run the examples, jupyter is required.

.. _installation-1:

Installation
~~~~~~~~~~~~

The simplest way to install ``steenroder`` is using ``pip`` ::

   python -m pip install -U steenroder

If necessary, this will also automatically install all the above
dependencies. Note: we recommend upgrading ``pip`` to a recent version
as the above may fail on very old versions.

Contributing
------------

We welcome new contributors of all experience levels. The Steenroder
community goals are to be helpful, welcoming, and effective. To learn
more about making a contribution to ``steenroder``, please consult the
`relevant
page <https://github.com/Steenroder/steenroder/CONTRIBUTING.md>`__.

Testing
~~~~~~~

After developer installation, you can launch the test suite from outside
the source directory:

::

   pytest steenroder
