.. _installation_setup:

Installation and setup
======================

.. _basic_installation:

Basic installation
------------------

The easiest way to install ``phys2denoise`` is to use ``pip``. Assuming you have
Python >= 3.6 installed, you can install ``phys2denoise`` by opening a terminal
and running the following:

.. code-block:: bash

   pip install phys2denoise

.. warning::

   If you encounter an ImportError related to numpy.core.multiarray, please try to update
   your matplotlib version to 3.9.

Developer installation
----------------------

This package requires Python >= 3.6. Assuming you have the correct version of
Python installed, you can install ``phys2denoise`` by opening a terminal and running
the following:

.. code-block:: bash

   git clone https://github.com/physiopy/phys2denoise.git
   cd phys2denoise
   pip install -e .[dev]
