.. _installation:

============
Installation
============

Requirements
------------

``phys2denoise`` requires python 3.6 or above, as well as the modules:

- ``numpy >= 1.9.3``
- ``matplotlib >= 3.1.1``

Depending on the processed files, it might require the **manual installation** of other modules.
At the moment, those modules are:

- `bioread`_, for AcqKnowledge (``.acq``) files.

.. _`bioread`: https://github.com/uwmadison-chm/bioread

Linux and mac installation
--------------------------

Install with ``pip``
^^^^^^^^^^^^^^^^^^^^

Pipy has the latest stable version of ``phys2denoise`` as a package. Just run
``pip3 install phys2denoise`` or ``pip install phys2denoise`` if your default python is python3.

If you want the latest development version of the program, download the package from `github <https://github.com/physiopy/phys2denoise>`_ and uncompress it.
Alternatively, if you have ``git``, use the command::

    git clone https://github.com/physiopy/phys2denoise.git

Open a terminal in the ``phy2bids`` folder and execute the command::

    pip3 install .

If python 3 is already your default, you might use instead::

    pip install .

If you need to install other libraries, you can call again ``pip``::

    pip3 install bioread

Install without ``pip``
^^^^^^^^^^^^^^^^^^^^^^^

Download the package from github and uncompress it.
Alternatively, if you have ``git``, use the command::

    git clone https://github.com/physiopy/phys2denoise.git

Open a terminal in the phy2bids folder and execute the command::

    python3 setup.py

If python 3 is already your default, you might use instead::

    python setup.py

Check your installation!
^^^^^^^^^^^^^^^^^^^^^^^^

Type the command::

    phys2denoise -v

If your output is: ``phys2denoise 1.3.0-beta`` or similar, ``phys2denoise`` is ready to be used.
