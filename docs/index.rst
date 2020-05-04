=========
phys2denoise
=========

.. image:: _static/phys2denoise_logo1280×640.png
    :alt: phys2denoise logo
    :align: center

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3653153.svg
    :target: https://doi.org/10.5281/zenodo.3653153
    :alt: DOI

.. image:: https://travis-ci.org/physiopy/phys2denoise.svg?branch=master
    :target: https://travis-ci.org/physiopy/phys2denoise
    :alt: Build status

.. image:: https://badges.gitter.im/phys2denoise/community.svg
    :target: https://badges.gitter.im/phys2denoise/community.svg)](https://gitter.im/phys2denoise/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
    :alt: Join the chat at https://gitter.im/phys2denoise/community

.. image:: https://readthedocs.org/projects/phys2denoise/badge/?version=latest
    :target: https://phys2denoise.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://codecov.io/gh/physiopy/phys2denoise/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/physiopy/phys2denoise
    :alt: codecov


``phys2denoise`` is a python3 library meant to format physiological files in BIDS.
It was born for AcqKnowledge files (BIOPAC), and at the moment it supports
``.acq`` files as well as ``.txt`` files obtained by labchart
(ADInstruments).
It doesn't support physiological files recorded with the MRI, as you can find a software for it `here <https://github.com/tarrlab/physio2bids>`_.

**The project is currently under development**.
Any suggestion/bug report is welcome! Feel free to open an issue.

Citing ``phys2denoise``
--------------------

If you use ``phy2bids``, please cite it using the Zenodo DOI as:

    The phys2denoise contributors, Daniel Alcalá, Apoorva Ayyagari, Molly Bright, César Caballero-Gaudes, Vicente Ferrer Gallardo, Soichi Hayashi, Ross Markello, Stefano Moia, Rachael Stickland, Eneko Uruñuela, & Kristina Zvolanek (2020, February 6). physiopy/phys2denoise: BIDS formatting of physiological recordings v1.3.0-beta (Version v1.3.0-beta). Zenodo. http://doi.org/10.5281/zenodo.3653153


Contents
--------

.. toctree::
   :maxdepth: 1

   installation
   howto
   heuristic
   bestpractice
   cli
   contributing
   contributorfile
   conduct
   license
   api
