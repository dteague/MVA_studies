.. MVA_studies documentation master file, created by
   sphinx-quickstart on Wed Jul  1 18:37:21 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
   
   
Installation Guide
==================

To install, the requirements are python and ROOT. For ease, this guide with assume one is using a virtual environment. If you are using ROOT 6.20 or greater, you can use python3. If you are using a version older than this (CMSSW areas use 6.12), use python2.7.

Python2.7 (CMSSW area)
**********************

.. code-block:: python
                
   python2.7 -m virtualenv env
   source env/bin/activate
   pip install -r requirements.txt

Python3 
*********************

.. code-block:: python

   python3 -m venv env
   source env/bin/activate
   pip install -r requirements.txt

**Note:** if you are new to virtual environments, to exit it, simply type ``deactivate``. You will need to make sure virtual environment is activated every time you want to run the code

Quick Start
===========

Before starting, you need to make sure you grab ``/afs/cern.ch/user/d/dteague/public/For_Deborah/inputTrees_new.root`` if you are on a computer connected to afs, or ``scp`` the file to your personal computer

To run the code, simply type

.. code-block:: bash

   ./runMVA.py -o <Output Dir> -t  # OR
   ./runMVA.py -o <Output Dir>     # Without training (just plotting)

Or if you need more information, just run ``./runMVA.py --help``

MVA_studies' documentation
==========================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   self
   running
   code

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`





  
