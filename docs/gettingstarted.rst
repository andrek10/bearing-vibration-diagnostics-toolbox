Getting Started
===============

Installation
------------

There are two ways to install this module.

1. Install using pip in your current Python environment
2. Download the repository and put `pyvib` folder in your project

Pip install
^^^^^^^^^^^

Within your current Python environment (either global, through `pyenv` or `conda`), run the following code to install from the git repository:

.. code-block::

    pip install git+https://github.com/andrek10/bearing-vibration-diagnostics-toolbox.git@master

This will install PyVib to your activated environment.

Project install
^^^^^^^^^^^^^^^

The repository can be cloned or downloaded, and then put the `pyvib` folder in your project folder.

To install the required packages, you may use `pip` to install them.

Navigate to the `pyvib` folder and run the following command:

.. code-block::

    pip install -r .

Example
-------

Here follows an example on running some of the code in the repository.

For this example, we will use some open-sourced dataset from University of Agder.
The data repository is accessible `here <https://doi.org/10.18710/BG1QNG>`_.
As an example, we will focus on a single file located `here <https://doi.org/10.18710/BG1QNG/PY6I6X>`_.
(`direct download link <https://dataverse.no/file.xhtml?persistentId=doi:10.18710/BG1QNG/PY6I6X&version=1.1>`_ to the file).

