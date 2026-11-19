Installation
============

Quick Installation
------------------

You can install REAX using pip:

.. code-block:: bash

    pip install reax

Prerequisites
-------------

REAX requires Python 3.11 or later. It is built on top of JAX and Flax.

We recommend using a conda environment to manage your dependencies and ensure a clean environment.

.. code-block:: bash

    conda create -n reax_env python=3.11
    conda activate reax_env

Installing from Source
----------------------

If you want to use the latest features or contribute to the development of REAX, you can install
it directly from the source code.

1.  Clone the repository:

    .. code-block:: bash

        git clone https://github.com/camml-lab/reax.git
        cd reax

2.  Install the package in editable mode:

    .. code-block:: bash

        pip install -e .

    This command allows you to modify the source code and have the changes reflected immediately
    in your environment.

Dependencies
------------

REAX depends on several key libraries:

*   **JAX**: The core numerical computing library.
*   **Flax**: A neural network library for JAX.
*   **Optax**: A gradient processing and optimization library for JAX.

These will be automatically installed when you install REAX.
