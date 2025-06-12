👋 Welcome to MLLM
==================

Install
--------

Build Python Package
~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   git clone --recursive https://github.com/chenghuaWang/mllm-advanced.git
   cd mllm-advanced
   pip install .

Build From Source
~~~~~~~~~~~~~~~~~
If you want to use c++ API or doing some development, you need build mllm from source. The following commands have been tested on Linux systems.

You should first clone the repository using the following command:

.. code-block:: shell

   git clone --recursive https://github.com/chenghuaWang/mllm-advanced.git

Mllm also provide docker file to help you build mllm. TODO

Build Android Target
^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

   export ANDROID_NDK_PATH = /path/to/android-ndk

   # build
   python task.py tasks/android_build.yaml

   # push to u device
   python task.py tasks/adb_push.yaml

Build X86 Target
^^^^^^^^^^^^^^^^

.. code-block:: shell

   # build
   python task.py tasks/x86_build.yaml

Build CUDA Target
^^^^^^^^^^^^^^^^^

.. code-block:: shell

   # build
   python task.py tasks/cuda_build.yaml

Build Document
^^^^^^^^^^^^^^
Before building the document, you need to install some dependencies using the following command:

.. code-block:: shell

   pip install -r docs/requirements.txt

Then you can build the document with a single command:

.. code-block:: shell

   python task.py tasks/build_doc.yaml

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Core Components
   :numbered:

   CoreComponents/index

.. toctree::
   :maxdepth: 3
   :caption: ARM Backend
   :numbered:

   ArmBackend/Design
   ArmBackend/KaiLinear
   ArmBackend/Benchmark/index

.. toctree::
   :maxdepth: 3
   :caption: QNN Backend
   :numbered:

   QnnBackend/QnnLoweringPipeline

.. toctree::
   :maxdepth: 3
   :caption: CUDA Backend
   :numbered:

   CudaBackend/Design
   CudaBackend/Kernels/index

.. toctree::
   :maxdepth: 1
   :caption: CONTRIBUTE
   :numbered:

   Contribute/CodeConventions
   Contribute/Roadmap

CPP API
-------

.. toctree::
   :maxdepth: 2
   :numbered:

   CppAPI/library_root


Translation
-----------

.. toctree::
   :maxdepth: 3
   :caption: ARM 后端

   ArmBackend/Design_zh
   ArmBackend/Benchmark/index

.. toctree::
   :maxdepth: 3
   :caption: CUDA 后端

   CudaBackend/Design_zh
   CudaBackend/Kernels/index

Indices and tables
------------------

* :ref:`genindex`
