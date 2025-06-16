.. raw:: html

   <h1 align="center">
   MLLM
   </h1>

   <h3 align="center">
   <span style="color:#2563eb">M</span>obile x <span style="color:#8b5cf6">M</span>ultimodal
   </h3>
 
   <p align="center">
   Fast and lightweight LLM inference engine for mobile and edge devices
   </p>

   <p align="center">
   | Arm CPU | X86 CPU | Qualcomm NPU(QNN) |
   </p>

Using MLLM
-----

MLLM is a library designed for multimodal inference, enabling users to rapidly develop their own applications. While MLLM provides auxiliary programs like `mllm-runner` and `demo_xxx` to demonstrate its capabilities, these tools are limited to demo purposes. We recommend utilizing the MLLM C++ API as the primary framework for application development.

The MLLM library is extremely easy to use. Taking the deployment of DeepSeek Qwen2 on Arm CPU as an example, you can create an LLM instance with just a few lines of code:

.. code-block:: cpp

   auto& ctx = MllmEngineCtx::instance();
   ctx.registerBackend(mllm::arm::createArmBackend());
   ctx.mem()->initBuddyCtx(kCPU);
   mllm::models::DeepSeekQwen2Tokenizer tokenizer(tokenizer_file_path);
   mllm::models::QWenConfig cfg(cfg_file_path);
   mllm::models::AutoLLM<mllm::models::QWenForCausalLM> auto_llm(cfg);
   auto loader = mllm::load(model_files_path);
   auto_llm.model()->load(loader);

With this setup, you can now feed data to the model:

.. code-block:: cpp

   auto input = tokenizer.convert2Ids(
        tokenizer.tokenize("<｜begin▁of▁sentence｜>You are a helpful assistant.<｜User｜>hello, "
                           "what's u name?<｜Assistant｜>"));
   auto_llm.generate(input, 1024, cfg.eos_token_id, [&](int64_t pos) -> void {
   std::wcout << tokenizer.detokenize(pos) << std::flush;
   });

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


Using Docker or vscode devcontainer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To simplify developers' experience with MLLM, we provide ready-to-use Dockerfile and DevContainer configurations.

**Use docker:**

.. code-block:: shell

   git clone --recursive https://github.com/chenghuaWang/mllm-advanced
   cd mllm-advanced/docker
   docker build -t mllm_advanced_arm -f Dockerfile.arm .
   docker run -it --cap-add=SYS_ADMIN --network=host --cap-add=SYS_PTRACE --shm-size=4G --security-opt seccomp=unconfined --security-opt apparmor=unconfined --name mllm_a_arm_dev mllm_advanced_arm bash


Important Notes:

1. Dockerfile.arm includes NDK downloads. By using this image, you agree to NDK's additional terms.
2. QNN SDK contains proprietary licensing terms. We don't bundle it in Dockerfile.qnn - please configure QNN SDK manually.

**Use devcontainer:**

To set up with VS Code Dev Containers:

1. Install prerequisites:
   - Docker
   - VS Code
   - Dev Containers extension

2. Clone repository with submodules:

.. code-block:: shell

   git clone --recursive https://github.com/chenghuaWang/mllm-advanced
   

3. Open project in VS Code:

.. code-block:: shell
   code mllm-advanced
   

4. When prompted:

   "Folder contains a Dev Container configuration file. Reopen in container?"
   Click Reopen in Container

   (Alternatively: Press F1 → "Dev Containers: Reopen in Container")

The container will automatically build and launch with:

* All dependencies pre-installed
* Correct environment configuration
* Shared memory and security settings applied

Preparing Model and Tokenizer
-----------------------------

TODO

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Core Components
   :numbered:

   CoreComponents/index

.. toctree::
   :maxdepth: 2
   :caption: Quantization
   :numbered:

   Quantization/index

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
   :caption: Contribute
   :numbered:

   Contribute/CodeConventions
   Contribute/Roadmap

.. toctree::
   :maxdepth: 2
   :caption: C++ API

   CppAPI/library_root

Indices and tables
------------------

* :ref:`genindex`
