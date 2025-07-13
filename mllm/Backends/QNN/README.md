# QNN Backend

## Overview

The QNN Backend enables hardware-accelerated inference on Qualcomm platforms. This backend integrates with the mllm framework to leverage Qualcomm's AI runtime and Hexagon DSP capabilities for efficient model execution.

## Prerequisites

Before compiling the QNN Backend, ensure you have the following dependencies installed:

1. **QAIRT (Qualcomm AI Runtime)**

    - Download the QAIRT package from [Qualcomm's official website](https://developer.qualcomm.com/).
    - It is recommended to install QAIRT to `/root/commonlib/qairt/<specific version>/...` when using the mllm QNN Docker image.
    - Follow the installation instructions provided by Qualcomm for your specific version.

2. **Hexagon SDK**

    - The Hexagon SDK is required for DSP offloading and development.
    - Install the SDK using the following commands:
      ```sh
      qpm-cli --login <username>
      qpm-cli --license-activate HexagonSDK5.x
      qpm-cli --install HexagonSDK5.x
      ```
    - By default, the Hexagon SDK will be installed to `/local/mnt/workspace/Qualcomm/Hexagon_SDK/`.

## Environment Setup

Before compiling, you must source the environment setup scripts for both QAIRT and Hexagon SDK:

- **QAIRT:** The setup script is typically located at `./bin/envsetup.sh` within the QAIRT installation directory.
- **Hexagon SDK:** The setup script is usually named `setup_sdk_env.source` in the Hexagon SDK installation directory.

Activate the environments by running:

```sh
source /root/commonlib/qairt/<specific version>/bin/envsetup.sh
source /local/mnt/workspace/Qualcomm/Hexagon_SDK/<specific version>/setup_sdk_env.source
```

## Compilation

The `mllm` repository provides a `task.py` script in the root directory to automate the build process for the QNN Backend.

To compile and install the QNN Backend, run:

```sh
python task.py tasks/android_build_qnn.sh
```

This command will build and install the mllm QNN backend, provided all dependencies and environment variables are correctly set.

## Troubleshooting

- Ensure all environment variables are set correctly after sourcing the setup scripts.
- Verify that the required versions of QAIRT and Hexagon SDK are installed.
- If you encounter build errors, consult the official documentation for QAIRT and Hexagon SDK, or check the mllm repository's issues page for known problems.
