# QNN Ops

The operators in the Qnn ops directory differ from those in Backends like `Arm` and `CUDA`, as their forward functions are empty. We need to reimplement the setup and reshape functions, along with the `Pattern` class within each operator. The `Pattern` class is utilized by Mllm's Graph Builder Pass.
