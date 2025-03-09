# Mllm Code Conventions

## Project Structure


## C++ Style

### Why C++20?

The MLLM project specifically adopts C++20 primarily to leverage the designated initialization feature (as seen in the code snippet MatMulOpCargo{.transpose_a = ...}). This feature allows explicit and readable initialization of struct members by name, which improves code clarity when configuring operation parameters (like MatMulOpCargo).

No other C++20 features are utilized in the project. The codebase intentionally avoids newer C++20 constructs (e.g., concepts, ranges, modules, etc.) to maintain compatibility with minimal compiler requirements while retaining the explicit initialization syntax as the sole justification for adopting C++20.

## Python Style
