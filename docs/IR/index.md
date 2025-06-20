# MLLM IR

:::{note}
author: chenghua.wang
date: 2025-06-19
:::

## Contents

:::{toctree}
:maxdepth: 1

LinalgDialect
CFDialect
TensorDialect
GraphDialect
:::

------- 

**MLLM IR** is a **graph-level intermediate representation (IR)** system. Inspired by the *Dialect* concept in MLIR, it employs Dialects such as `Linalg`, `CF`, `Tensor`, and `Graph` to describe static graphs. Currently, MLLM IR only supports the construction of **static graphs**, with its design objectives as follows:  

1. To provide a concise and intuitive toolchain for facilitating graph construction by static graph backends (e.g., Qnn Graph, CUDA Graph, Xnnpack Graph, etc.);  
2. To offer *Pass* tools to assist in static graph optimization and analysis;  
3. To provide unified graph traversal methods.

**NOTE:**

Mllm's IR lacks support for conditional operations, making the trace function incompatible with dynamic control flows like `if statements`, `loops`, or `conditional constructs`. This stems from its static computational graph design, which enforces compile-time determined execution paths without runtime branching. When using the trace function, ensure your code contains no conditional logic, as it exclusively handles static graphs. For programs requiring dynamic control flow, either refactor code to eliminate conditionals or adopt mllm eager mode.

## How to Obtain MLLM IR via Tracing ?

MLLM provides a comprehensive tracing system to aid in the construction of static graph IR. You can obtain the IR through the following code:

```c++
// ... context init
mllm::models::QWenConfig cfg;
mllm::models::QWenForCausalLM llm(cfg);

// the model's parameters must be loaded before trace.
llm.load(mllm::load(model_files.get()));
auto ir_ctx = mllm::ir::trace(llm, Tensor::empty({1, 128}, kInt64));
auto dump_printer = IRPrinter();
ir_ctx->topLevelOp()->dump(dump_printer);
```

It will give the following output:

```text
@main () -> (){
    graph.CallGraphOp @model (%453:tensor<[1, 128], kInt64, kCPU>) -> (%1212:tensor<[1, 128, 151936], kFp32, kCPU>)
    graph.SubGraphOp @model <kCPU>{
        (%453:tensor<[1, 128], kInt64, kCPU>) -> (%1212:tensor<[1, 128, 151936], kFp32, kCPU>){
            linalg.kCPU.LLMEmbeddingTokenOp(%453:tensor<[1, 128], kInt64, kCPU>) -> (%454:tensor<[1, 128, 1536], kFp32, kCPU>)
            graph.CallGraphOp @model.layers.0 (%454:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%481:tensor<[1, 128, 1536], kFp32, kCPU>)
            graph.CallGraphOp @model.layers.1 (%481:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%508:tensor<[1, 128, 1536], kFp32, kCPU>)
            graph.CallGraphOp @model.layers.2 (%508:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%535:tensor<[1, 128, 1536], kFp32, kCPU>)
            graph.CallGraphOp @model.layers.3 (%535:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%562:tensor<[1, 128, 1536], kFp32, kCPU>)
            graph.CallGraphOp @model.layers.4 (%562:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%589:tensor<[1, 128, 1536], kFp32, kCPU>)
        ...
        }
    }
    ...
}
```

## Dialects: The Foundation of Modular IR Design

### What is a Dialect?

In MLIR-like IRs, a **Dialect** is a modular extension mechanism that encapsulates domain-specific operations, values, and attributes. It serves as a "language module" allowing different domains (e.g., machine learning, numerical computing, or hardware-specific optimizations) to define their own constructs while maintaining interoperability with the broader IR ecosystem. Dialects are named to avoid conflicts (e.g., `tensor.`, `arith.`, `llvm.`), and each can define:  

1. **Operations (Ops)**: Domain-specific computations or transformations.  
2. **Types**: Data representations (e.g., tensors, buffers, or custom structures).

    NOTE: In MLLM IR, we directly define value(e.g. TensorValue) instead of types.

3. **Attributes**: Immutable metadata configuring Ops (e.g., symbols, dtype, etc..).

By structuring IR into dialects, compilers can: 

1. **Specialize optimizations** for specific domains.
2. **Enable incremental compilation** by lowering dialects sequentially.
3. **Support heterogeneous hardware** through hardware-specific dialects.

For example, the `tensor` dialect handles tensor manipulations, while the `graph` dialect handles graph-level operations.

### **Operations (Ops)**

Ops are the atomic units of computation in a dialect. Each Op:

1. **Defines inputs and outputs** (values).  
2. **May carry attributes**.
3. **Implements semantic rules** (e.g., shape constraints for matrix multiplication).

**Key Properties**:

1. **SSA Form**: Each Op's result is a single static assignment (SSA value), ensuring deterministic data flow.

    NOTE: MLLM IR Do not have built-in support for SSA.

2. **Verification**: Ops validate their inputs and attributes during IR construction. 

    NOTE: MLLM IR has no built-in support for Verification.

3. **Lowering**: Ops can be transformed into lower-level ops or hardware instructions.

### **Values**

A **Value** represents a single result of an Op or a region argument.

**Key Roles**: 

1. **Inputs/Outputs of Ops**: Values are passed between Ops as operands.  
2. **Region Arguments**: Values entering a block.

### **Attributes** 

**Attributes** are immutable metadata attached to Ops, types, or other attributes. They configure Op behavior without altering data flow.

**Key Properties**:

1. **Immutability**: Once defined, attributes cannot change.
2. **Serialization**: Attributes are embedded in the IR text for reproducibility.

**Use Cases**:
1. **Op Configuration**: Strides in a convolution, element type in a tensor.  
2. **Type Parameters**: Rank or element type of a tensor.  
3. **Metadata**: Source location, debug information.  

## MLLM IR Syntax Specification

MLLM IR currently does not have a strictly formalized syntax; the dumped content shown above is solely for human observation and debugging, rather than for lexer/parser parsing. However, MLLM IR adheres to several fundamental conventions inspired by MLIR, detailed as follows:

### Symbol

In MLLM IR, a **Symbol** corresponds to the `SymbolAttribute`, used to identify operations (Ops) or values (Values) with global scope. Its syntax is defined as:  

```text
graph.CallGraphOp @<symbol_name> (<input_list>) -> (<output_list>)
```

**Design Semantics**:

1. Symbols share the same concept as those used in compiler symbol linking, enabling cross-module references to entities.
2. When an Op or Value is declared as a symbol, it is registered in the **IR Context**. Symbols can be retrieved by name via the `lookupSymbol` method in the IR Context.
3. **Scope Rule**: Symbol names must be globally unique (e.g., `@model.layers.0.mlp`).

### Operator

Operators are the core computational units in MLLM IR, describing specific computation logic or control flow. Their syntax follows this paradigm:  

```text
<dialect_name>.<op_name> (<input_type_list>) -> (<output_type_list>)
``` 

**Key Elements**:

1. **Dialect Name Prefix**

   Adopts MLIR-style hierarchical Name, e.g.:
   - `graph` for graph-structured operations (subgraph create, graph calls)
   - `tensor` for tensor operations

2. **Value List**

   Supports a mix of positional and named Values, e.g.:
   
    ```text
    linalg.kCPU.LinearOp(%455:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%456:tensor<[1, 128, 12, 128], kFp32, kCPU>)
    ```

### Value

Values are data carriers in MLLM IR, representing intermediate results or inputs/outputs in computations. Key characteristics include:

1. **Unique Identifier**  
   Each value is referenced by a `%<number>` symbol (e.g., `%1`, `%input_tensor`), unique within its scope.  

2. **Type System**
   Values are associated with specific types


### Extended Conventions  

1. **Graph Structure Representation**  
   Defines function-level graph structures using `graph.SubGraph` with `region`, e.g.: 

   ```text
    graph.SubGraphOp @model.layers.0.mlp <kCPU>{
        (%475:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%480:tensor<[1, 128, 1536], kFp32, kCPU>){
            linalg.kCPU.LinearOp(%475:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%476:tensor<[1, 128, 8960], kFp32, kCPU>)
            linalg.kCPU.SiLUOp(%476:tensor<[1, 128, 8960], kFp32, kCPU>) -> (%477:tensor<[1, 128, 8960], kFp32, kCPU>)
            linalg.kCPU.LinearOp(%475:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%478:tensor<[1, 128, 8960], kFp32, kCPU>)
            linalg.kCPU.MulOp(%477:tensor<[1, 128, 8960], kFp32, kCPU>, %478:tensor<[1, 128, 8960], kFp32, kCPU>) -> (%479:tensor<[1, 128, 8960], kFp32, kCPU>)
            linalg.kCPU.LinearOp(%479:tensor<[1, 128, 8960], kFp32, kCPU>) -> (%480:tensor<[1, 128, 1536], kFp32, kCPU>)
            cf.ReturnOp(%480:tensor<[1, 128, 1536], kFp32, kCPU>) -> ()
        }
    }
   ```

## The Organization and Storage of Node Information in MLLM IR Context

### RTTI Architecture

The MLLM IR Context utilizes a robust RTTI (Run-Time Type Information) system to organize and store node information. This mechanism enables type-safe operations and dynamic dispatch in the intermediate representation hierarchy.

1. NodeKind Enumeration

    - Defined in `GeneratedRTTIKind.hpp`
    - Serves as the core type identifier for all IR nodes
    - Contains specific values like:
        - RK_None (default null type)
        - RK_Op (operation nodes)
        - RK_Val (value nodes)
        - RK_Attr (attribute nodes)

2. Type Identification System

    - Implemented through:

        ```c++
        NodeKind getKind() const { return kind_; }
        ```
    - Each derived class provides type-checking methods:

        ```c++
        static inline bool classof(const Node* node) { 
            RTTI_RK_OP_IMPL(node); 
        }
        ```

3. Macro-based Type Hierarchy

    - Specialized macros implement type checking:

        ```c++
        #define DEFINE_SPECIFIC_IR_CLASS(_Type) \
            using self_ptr_t = std::shared_ptr<_Type>

        // In Op class:
        RTTI_RK_OP_IMPL(node);  // For operation nodes

        // In Attr class:
        RTTI_RK_ATTR_IMPL(node);  // For attribute nodes
        ```

4. Dynamic Casting System

    - Template-based casting methods ensure type safety:

        ```c++
        template<typename T>
        bool isa_() {
            return isa<T>(shared_from_this());
        }

        template<typename T>
        std::shared_ptr<T> cast_() {
            return cast<T>(shared_from_this());
        }
        ```

    - Check `mllm/Utils/RTTIHelper.hpp` for more details.

5. Context-aware Type Management

    - The `IRContext` class handles type-specific creation:

        ```c++
        template<typename T, typename... Args>
        std::shared_ptr<T> create(Args&&... args) {
            // Type-aware node creation logic
            if (created_node->isa_<Op>()) { /* Special handling */ }
            if (created_node->isa_<Val>()) { /* Symbol table management */ }
        }
        ```

### How to walk a region ?

#### Basic Concepts

**Region Structure**

```c++
class Region : public std::enable_shared_from_this<Region> {
    // Operation sequence container
    std::list<op_ptr_t> ops_;
    // Input/output value tracking
    std::list<val_ptr_t> inputs_;
    std::list<val_ptr_t> outputs_;
};
```

#### Using IRWriter's Built-in Walker

The recommended method for region traversal is through the `IRWriter::walk()` interface:

```c++
template<typename T>
bool walk(const std::function<WalkResult(IRWriter&, const std::shared_ptr<T>&)>& func) {
    // Implementation details...
}
```

**Example:**

```c++
// Find the top CallGraphOp
ir::graph::CallGraphOp::self_ptr_t call_main_graph_op = nullptr;
r.walk<ir::graph::CallGraphOp>(
    [&](ir::IRWriter& remover,
        const ir::graph::CallGraphOp::self_ptr_t& op) -> ir::IRWriter::WalkResult {
    // Make sure there is only one call graph op in the ModuleOp
    MLLM_RT_ASSERT_EQ(call_main_graph_op, nullptr);

    call_main_graph_op = op;
    return ir::IRWriter::WalkResult::WALK_CONTINUE;
});
```

#### Manual Region Traversal

For direct access to the region's operation list:

```c++
// Get region's operation list
auto& ops = region->ops();

// Forward iteration
for (const auto& op : ops) {
    process(op);
}

// Reverse iteration
for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
    process(*it);
}
```

#### Node Manipulation Techniques

**Insertion:**

```c++
// Insert at specific position
writer.createAtPos<NewOp>(target_op, IRWriter::AFTER, args...);
```

**Replacement:**

```c++
// Replace existing node
writer.replaceOp(old_op, new_op);
```

**Removal:**

```c++
// Remove operation
writer.removeOp(op_to_remove);
```

:::{note}
1. Use `IRWriter::walk()` for safe modifications
2. The walker automatically handles iterator invalidation
3. Manual iteration requires careful iterator management
4. Always check node validity with RTTI:
```cpp
if (op->isa_<SpecificOpType>()) {
    auto specific_op = op->cast_<SpecificOpType>();
}
```
:::

## How to Add New Dialect ?

TODO

## What about dynamic shape ?

TODO
