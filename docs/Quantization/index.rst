Mllm Quantize Tool
==================

.. note:: 

   author: chenghua.wang
   date: 2025-06-19

.. toctree::
   :numbered:

   Kleidiai


The MLLM Quantization Tool is designed to optimize large language models (LLMs) for deployment on resource-constrained devices. The `mllm-quantizer` only implements the PTQ methods.

.. note:: 
   
   The `mllm-quantizer` is currently optimized for mobile deployment scenarios, with full validation on Arm CPU and Qualcomm QNN backends. While additional backend integrations are in progress.


How to use `mllm-quantizer`
--------------------------

After compiling the mllm project, you will obtain an `mllm-quantizer` tool that helps quantify models. To perform model quantization, you need to provide two input files:

1. **mllm model file:** An un-quantized model file.
2. **Configuration file:** Specifies which operators to quantize and the quantization method for each operator.

For example, to quantize the `DeepSeek-Qwen2-1.5B` model, you can use the following command:

.. code-block:: shell 

   mllm-quantizer /path/to/DeepSeek-Qwen2-1.5B.mllm -c /path/to/DeepSeek-Qwen2-1.5B-QuantCfg.json -o /path/to/DeepSeek-Qwen2-1.5B-quantized.mllm

The `DeepSeek-Qwen2-1.5B-QuantCfg.json` file is defined in the following format:

.. code-block:: json 

   {
      "ModelName": "DeepSeek-R1-Distill-Qwen-1.5B",
      "Ops": {
         "model.layers.0.mlp.down_proj": {
            "type": "Linear",
            "implType": "KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk:qai8dxp4x8_qsi4c32p4x8_8x4x32"
         },
         "model.layers.0.mlp.gate_proj": {
            "type": "Linear",
            "implType": "KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk:qai8dxp4x8_qsi4c32p4x8_8x4x32"
         }
      }
   }

How `mllm-quantizer` works
--------------------------

**TL;DR:**

`mllm-quantizer` reads the input model file, generates a Flattened IR Module, and then utilizes Backend-specific Passes to transform this IR.

The Flattened IR Module has the following form:

.. code-block:: text

   @main () -> (){
      graph.CallGraphOp @<QUANT ANONYMOUS FLAT MODULE> () -> ()
      graph.SubGraphOp @<QUANT ANONYMOUS FLAT MODULE> <kCPU>{
         linalg.CPU.LinearOp() -> ()
         linalg.CPU.LinearOp() -> ()
         linalg.CPU.LinearOp() -> ()
      }
   }

The IR in the Flattened IR Module has no specific meaning, nor does it represent any model, and has no inputs or outputs. Its purpose is to represent the operators requiring quantization within the IR, enabling backend-specific Passes to traverse them.