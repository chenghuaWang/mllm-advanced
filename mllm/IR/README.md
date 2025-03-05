# MLLM IR

Mllm's IR lacks support for conditional operations, making the trace function incompatible with dynamic control flows like `if statements`, `loops`, or `conditional constructs`. This stems from its static computational graph design, which enforces compile-time determined execution paths without runtime branching. When using the trace function, ensure your code contains no conditional logic, as it exclusively handles static graphs. For programs requiring dynamic control flow, either refactor code to eliminate conditionals or adopt mllm eager mode.

## llm examples

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
            graph.CallGraphOp @model.layers.5 (%589:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%616:tensor<[1, 128, 1536], kFp32, kCPU>)
            graph.CallGraphOp @model.layers.6 (%616:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%643:tensor<[1, 128, 1536], kFp32, kCPU>)
            graph.CallGraphOp @model.layers.7 (%643:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%670:tensor<[1, 128, 1536], kFp32, kCPU>)
            graph.CallGraphOp @model.layers.8 (%670:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%697:tensor<[1, 128, 1536], kFp32, kCPU>)
            graph.CallGraphOp @model.layers.9 (%697:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%724:tensor<[1, 128, 1536], kFp32, kCPU>)
            graph.CallGraphOp @model.layers.10 (%724:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%751:tensor<[1, 128, 1536], kFp32, kCPU>)
            graph.CallGraphOp @model.layers.11 (%751:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%778:tensor<[1, 128, 1536], kFp32, kCPU>)
            graph.CallGraphOp @model.layers.12 (%778:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%805:tensor<[1, 128, 1536], kFp32, kCPU>)
            graph.CallGraphOp @model.layers.13 (%805:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%832:tensor<[1, 128, 1536], kFp32, kCPU>)
            graph.CallGraphOp @model.layers.14 (%832:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%859:tensor<[1, 128, 1536], kFp32, kCPU>)
            graph.CallGraphOp @model.layers.15 (%859:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%886:tensor<[1, 128, 1536], kFp32, kCPU>)
            graph.CallGraphOp @model.layers.16 (%886:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%913:tensor<[1, 128, 1536], kFp32, kCPU>)
            graph.CallGraphOp @model.layers.17 (%913:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%940:tensor<[1, 128, 1536], kFp32, kCPU>)
            graph.CallGraphOp @model.layers.18 (%940:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%967:tensor<[1, 128, 1536], kFp32, kCPU>)
            graph.CallGraphOp @model.layers.19 (%967:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%994:tensor<[1, 128, 1536], kFp32, kCPU>)
            graph.CallGraphOp @model.layers.20 (%994:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%1021:tensor<[1, 128, 1536], kFp32, kCPU>)
            graph.CallGraphOp @model.layers.21 (%1021:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%1048:tensor<[1, 128, 1536], kFp32, kCPU>)
            graph.CallGraphOp @model.layers.22 (%1048:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%1075:tensor<[1, 128, 1536], kFp32, kCPU>)
            graph.CallGraphOp @model.layers.23 (%1075:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%1102:tensor<[1, 128, 1536], kFp32, kCPU>)
            graph.CallGraphOp @model.layers.24 (%1102:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%1129:tensor<[1, 128, 1536], kFp32, kCPU>)
            graph.CallGraphOp @model.layers.25 (%1129:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%1156:tensor<[1, 128, 1536], kFp32, kCPU>)
            graph.CallGraphOp @model.layers.26 (%1156:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%1183:tensor<[1, 128, 1536], kFp32, kCPU>)
            graph.CallGraphOp @model.layers.27 (%1183:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%1210:tensor<[1, 128, 1536], kFp32, kCPU>)
            linalg.kCPU.RMSNormOp(%1210:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%1211:tensor<[1, 128, 1536], kFp32, kCPU>)
            linalg.kCPU.MatMulOp(%1211:tensor<[1, 128, 1536], kFp32, kCPU>, %56:tensor<[151936, 1536], kFp32, kCPU>) -> (%1212:tensor<[1, 128, 151936], kFp32, kCPU>)
            cf.ReturnOp(%1212:tensor<[1, 128, 151936], kFp32, kCPU>) -> ()
        }
    }
    graph.SubGraphOp @model.layers.0 <kCPU>{
        (%454:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%481:tensor<[1, 128, 1536], kFp32, kCPU>){
            linalg.kCPU.RMSNormOp(%454:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%455:tensor<[1, 128, 1536], kFp32, kCPU>)
            graph.CallGraphOp @model.layers.0.self_attn (%455:tensor<[1, 128, 1536], kFp32, kCPU>, %455:tensor<[1, 128, 1536], kFp32, kCPU>, %455:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%473:tensor<[1, 128, 1536], kFp32, kCPU>)
            linalg.kCPU.AddOp(%473:tensor<[1, 128, 1536], kFp32, kCPU>, %454:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%474:tensor<[1, 128, 1536], kFp32, kCPU>)
            linalg.kCPU.RMSNormOp(%474:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%475:tensor<[1, 128, 1536], kFp32, kCPU>)
            graph.CallGraphOp @model.layers.0.mlp (%475:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%480:tensor<[1, 128, 1536], kFp32, kCPU>)
            linalg.kCPU.AddOp(%480:tensor<[1, 128, 1536], kFp32, kCPU>, %474:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%481:tensor<[1, 128, 1536], kFp32, kCPU>)
            cf.ReturnOp(%481:tensor<[1, 128, 1536], kFp32, kCPU>) -> ()
        }
    }
    graph.SubGraphOp @model.layers.0.self_attn <kCPU>{
        (%455:tensor<[1, 128, 1536], kFp32, kCPU>, %455:tensor<[1, 128, 1536], kFp32, kCPU>, %455:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%473:tensor<[1, 128, 1536], kFp32, kCPU>){
            linalg.kCPU.LinearOp(%455:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%456:tensor<[1, 128, 12, 128], kFp32, kCPU>)
            linalg.kCPU.LinearOp(%455:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%457:tensor<[1, 128, 2, 128], kFp32, kCPU>)
            linalg.kCPU.LinearOp(%455:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%458:tensor<[1, 128, 2, 128], kFp32, kCPU>)
            linalg.kCPU.TransposeOp(%456:tensor<[1, 128, 12, 128], kFp32, kCPU>) -> (%459:tensor<[1, 12, 128, 128], kFp32, kCPU>)
            linalg.kCPU.TransposeOp(%457:tensor<[1, 128, 2, 128], kFp32, kCPU>) -> (%460:tensor<[1, 2, 128, 128], kFp32, kCPU>)
            linalg.kCPU.TransposeOp(%458:tensor<[1, 128, 2, 128], kFp32, kCPU>) -> (%461:tensor<[1, 2, 128, 128], kFp32, kCPU>)
            linalg.kCPU.RoPEOp(%459:tensor<[1, 12, 128, 128], kFp32, kCPU>) -> (%462:tensor<[1, 12, 128, 128], kFp32, kCPU>)
            linalg.kCPU.RoPEOp(%460:tensor<[1, 2, 128, 128], kFp32, kCPU>) -> (%463:tensor<[1, 2, 128, 128], kFp32, kCPU>)
            linalg.kCPU.KVCacheOp(%463:tensor<[1, 2, 128, 128], kFp32, kCPU>) -> (%464:tensor<[1, 12, 128, 128], kFp32, kCPU>)
            linalg.kCPU.KVCacheOp(%461:tensor<[1, 2, 128, 128], kFp32, kCPU>) -> (%465:tensor<[1, 12, 128, 128], kFp32, kCPU>)
            linalg.kCPU.MatMulOp(%462:tensor<[1, 12, 128, 128], kFp32, kCPU>, %464:tensor<[1, 12, 128, 128], kFp32, kCPU>) -> (%466:tensor<[1, 12, 128, 128], kFp32, kCPU>)
            linalg.kCPU.MulOp(%466:tensor<[1, 12, 128, 128], kFp32, kCPU>, %467:tensor<[1], kFp32, kCPU>) -> (%468:tensor<[1, 12, 128, 128], kFp32, kCPU>)
            linalg.kCPU.CausalMaskOp(%468:tensor<[1, 12, 128, 128], kFp32, kCPU>) -> (%469:tensor<[1, 12, 128, 128], kFp32, kCPU>)
            linalg.kCPU.SoftmaxOp(%469:tensor<[1, 12, 128, 128], kFp32, kCPU>) -> (%470:tensor<[1, 12, 128, 128], kFp32, kCPU>)
            linalg.kCPU.MatMulOp(%470:tensor<[1, 12, 128, 128], kFp32, kCPU>, %465:tensor<[1, 12, 128, 128], kFp32, kCPU>) -> (%471:tensor<[1, 12, 128, 128], kFp32, kCPU>)
            linalg.kCPU.TransposeOp(%471:tensor<[1, 12, 128, 128], kFp32, kCPU>) -> (%472:tensor<[1, 128, 1536], kFp32, kCPU>)
            linalg.kCPU.LinearOp(%472:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%473:tensor<[1, 128, 1536], kFp32, kCPU>)
            cf.ReturnOp(%473:tensor<[1, 128, 1536], kFp32, kCPU>) -> ()
        }
    }
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
    ...
    graph.SubGraphOp @model.layers.27.mlp <kCPU>{
        (%1204:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%1209:tensor<[1, 128, 1536], kFp32, kCPU>){
            linalg.kCPU.LinearOp(%1204:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%1205:tensor<[1, 128, 8960], kFp32, kCPU>)
            linalg.kCPU.SiLUOp(%1205:tensor<[1, 128, 8960], kFp32, kCPU>) -> (%1206:tensor<[1, 128, 8960], kFp32, kCPU>)
            linalg.kCPU.LinearOp(%1204:tensor<[1, 128, 1536], kFp32, kCPU>) -> (%1207:tensor<[1, 128, 8960], kFp32, kCPU>)
            linalg.kCPU.MulOp(%1206:tensor<[1, 128, 8960], kFp32, kCPU>, %1207:tensor<[1, 128, 8960], kFp32, kCPU>) -> (%1208:tensor<[1, 128, 8960], kFp32, kCPU>)
            linalg.kCPU.LinearOp(%1208:tensor<[1, 128, 8960], kFp32, kCPU>) -> (%1209:tensor<[1, 128, 1536], kFp32, kCPU>)
            cf.ReturnOp(%1209:tensor<[1, 128, 1536], kFp32, kCPU>) -> ()
        }
    }
}
```