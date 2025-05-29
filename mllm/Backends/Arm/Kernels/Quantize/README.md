# Arm Quantization Kernels

|Name|From dType|To dType|Type|Other|
|:--:|:--:|:--:|:--:|:--:|
|fp32_s8s_pt|fp32|s8(signed int8) symmetry|Per Tensor|/|
|fp32_s8s_pto_key|fp32|s8(signed int8) symmetry|Per Token|Operating on Key online.  Will do $\hat{K}=K-\text{mean}{(K)}$ before quantization. $\text{Softmax}(QK^T) == {Softmax}(Q\hat{K}^T)$|

