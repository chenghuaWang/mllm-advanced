/**
 * @file QnnTensorTransform.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-10
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/QNN/Runtime/QnnTensorTransform.hpp"
#include "mllm/Backends/QNN/QnnTensorHelpMacros.hpp"

#include <QNN/QnnTypes.h>

namespace mllm::qnn {

const std::unordered_map<Qnn_DataType_t, size_t> QnnTensorTransform::dtype_to_size_ = {
    {QNN_DATATYPE_INT_8, 1},           {QNN_DATATYPE_INT_16, 2},
    {QNN_DATATYPE_INT_32, 4},          {QNN_DATATYPE_INT_64, 8},
    {QNN_DATATYPE_UINT_8, 1},          {QNN_DATATYPE_UINT_16, 2},
    {QNN_DATATYPE_UINT_32, 4},         {QNN_DATATYPE_UINT_64, 8},
    {QNN_DATATYPE_FLOAT_16, 2},        {QNN_DATATYPE_FLOAT_32, 4},
    {QNN_DATATYPE_BOOL_8, 1},          {QNN_DATATYPE_SFIXED_POINT_8, 1},
    {QNN_DATATYPE_SFIXED_POINT_16, 2}, {QNN_DATATYPE_SFIXED_POINT_32, 4},
    {QNN_DATATYPE_UFIXED_POINT_8, 1},  {QNN_DATATYPE_UFIXED_POINT_16, 2},
    {QNN_DATATYPE_UFIXED_POINT_32, 4},
};

QnnTensorTransform::~QnnTensorTransform() {
  for (auto& t : qnn_tensors_) {
    for (auto& ptr : t.anonymous_trash_) {
      // NOTE:
      // Always use malloc and free !!! Do not use new and delete !!!
      //
      // new and delete need to know the pointer is an array or not while malloc and free don't.
      free(ptr);
    }
  }
}

Qnn_Tensor_t QnnTensorTransform::transform(const ir::tensor::TensorValue::self_ptr_t& tensor_ir,
                                           Qnn_TensorVersion_t version) {
  switch (version) {
    case QNN_TENSOR_VERSION_1: return transformV1(tensor_ir);
    case QNN_TENSOR_VERSION_2: return transformV2(tensor_ir);
    default: NYI("The QNN Tensor {} Version is not supported yet.", (int32_t)(version));
  }

  // Fall back to try. But may not correct.
  return transformV2(tensor_ir);
}

Qnn_Tensor_t QnnTensorTransform::transform(Tensor& mllm_tensor, Qnn_TensorVersion_t version) {
  switch (version) {
    case QNN_TENSOR_VERSION_1: return transformV1(mllm_tensor);
    case QNN_TENSOR_VERSION_2: return transformV2(mllm_tensor);
    default: NYI("The QNN Tensor {} Version is not supported yet.", (int32_t)(version));
  }

  // Fall back to try. But may not correct.
  return transformV2(mllm_tensor);
}

Qnn_Tensor_t QnnTensorTransform::deepCopy(Qnn_Tensor_t* src_tensor) {
  if (!src_tensor) { MLLM_ERROR_EXIT(kError, "src_tensor is nullptr!"); }

  if (src_tensor->version != QNN_TENSOR_VERSION_2) {
    MLLM_ERROR_EXIT(kError, "The QNN Tensor Version is not supported for DeepCopy in mllm yet. "
                            "Trying to use Qnn Tensor V2");
  }

  Qnn_Tensor_t ret{
      .version = QNN_TENSOR_VERSION_2,
      .v2 = QNN_TENSOR_V2_INIT,
  };
  QnnTensorTransformMetaInfo mllm_handled_qnn_tensor_meta_info;

  // Things we need to copy
  //
  //   {
  //       0u,                                 /*id*/
  //       NULL,                               /*name*/
  //       QNN_TENSOR_TYPE_UNDEFINED,          /*type*/
  //       QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER, /*dataFormat*/
  //       QNN_DATATYPE_UNDEFINED,             /*dataType*/
  //       QNN_QUANTIZE_PARAMS_INIT,           /*quantizeParams*/
  //       0u,                                 /*rank*/
  //       NULL,                               /*dimensions*/
  //       QNN_TENSORMEMTYPE_UNDEFINED,        /*memType*/
  //       {
  //           QNN_CLIENT_BUFFER_INIT /*clientBuf*/
  //       },
  //       NULL,                   /*isDynamicDimension*/
  //       QNN_SPARSE_PARAMS_INIT, /*sparseParams*/
  //       0u                      /*isProduced*/
  //   }

  // == ID
  { HELP_QNN_TENSOR_SET_ID(ret, HELP_QNN_TENSOR_GET_ID(src_tensor)); }

  // == Name
  {
    auto name_ptr = strdup(HELP_QNN_TENSOR_GET_NAME(src_tensor));
    mllm_handled_qnn_tensor_meta_info.anonymous_trash_.emplace_back(name_ptr);
    HELP_QNN_TENSOR_SET_NAME(ret, name_ptr);
  }

  // == Type
  { HELP_QNN_TENSOR_SET_TYPE(ret, HELP_QNN_TENSOR_GET_TYPE(src_tensor)); }

  // == Dataformat
  { HELP_QNN_TENSOR_SET_DATA_FORMAT(ret, HELP_QNN_TENSOR_GET_DATA_FORMAT(src_tensor)); }

  // == DataType
  { HELP_QNN_TENSOR_SET_DATA_TYPE(ret, HELP_QNN_TENSOR_GET_DATA_TYPE(src_tensor)); }

  // == QuantizeParams
  {
    auto src_quant_cfg = HELP_QNN_TENSOR_GET_QUANT_PARAMS(src_tensor);
    Qnn_QuantizeParams_t ret_quant_cfg = QNN_QUANTIZE_PARAMS_INIT;
    ret_quant_cfg.encodingDefinition = src_quant_cfg.encodingDefinition;
    ret_quant_cfg.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;

    switch (ret_quant_cfg.quantizationEncoding) {
      case QNN_QUANTIZATION_ENCODING_SCALE_OFFSET: {
        ret_quant_cfg.quantizationEncoding = src_quant_cfg.quantizationEncoding;
        ret_quant_cfg.scaleOffsetEncoding = src_quant_cfg.scaleOffsetEncoding;
        break;
      }
      case QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET: {
        ret_quant_cfg.quantizationEncoding = src_quant_cfg.quantizationEncoding;
        ret_quant_cfg.axisScaleOffsetEncoding.axis = src_quant_cfg.axisScaleOffsetEncoding.axis;
        ret_quant_cfg.axisScaleOffsetEncoding.numScaleOffsets =
            src_quant_cfg.axisScaleOffsetEncoding.numScaleOffsets;

        if (src_quant_cfg.axisScaleOffsetEncoding.numScaleOffsets > 0) {
          ret_quant_cfg.axisScaleOffsetEncoding.scaleOffset = (Qnn_ScaleOffset_t*)malloc(
              src_quant_cfg.axisScaleOffsetEncoding.numScaleOffsets * sizeof(Qnn_ScaleOffset_t));
          mllm_handled_qnn_tensor_meta_info.anonymous_trash_.emplace_back(
              ret_quant_cfg.axisScaleOffsetEncoding.scaleOffset);
          if (ret_quant_cfg.axisScaleOffsetEncoding.scaleOffset) {
            for (size_t idx = 0; idx < src_quant_cfg.axisScaleOffsetEncoding.numScaleOffsets;
                 idx++) {
              ret_quant_cfg.axisScaleOffsetEncoding.scaleOffset[idx].scale =
                  src_quant_cfg.axisScaleOffsetEncoding.scaleOffset[idx].scale;
              ret_quant_cfg.axisScaleOffsetEncoding.scaleOffset[idx].offset =
                  src_quant_cfg.axisScaleOffsetEncoding.scaleOffset[idx].offset;
            }
          }
        }
        break;
      }
      default: {
        MLLM_ERROR_EXIT(kError, "This type's Quantization Encoding is not implemented yet.");
      }
    }
    HELP_QNN_TENSOR_SET_QUANT_PARAMS(ret, ret_quant_cfg);
  }

  // == Rank
  { HELP_QNN_TENSOR_SET_RANK(ret, HELP_QNN_TENSOR_GET_RANK(src_tensor)); }

  // == Dimensions
  {
    auto _dim_ptr = HELP_QNN_TENSOR_GET_DIMENSIONS(src_tensor);
    auto dim_ptr = (uint32_t*)malloc(HELP_QNN_TENSOR_GET_RANK(src_tensor) * sizeof(uint32_t));
    mllm_handled_qnn_tensor_meta_info.anonymous_trash_.emplace_back(dim_ptr);
    auto rank = HELP_QNN_TENSOR_GET_RANK(src_tensor);
    for (size_t i = 0; i < rank; i++) { dim_ptr[i] = _dim_ptr[i]; }
    HELP_QNN_TENSOR_SET_DIMENSIONS(ret, dim_ptr);
  }

  // == MemType
  { HELP_QNN_TENSOR_SET_MEM_TYPE(ret, HELP_QNN_TENSOR_GET_MEM_TYPE(src_tensor)); }

  // == ClientBuf
  { HELP_QNN_TENSOR_SET_CLIENT_BUF(ret, HELP_QNN_TENSOR_GET_CLIENT_BUF(src_tensor)); }

  // == Is Dynamic Dimension
  {
    // TODO
  }

  // == SparseParams
  {
    // TODO
  }

  // == Is Produced
  {
    // TODO
  }
  return ret;
}

Qnn_Tensor_t QnnTensorTransform::transformV1(const ir::tensor::TensorValue::self_ptr_t& tensor_ir) {
  Qnn_Tensor_t ret_qnn_tensor;
  NYI("v1 Qnn_Tensor_t is legacy. Why not try v2 of Qnn_Tensor_t");
  return ret_qnn_tensor;
}

Qnn_Tensor_t QnnTensorTransform::transformV2(const ir::tensor::TensorValue::self_ptr_t& tensor_ir) {
  // Init all members in Qnn_Tensor_t
  Qnn_Tensor_t ret_qnn_tensor{
      .version = QNN_TENSOR_VERSION_2,
      .v2 = QNN_TENSOR_V2_INIT,
  };

  QnnTensorTransformMetaInfo mllm_handled_qnn_tensor_meta_info;

  // Things we need to init
  //
  //   {
  //       0u,                                 /*id*/
  //       NULL,                               /*name*/
  //       QNN_TENSOR_TYPE_UNDEFINED,          /*type*/
  //       QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER, /*dataFormat*/
  //       QNN_DATATYPE_UNDEFINED,             /*dataType*/
  //       QNN_QUANTIZE_PARAMS_INIT,           /*quantizeParams*/
  //       0u,                                 /*rank*/
  //       NULL,                               /*dimensions*/
  //       QNN_TENSORMEMTYPE_UNDEFINED,        /*memType*/
  //       {
  //           QNN_CLIENT_BUFFER_INIT /*clientBuf*/
  //       },
  //       NULL,                   /*isDynamicDimension*/
  //       QNN_SPARSE_PARAMS_INIT, /*sparseParams*/
  //       0u                      /*isProduced*/
  //   }

  // == Name
  {
    auto name_ptr = strdup(tensor_ir->name().c_str());
    mllm_handled_qnn_tensor_meta_info.anonymous_trash_.emplace_back(name_ptr);
    HELP_QNN_TENSOR_SET_NAME(ret_qnn_tensor, name_ptr);
  }

  // == Type
  { HELP_QNN_TENSOR_SET_TYPE(ret_qnn_tensor, autoQnnTensorType(tensor_ir)); }

  // == Dataformat
  { HELP_QNN_TENSOR_SET_DATA_FORMAT(ret_qnn_tensor, autoQnnTensorDataFormat(tensor_ir)); }

  // == DataType
  { HELP_QNN_TENSOR_SET_DATA_TYPE(ret_qnn_tensor, autoQnnTensorDataType(tensor_ir)); }

  // == Quantization
  { HELP_QNN_TENSOR_SET_QUANT_PARAMS(ret_qnn_tensor, autoQnnTensorQuantParams(tensor_ir)); }

  // == Rank
  { HELP_QNN_TENSOR_SET_RANK(ret_qnn_tensor, tensor_ir->tensor_.shape().size()); }

  // == Dimensions
  {
    auto dim_ptr = (uint32_t*)malloc(tensor_ir->tensor_.shape().size() * sizeof(uint32_t));
    mllm_handled_qnn_tensor_meta_info.anonymous_trash_.emplace_back(dim_ptr);
    auto shape = tensor_ir->tensor_.shape();
    for (size_t i = 0; i < shape.size(); i++) { dim_ptr[i] = shape[i]; }
    HELP_QNN_TENSOR_SET_DIMENSIONS(ret_qnn_tensor, dim_ptr);
  }

  // == MemType
  { HELP_QNN_TENSOR_SET_MEM_TYPE(ret_qnn_tensor, autoQnnTensorMemType(tensor_ir)); }

  // == Client Buf
  {
    // Only mllm's tensor who has kParams memtype should init client buf
    if (tensor_ir->tensor_.memType() == kParams) {
      Qnn_ClientBuffer_t cb = QNN_CLIENT_BUFFER_INIT;
      cb.data = tensor_ir->tensor_.ptr<void>();
      cb.dataSize = tensor_ir->tensor_.bytes();
      HELP_QNN_TENSOR_SET_CLIENT_BUF(ret_qnn_tensor, cb);
    }
  }

  // == Is Dynamic Dimensions
  {
    // TODO
  }

  // == Sparse Params
  {
    // TODO
  }

  // == Is produced
  {
    // TODO
  }

  mllm_handled_qnn_tensor_meta_info.qnn_tensor_ = ret_qnn_tensor;
  qnn_tensors_.emplace_back(mllm_handled_qnn_tensor_meta_info);
  return ret_qnn_tensor;
}

Qnn_Tensor_t QnnTensorTransform::transformV1(Tensor& mllm_tensor) {
  Qnn_Tensor_t ret_qnn_tensor;
  NYI("v1 Qnn_Tensor_t is legacy. Why not try v2 of Qnn_Tensor_t");
  return ret_qnn_tensor;
}

Qnn_Tensor_t QnnTensorTransform::transformV2(Tensor& mllm_tensor) {
  // Init all members in Qnn_Tensor_t
  Qnn_Tensor_t ret_qnn_tensor{
      .version = QNN_TENSOR_VERSION_2,
      .v2 = QNN_TENSOR_V2_INIT,
  };

  QnnTensorTransformMetaInfo mllm_handled_qnn_tensor_meta_info;

  // Things we need to init
  //
  //   {
  //       0u,                                 /*id*/
  //       NULL,                               /*name*/
  //       QNN_TENSOR_TYPE_UNDEFINED,          /*type*/
  //       QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER, /*dataFormat*/
  //       QNN_DATATYPE_UNDEFINED,             /*dataType*/
  //       QNN_QUANTIZE_PARAMS_INIT,           /*quantizeParams*/
  //       0u,                                 /*rank*/
  //       NULL,                               /*dimensions*/
  //       QNN_TENSORMEMTYPE_UNDEFINED,        /*memType*/
  //       {
  //           QNN_CLIENT_BUFFER_INIT /*clientBuf*/
  //       },
  //       NULL,                   /*isDynamicDimension*/
  //       QNN_SPARSE_PARAMS_INIT, /*sparseParams*/
  //       0u                      /*isProduced*/
  //   }

  // == Name
  {
    auto name_ptr = strdup(mllm_tensor.name().c_str());
    mllm_handled_qnn_tensor_meta_info.anonymous_trash_.emplace_back(name_ptr);
    HELP_QNN_TENSOR_SET_NAME(ret_qnn_tensor, name_ptr);
  }

  // == Type
  { HELP_QNN_TENSOR_SET_TYPE(ret_qnn_tensor, autoQnnTensorType(mllm_tensor)); }

  // == Dataformat
  { HELP_QNN_TENSOR_SET_DATA_FORMAT(ret_qnn_tensor, autoQnnTensorDataFormat(mllm_tensor)); }

  // == DataType
  { HELP_QNN_TENSOR_SET_DATA_TYPE(ret_qnn_tensor, autoQnnTensorDataType(mllm_tensor)); }

  // == Quantization
  { HELP_QNN_TENSOR_SET_QUANT_PARAMS(ret_qnn_tensor, autoQnnTensorQuantParams(mllm_tensor)); }

  // == Rank
  { HELP_QNN_TENSOR_SET_RANK(ret_qnn_tensor, mllm_tensor.shape().size()); }

  // == Dimensions
  {
    auto dim_ptr = (uint32_t*)malloc(mllm_tensor.shape().size() * sizeof(uint32_t));
    mllm_handled_qnn_tensor_meta_info.anonymous_trash_.emplace_back(dim_ptr);
    auto shape = mllm_tensor.shape();
    for (size_t i = 0; i < shape.size(); i++) { dim_ptr[i] = shape[i]; }
    HELP_QNN_TENSOR_SET_DIMENSIONS(ret_qnn_tensor, dim_ptr);
  }

  // == MemType
  { HELP_QNN_TENSOR_SET_MEM_TYPE(ret_qnn_tensor, autoQnnTensorMemType(mllm_tensor)); }

  // == Client Buf
  {
    // Only mllm's tensor who has kParams memtype should init client buf
    if (mllm_tensor.memType() == kParams) {
      Qnn_ClientBuffer_t cb = QNN_CLIENT_BUFFER_INIT;
      cb.data = mllm_tensor.ptr<void>();
      cb.dataSize = mllm_tensor.bytes();
      HELP_QNN_TENSOR_SET_CLIENT_BUF(ret_qnn_tensor, cb);
    }
  }

  // == Is Dynamic Dimensions
  {
    // TODO
  }

  // == Sparse Params
  {
    // TODO
  }

  // == Is produced
  {
    // TODO
  }

  mllm_handled_qnn_tensor_meta_info.qnn_tensor_ = ret_qnn_tensor;
  qnn_tensors_.emplace_back(mllm_handled_qnn_tensor_meta_info);
  return ret_qnn_tensor;
}

Qnn_TensorType_t QnnTensorTransform::autoQnnTensorType(
    const ir::tensor::TensorValue::self_ptr_t& tensor_ir) {
  if (tensor_ir->getAttr("is_graph_output") != nullptr) {
    return QNN_TENSOR_TYPE_APP_READ;
  } else if (tensor_ir->getAttr("is_graph_input") != nullptr) {
    return QNN_TENSOR_TYPE_APP_WRITE;
  }

  if (tensor_ir->tensor_.memType() == kParams) { return QNN_TENSOR_TYPE_STATIC; }

  // FIXME: Handle Static and changeable Qnn quantized tensors.

  return QNN_TENSOR_TYPE_NATIVE;
}

Qnn_TensorDataFormat_t QnnTensorTransform::autoQnnTensorDataFormat(
    const ir::tensor::TensorValue::self_ptr_t& tensor_ir) {
  // FIXME: Handle other things
  return QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
}

Qnn_DataType_t QnnTensorTransform::autoQnnTensorDataType(
    const ir::tensor::TensorValue::self_ptr_t& tensor_ir) {
  Qnn_DataType_t ret_qnn_dtype;

  auto& mllm_tensor = tensor_ir->tensor_;

  switch (mllm_tensor.dtype()) {
    case kInt32: ret_qnn_dtype = QNN_DATATYPE_INT_32; break;
    case kFp32: ret_qnn_dtype = QNN_DATATYPE_FLOAT_32; break;
    case kFp16: ret_qnn_dtype = QNN_DATATYPE_FLOAT_16; break;
    case kPTInt8_Sym: ret_qnn_dtype = QNN_DATATYPE_SFIXED_POINT_8; break;
    default: NYI("Not supported data type {}", (int)mllm_tensor.dtype()); break;
  }

  return ret_qnn_dtype;
}

Qnn_QuantizeParams_t QnnTensorTransform::autoQnnTensorQuantParams(
    const ir::tensor::TensorValue::self_ptr_t& tensor_ir) {
  Qnn_QuantizeParams_t ret_quantize_params = QNN_QUANTIZE_PARAMS_INIT;

  auto& mllm_tensor = tensor_ir->tensor_;

  switch (mllm_tensor.dtype()) {
    case kInt32:
    case kFp32:
    case kFp16: break;
    case kPTInt8_Sym: {
      ret_quantize_params.encodingDefinition = QNN_DEFINITION_DEFINED;
      ret_quantize_params.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
      auto scale = mllm_tensor.getExtraTensorViewInTensor("scale").item<float>();
      ret_quantize_params.scaleOffsetEncoding = {
          .scale = scale,
          .offset = 0,
      };
      break;
    }
    default: NYI("Not supported data type {}", (int)mllm_tensor.dtype()); break;
  }

  return ret_quantize_params;
}

Qnn_TensorMemType_t QnnTensorTransform::autoQnnTensorMemType(
    const ir::tensor::TensorValue::self_ptr_t& tensor_ir) {
  Qnn_TensorMemType_t ret_mem_type = QNN_TENSORMEMTYPE_RAW;

  // FIXME: handle others.

  return ret_mem_type;
}

Qnn_TensorType_t QnnTensorTransform::autoQnnTensorType(Tensor& mllm_tensor) {
  switch (mllm_tensor.memType()) {
    case kExtraInput: return QNN_TENSOR_TYPE_APP_WRITE;
    case kExtraOutput: return QNN_TENSOR_TYPE_APP_READ;
    case kParams: return QNN_TENSOR_TYPE_STATIC;
    case kQnnAppReadWrite: return QNN_TENSOR_TYPE_APP_READWRITE;
    default: return QNN_TENSOR_TYPE_NATIVE;
  }

  // FIXME: Handle Static and changeable Qnn quantized tensors.

  return QNN_TENSOR_TYPE_NATIVE;
}

Qnn_TensorDataFormat_t QnnTensorTransform::autoQnnTensorDataFormat(
    Tensor& mllm_tensor) {  // FIXME: Handle other things
  return QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
}

Qnn_DataType_t QnnTensorTransform::autoQnnTensorDataType(Tensor& mllm_tensor) {
  Qnn_DataType_t ret_qnn_dtype;

  switch (mllm_tensor.dtype()) {
    case kInt32: ret_qnn_dtype = QNN_DATATYPE_INT_32; break;
    case kFp32: ret_qnn_dtype = QNN_DATATYPE_FLOAT_32; break;
    case kFp16: ret_qnn_dtype = QNN_DATATYPE_FLOAT_16; break;
    case kPTInt8_Sym: ret_qnn_dtype = QNN_DATATYPE_SFIXED_POINT_8; break;
    default: NYI("Not supported data type {}", (int)mllm_tensor.dtype()); break;
  }

  return ret_qnn_dtype;
}

Qnn_QuantizeParams_t QnnTensorTransform::autoQnnTensorQuantParams(Tensor& mllm_tensor) {
  Qnn_QuantizeParams_t ret_quantize_params = QNN_QUANTIZE_PARAMS_INIT;

  switch (mllm_tensor.dtype()) {
    case kInt32:
    case kFp32:
    case kFp16: break;
    case kPTInt8_Sym: {
      ret_quantize_params.encodingDefinition = QNN_DEFINITION_DEFINED;
      ret_quantize_params.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
      auto scale = mllm_tensor.getExtraTensorViewInTensor("scale").item<float>();
      ret_quantize_params.scaleOffsetEncoding = {
          .scale = scale,
          .offset = 0,
      };
      break;
    }
    default: NYI("Not supported data type {}", (int)mllm_tensor.dtype()); break;
  }

  return ret_quantize_params;
}

Qnn_TensorMemType_t QnnTensorTransform::autoQnnTensorMemType(Tensor& mllm_tensor) {
  Qnn_TensorMemType_t ret_mem_type = QNN_TENSORMEMTYPE_RAW;

  // FIXME: handle others.

  return ret_mem_type;
}

}  // namespace mllm::qnn
