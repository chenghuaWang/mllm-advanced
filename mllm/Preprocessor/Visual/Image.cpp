/**
 * @file Image.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-04-13
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <algorithm>
#include <filesystem>
#include "mllm/Preprocessor/Visual/Image.hpp"

namespace mllm {

_ImagePtr::~_ImagePtr() { stbi_image_free(ptr_); }

Image Image::open(const std::string& fp) {
  Image ret_image;

  if (!stbi_info(fp.c_str(), &ret_image.w_, &ret_image.h_, &ret_image.c_)) {
    MLLM_ERROR_EXIT(kError, "Can't get infomation of image: {}", fp);
  }

  if (ret_image.c_ > 4) {
    MLLM_WARN("Found image: {} has {} channel. MLLM Image::open method will force convert this "
              "image to RGB.",
              fp, ret_image.c_);
  }

  auto _ptr = stbi_load(fp.c_str(), &ret_image.w_, &ret_image.h_, &ret_image.c_, 3);

  if (_ptr) {
    ret_image.image_ptr_ = std::make_shared<_ImagePtr>();
    ret_image.image_ptr_->ptr_ = _ptr;
  } else {
    MLLM_ERROR_EXIT(kError, "stbi_load load error. Found i_data is empty pointer!");
  }

  return ret_image;
}

Image Image::resize(int new_w, int new_h) {
  Image new_img;
  new_img.w_ = new_w;
  new_img.h_ = new_h;
  new_img.c_ = 3;

  unsigned char* output_data = nullptr;

  // stb will alloc memory for us
  stbir_resize_uint8_linear(static_cast<unsigned char*>(image_ptr_->ptr_), w_, h_, 0, output_data,
                            new_w, new_h, 0, STBIR_RGB);

  new_img.image_ptr_ = std::make_shared<_ImagePtr>();
  new_img.image_ptr_->ptr_ = output_data;
  return new_img;
}

void Image::save(const std::string& fp) {
  namespace fs = std::filesystem;

  fs::path file_path(fp);

  std::string ext = file_path.extension().string();

  if (!ext.empty() && ext[0] == '.') { ext = ext.substr(1); }
  std::ranges::transform(ext, ext.begin(), ::tolower);

  if (ext == "png") {
    stbi_write_png(fp.c_str(), w_, h_, c_, image_ptr_->ptr_, w_ * c_);
  } else if (ext == "jpg" || ext == "jpeg") {
    stbi_write_jpg(fp.c_str(), w_, h_, c_, image_ptr_->ptr_, 95);
  } else if (ext == "bmp") {
    stbi_write_bmp(fp.c_str(), w_, h_, c_, image_ptr_->ptr_);
  } else {
    MLLM_ERROR_EXIT(kError, "Unsupported image format: {}", ext);
  }
}

Tensor Image::tensor() {
  auto ret_tensor = Tensor::empty({c_, h_, w_}, kFp32, kCPU).alloc();

  auto bare_tensor_ptr = ret_tensor.ptr<float>();
  auto bare_stb_ptr = static_cast<unsigned char*>(image_ptr_->ptr_);

  // H, W, C -> C, H, W
  for (int i_h = 0; i_h < h_; ++i_h) {
    for (int i_w = 0; i_w < w_; ++i_w) {
      int src_index = (i_h * w_ + i_w) * 3;
      int dst_index = (i_h * w_ + i_w) * 3;

      bare_tensor_ptr[dst_index] = static_cast<float>(bare_stb_ptr[src_index]) / 255.0f;
      bare_tensor_ptr[dst_index + 1] = static_cast<float>(bare_stb_ptr[src_index + 1]) / 255.0f;
      bare_tensor_ptr[dst_index + 2] = static_cast<float>(bare_stb_ptr[src_index + 2]) / 255.0f;
    }
  }

  return ret_tensor;
}

unsigned char* Image::ptr() {
  if (!image_ptr_) return nullptr;
  return static_cast<unsigned char*>(image_ptr_->ptr_);
}

int Image::w() { return w_; }

int Image::h() { return h_; }

int Image::c() { return c_; }

}  // namespace mllm
