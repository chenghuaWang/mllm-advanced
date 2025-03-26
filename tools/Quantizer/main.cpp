/**
 * @file main.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-03
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <memory>
#include <vector>
#include "fmt/base.h"
#include "fmt/ranges.h"
#include "mllm/Core/DataTypes.hpp"
#include "mllm/Engine/ParameterReader.hpp"
#include "mllm/Utils/Argparse.hpp"

using namespace mllm;

static void printLoaderMetaData(std::shared_ptr<ParameterLoader>& p) {
  std::vector<std::shared_ptr<TensorViewImpl>> _p{p->params().size(), nullptr};
  for (auto& item : p->params()) { _p[item.second->uuid()] = item.second; }
  for (auto& param : _p) {
    fmt::println("id: {}, name: {}, type: {}, shape: {}", param->uuid(), param->name(),
                 dataTypes2Str(param->dtype()), fmt::join(param->shape(), "x"));
  }
}

int main(int argc, char* argv[]) {
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& input_files =
      Argparse::add<std::string>("-i|--input").help("Input file path").meta("FILE").positional();
  auto& show_parmas = Argparse::add<bool>("-s|--show").help("Show parameters meta data");

  Argparse::parse(argc, argv);

  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }

  if (!input_files.isSet()) {
    MLLM_ERROR_EXIT(kError, "No input file provided");
    Argparse::printHelp();
    return -1;
  }

  auto loader = load(input_files.get());

  if (show_parmas.isSet()) { printLoaderMetaData(loader); }
}
