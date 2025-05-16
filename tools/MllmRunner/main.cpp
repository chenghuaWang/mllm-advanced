/**
 * @file main.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-05-16
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <iostream>
#include "mllm/Utils/Argparse.hpp"
#include "mllm/Server/RequestServer.hpp"

using namespace mllm;  // NOLINT

int main(int argc, char* argv[]) {
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message.");
  auto& port = Argparse::add<int>("-p|--port").help("Port to listen on.").meta("PORT");
  auto& model_tag =
      Argparse::add<std::string>("-m|--model").help("Model tag to use.").meta("MODEL");
  auto& model_param_path = Argparse::add<std::string>("-mp|--model-param-path")
                               .help("Model param path.")
                               .meta("MODEL_PARAM_PATH");
  auto& model_vocab_path = Argparse::add<std::string>("-vp|--model-vocab-path")
                               .help("Model vocab path.")
                               .meta("MODEL_VOCAB_PATH");

  Argparse::parse(argc, argv);

  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }

  MllmRequestServer server(port.get());
  server.start();
  MLLM_INFO("Press q to quit.");
  char ch;
  while (std::cin >> ch) {
    if (ch == 'q') {
      server.stop();
      break;
    }
  }
  MLLM_INFO("BYE");
}
