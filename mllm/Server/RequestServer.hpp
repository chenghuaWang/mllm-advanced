/**
 * @file RequestServer.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-05-16
 *
 * @copyright Copyright (c) 2025
 *
 */
/// There is no need to use Message Queue. Mllm always process one request at a time.
#pragma once

#include <memory>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <nlohmann/json.hpp>
#include "mllm/Server/Generator.hpp"

namespace mllm {

/// SSE Stream Server.
class MllmRequestServer {
  using json = nlohmann::json;

  int port_ = 9808;
  std::shared_ptr<MllmStreamModelGenerator> model_ = nullptr;
  std::atomic<bool> running_;
  std::thread server_thread_;

  std::mutex request_working_mutex_;
  std::atomic<bool> processing_;

 public:
  explicit MllmRequestServer(int port) : port_(port), running_(false), processing_(false) {}

  void start();

  void stop();

 private:
  int createSocket();

  void setupSocket(int server_fd);

  void handleConnection(int client_socket);

  void processStreamRequest(int socket, const json& request);

  std::string buildPrompt(const json& messages);

  void sendResponseHeaders(int socket);

  void sendStreamChunk(int socket, const std::string& content, const std::string& type);

  std::pair<std::string, std::string> parseRequest(int socket);

  void sendErrorResponse(int socket, int code, const std::string& message);

  void acceptConnections(int server_fd);

  void run();
};

}  // namespace mllm
