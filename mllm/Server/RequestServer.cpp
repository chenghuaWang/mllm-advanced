/**
 * @file RequestServer.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-05-16
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Engine/Context.hpp"
#include "mllm/Utils/Common.hpp"
#include "mllm/Server/RequestServer.hpp"
#include "mllm/Preprocessor/Tokenizers/Unicode.hpp"

namespace mllm {
void MllmRequestServer::start() {
  running_ = true;
  server_thread_ = std::thread([this] { run(); });

  MLLM_INFO("Mllm Runner start at port {}", port_);
}

void MllmRequestServer::stop() {
  running_ = false;
  if (server_thread_.joinable()) { server_thread_.join(); }
}

int MllmRequestServer::createSocket() {
  int server_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (server_fd < 0) {
    MLLM_ERROR_EXIT(kError, "Socket creation failed");
    return -1;
  }
  return server_fd;
}

void MllmRequestServer::setupSocket(int server_fd) {
  int opt = 1;
  setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = INADDR_ANY;
  address.sin_port = htons(port_);

  if (bind(server_fd, (sockaddr*)&address, sizeof(address)) < 0) {
    MLLM_ERROR_EXIT(kError, "Bind failed");
    close(server_fd);
    return;
  }

  if (listen(server_fd, 5) < 0) {
    MLLM_ERROR_EXIT(kError, "Listen failed");
    close(server_fd);
    return;
  }
}

void MllmRequestServer::handleConnection(int client_socket) {
  std::string headers, body;
  std::tie(headers, body) = parseRequest(client_socket);

  if (headers.find("POST /v1/chat/completions HTTP/1.1") == std::string::npos) {
    sendErrorResponse(client_socket, 404, "Not Found");
    return;
  }

  // Mllm can only process request only once.
  {
    std::lock_guard<std::mutex> lock(request_working_mutex_);
    if (processing_) {
      sendErrorResponse(client_socket, 429, "Too Many Requests");
      return;
    }
    processing_ = true;
  }

  try {
    json request = json::parse(body);
    processStreamRequest(client_socket, request);
  } catch (const json::exception&) { sendErrorResponse(client_socket, 400, "Invalid JSON"); }

  {
    std::lock_guard<std::mutex> lock(request_working_mutex_);
    processing_ = false;
  }
}

void MllmRequestServer::processStreamRequest(int socket, const json& request) {
  // Set SSE Head
  sendResponseHeaders(socket);

  std::string model = request.value("model", "default");
  std::string prompt = buildPrompt(request["messages"]);
  float temperature = request.value("temperature", 0.7f);
  int max_tokens = request.value("max_tokens", 100);

  MLLM_RT_ASSERT(model == model_tag_);

  // TODO process image

  model_->generate(model_->encode(prompt), max_tokens, [&](const std::wstring& output) {
    sendStreamChunk(socket, preprocessor::wideString2Utf8String(output), "");
  });

  sendStreamChunk(socket, "", "done");
}

std::string MllmRequestServer::buildPrompt(const json& messages) {
  std::vector<std::pair<std::string, std::string>> prompt_parts;
  prompt_parts.reserve(messages.size());
  for (const auto& msg : messages) {
    prompt_parts.emplace_back(msg["role"].get<std::string>(), msg["content"].get<std::string>());
  }
  return model_->buildPrompt(prompt_parts);
}

void MllmRequestServer::sendResponseHeaders(int socket) {
  std::string headers = "HTTP/1.1 200 OK\r\n"
                        "Content-Type: text/event-stream\r\n"
                        "Cache-Control: no-cache\r\n"
                        "Connection: keep-alive\r\n"
                        "Access-Control-Allow-Origin: *\r\n\r\n";

  send(socket, headers.c_str(), headers.size(), 0);
}

void MllmRequestServer::sendStreamChunk(int socket, const std::string& content,
                                        const std::string& type) {
  // Get current time in ISO 8601 format
  auto now = std::time(nullptr);
  auto tm_info = std::localtime(&now);
  char buffer[26];
  std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%S", tm_info);

  json chunk = {{"id", "chatcmpl-" + std::to_string(rand())},  // TODO
                {"object", "chat.completion.chunk"},
                {"created", buffer},
                {"model", model_tag_},
                {"choices",
                 {{{"delta", {{"content", content}}},
                   {"finish_reason", type == "done" ? "stop" : ""},
                   {"index", 0}}}}};  // TODO

  std::string event = "data: " + chunk.dump() + "\n\n";
  send(socket, event.c_str(), event.size(), 0);
}

std::pair<std::string, std::string> MllmRequestServer::parseRequest(int socket) {
  constexpr int BUFFER_SIZE = 4096;
  char buffer[BUFFER_SIZE];
  std::string request;

  ssize_t bytes_read = recv(socket, buffer, BUFFER_SIZE, 0);
  if (bytes_read > 0) { request.append(buffer, bytes_read); }

  size_t header_end = request.find("\r\n\r\n");
  if (header_end == std::string::npos) return {"", ""};

  return {request.substr(0, header_end), request.substr(header_end + 4)};
}

void MllmRequestServer::sendErrorResponse(int socket, int code, const std::string& message) {
  json error = {{"error", {{"code", code}, {"message", message}}}};
  std::string response = "HTTP/1.1 " + std::to_string(code)
                         + " Error\r\n"
                           "Content-Type: application/json\r\n"
                           "Content-Length: "
                         + std::to_string(error.dump().size())
                         + "\r\n"
                           "Connection: close\r\n\r\n"
                         + error.dump();

  send(socket, response.c_str(), response.size(), 0);
}

void MllmRequestServer::acceptConnections(int server_fd) {
  while (running_) {
    sockaddr_in client_addr{};
    socklen_t addr_len = sizeof(client_addr);

    int client_socket = accept(server_fd, (sockaddr*)&client_addr, &addr_len);
    if (client_socket < 0) continue;

    handleConnection(client_socket);
    close(client_socket);
  }
}

void MllmRequestServer::run() {
  // clone model to this thread
  auto& ctx = MllmEngineCtx::instance();
  ctx.thisThread()->layer_ops_table = ctx.mainThread()->layer_ops_table;

  int server_fd = createSocket();
  if (server_fd < 0) return;
  setupSocket(server_fd);
  acceptConnections(server_fd);
  close(server_fd);
}

}  // namespace mllm
