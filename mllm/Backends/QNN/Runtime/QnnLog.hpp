/**
 * @file QnnLog.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-03
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <cstdint>
#include <atomic>
#include <string>
#include <chrono>

#include <QNN/QnnLog.h>

namespace mllm::qnn {

void __mllmLoggerCallback4QnnLogger(const char* fmt, QnnLog_Level_t level, uint64_t times_tamp,
                                    va_list argp);

class QnnLogger final {
 public:
  static QnnLogger& instance() {
    static QnnLogger instance{nullptr, QNN_LOG_LEVEL_INFO, nullptr};
    return instance;
  }

  explicit QnnLogger(QnnLog_Callback_t callback = nullptr,
                     QnnLog_Level_t max_log_level = QNN_LOG_LEVEL_INFO,
                     QnnLog_Error_t* status = nullptr);

  QnnLogger() = default;

  QnnLogger(const QnnLogger&) = delete;

  QnnLogger& operator=(const QnnLogger&) = delete;

  inline void set_max_log_level(QnnLog_Level_t max_log_level) {
    max_log_level_.store(max_log_level, std::memory_order_seq_cst);
  }

  inline QnnLog_Level_t getMaxLevel() { return max_log_level_.load(std::memory_order_seq_cst); }

  inline QnnLog_Callback_t getLogCallback() { return callback_; }

  inline void log(QnnLog_Level_t level, const char* file, int64_t line, const char* fmt, ...) {
    if (callback_) {
      if (level > max_log_level_.load(std::memory_order_seq_cst)) { return; }
      va_list argp;
      va_start(argp, fmt);
      std::string log_str(fmt);
      std::ignore = file;
      std::ignore = line;
      (*callback_)(log_str.c_str(), level, get_time_stamp() - epoch_, argp);
      va_end(argp);
    }
  }

 private:
  inline uint64_t get_time_stamp() const {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
  }

  QnnLog_Callback_t callback_;
  std::atomic<QnnLog_Level_t> max_log_level_;
  uint64_t epoch_;
};

}  // namespace mllm::qnn