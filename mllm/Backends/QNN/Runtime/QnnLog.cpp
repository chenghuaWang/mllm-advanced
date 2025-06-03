/**
 * @file QnnLog.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-06-03
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Backends/QNN/Runtime/QnnLog.hpp"

namespace mllm::qnn {

void __mllmLoggerCallback4QnnLogger(const char* fmt, QnnLog_Level_t level, uint64_t times_tamp,
                                    va_list argp) {
  const char* level_str = "";
  switch (level) {
    case QNN_LOG_LEVEL_ERROR: level_str = "[ERROR]"; break;
    case QNN_LOG_LEVEL_WARN: level_str = "[WARN]"; break;
    case QNN_LOG_LEVEL_INFO: level_str = "[INFO]"; break;
    case QNN_LOG_LEVEL_DEBUG: level_str = "[DEBUG]"; break;
    case QNN_LOG_LEVEL_VERBOSE: level_str = "[VERBOSE]"; break;
    case QNN_LOG_LEVEL_MAX: level_str = "[UNKNOWN]"; break;
  }

  double ms = (double)times_tamp / 1000000.0;

  {
    fprintf(stdout, "QnnLogger(%8.1fms) %s: ", ms, level_str);
    vfprintf(stdout, fmt, argp);
  }
}

QnnLogger::QnnLogger(QnnLog_Callback_t callback, QnnLog_Level_t max_log_level,
                     QnnLog_Error_t* status)
    : callback_(callback), max_log_level_(max_log_level), epoch_(get_time_stamp()) {
  if (!callback) { callback_ = __mllmLoggerCallback4QnnLogger; }
}

}  // namespace mllm::qnn
