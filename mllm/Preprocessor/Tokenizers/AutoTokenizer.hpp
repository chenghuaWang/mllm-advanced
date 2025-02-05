/**
 * @file AutoTokenizer.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-05
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

// Ths json lib is head only. Include all functionall <nlohmann/json.hpp> will increase the compile
// time. Hence, <nlohmann/json_fwd.hpp> is provided for decrease compile time.
//
// json_fwd.hpp:
// Used for forward declaring the nlohmann::json type, suitable for scenarios where only the type
// needs to be declared, reducing compilation time and dependencies.
//
// json.hpp:
// Contains the full implementation of the JSON library, suitable for scenarios where JSON data
// needs to be manipulated.
#include <nlohmann/json_fwd.hpp>
using json = nlohmann::json;

namespace mllm::preprocessor {}
