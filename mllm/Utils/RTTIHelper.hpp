/**
 * @file RTTIHelper.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-16
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

// ref: https://llvm.org/docs/HowToSetUpLLVMStyleRTTI.html

#include <cassert>
#include <memory>
#include <type_traits>

namespace mllm {

/**
 * @brief isa Implement trait
 *
 * @tparam To
 * @tparam From
 * @tparam Enabler: std::enable_if_t<std::is_base_of<To, From>::value>
 */
template<typename To, typename From, typename Enabler = void>
struct isaImpl {
  static inline bool exec(const From& val) { return To::classof(&val); }
};

// upcast(child class -> father class) always true.
template<typename To, typename From>
struct isaImpl<To, From, std::enable_if_t<std::is_base_of<To, From>::value>> {
  static inline bool exec(const From& val) { return true; }
};

// const From
template<typename To, typename From>
struct isaImpl<To, const From> {
  static inline bool exec(const From& val) { return isaImpl<To, From>::exec(val); }
};

// From*.
template<typename To, typename From>
struct isaImpl<To, From*> {
  static inline bool exec(const From* val) {
    assert(val && "nullptr passed to isa");
    return val ? isaImpl<To, From>::exec(*val) : false;
  }
};

// unique_ptr<From>
template<typename To, typename From>
struct isaImpl<To, std::unique_ptr<From>> {
  static inline bool exec(const std::unique_ptr<From>& val) {
    assert(val && "nullptr[unique_ptr] passed to isa");
    return val ? isaImpl<To, From>::exec(*val) : false;
  }
};

// shared_ptr<From>
template<typename To, typename From>
struct isaImpl<To, std::shared_ptr<From>> {
  static inline bool exec(const std::shared_ptr<From>& val) {
    assert(val && "nullptr[shared_ptr] passed to isa");
    return val ? isaImpl<To, From>::exec(*val) : false;
  }
};

// isa
template<typename To, typename From>
bool isa(const From& val) {
  return isaImpl<To, From>::exec(val);
}

/**
 * @brief Cast Impl. From Value to Value.
 *
 * @tparam To
 * @tparam From
 */
template<typename To, typename From>
struct castImpl {
  static inline To& exec(const From& val) {
    assert(isa<To>(val) && "not castable");
    return *(To*)(&val);
  }
};

// ptr 2 ptr
template<typename To, typename From>
struct castImpl<To*, From*> {
  static inline To* exec(From* val) {
    assert(isa<To>(val) && "not castable");
    return static_cast<To*>(val);
  }
};

// unique_ptr to unique_ptr
template<typename To, typename From>
struct castImpl<std::unique_ptr<To>, std::unique_ptr<From>> {
  static inline std::unique_ptr<To> exec(std::unique_ptr<From> val) {
    assert(isa<To>(val) && "not castable");
    return std::unique_ptr<To>(static_cast<To*>(val.release()));
  }
};

// casts
template<typename To, typename From>
auto cast(From val) ->
    typename std::enable_if<!std::is_pointer<From>::value
                                && !std::is_same<std::unique_ptr<To>, From>::value,
                            To>::type {
  return castImpl<To, From>::exec(val);
}

template<typename To, typename From>
auto cast(From val) ->
    typename std::enable_if<std::is_pointer<From>::value
                                && !std::is_same<std::unique_ptr<To>, From>::value,
                            To*>::type {
  assert(isa<To>(val) && "not castable");
  return static_cast<To*>(val);
}

template<typename To, typename From>
std::shared_ptr<To> cast(std::shared_ptr<From>& val) {
  assert(isa<To>(val) && "not castable");
  return std::static_pointer_cast<To>(val);
}

template<typename To, typename From>
std::shared_ptr<To> cast(const std::shared_ptr<From>& val) {
  assert(isa<To>(val) && "not castable");
  return std::static_pointer_cast<To>(val);
}

template<typename To, typename From>
auto cast(From val) -> typename std::enable_if<std::is_same<std::unique_ptr<To>, From>::value,
                                               std::unique_ptr<To>>::type {
  return castImpl<To, From>::exec(std::move(val));
}

template<typename To, typename From>
auto cast(From val) ->
    typename std::enable_if<!std::is_pointer<From>::value
                                && std::is_same<std::unique_ptr<To>, From>::value,
                            To*>::type {
  return castImpl<To, From>::exec(val);
}

}  // namespace mllm