/**
 * @file Dumpper.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-01-29
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "mllm/Utils/DumpPrinter.hpp"

namespace mllm {

DumpPrinter::DumpPrinter(int32_t indent) : indent_(indent) {}

void DumpPrinter::inc() { depth_++; }

void DumpPrinter::dec() { depth_--; }

DumpPrinterGuard::DumpPrinterGuard(DumpPrinter& printer) : printer_(printer) { printer_.inc(); }

DumpPrinterGuard::~DumpPrinterGuard() { printer_.dec(); }

}  // namespace mllm
