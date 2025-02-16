/**
 * @file Node.cpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2025-02-16
 *
 * @copyright Copyright (c) 2025
 *
 */
#include <algorithm>
#include <memory>
#include "mllm/IR/Node.hpp"
#include "mllm/IR/GeneratedRTTIKind.hpp"
#include "mllm/Utils/IRPrinter.hpp"
#include "mllm/Utils/RTTIHelper.hpp"

namespace mllm::ir {

Node::Node(const NodeKind& kind) : kind_(kind) {}

std::list<node_ptr_t>& Node::inputs() { return inputs_; }

std::list<node_ptr_t>& Node::outputs() { return outputs_; }

node_ptr_t Node::prevOp() { return prev_op_node_; }

node_ptr_t Node::nextOp() { return next_op_node_; }

node_ptr_t Node::belongsTo() { return belongs_to_parent_; }

void Node::setAttr(const std::string& str, const attr_ptr_t& attr) { attrs_.insert({str, attr}); }

attr_ptr_t Node::getAttr(const std::string& str) {
  auto it = attrs_.find(str);
  return it != attrs_.end() ? it->second : nullptr;
}

Region::Region(const op_ptr_t& belongs_to) : belongs_to_(belongs_to) {}

std::list<op_ptr_t>& Region::ops() { return ops_; }

std::list<val_ptr_t>& Region::inputs() { return inputs_; }

std::list<val_ptr_t>& Region::outputs() { return outputs_; }

op_ptr_t& Region::belongsTo() { return belongs_to_; }

void Region::dump(IRPrinter& p) {
  // inputs
  {
    IRPrinter::lparentheses();
    size_t cnt = 0;
    auto size = inputs_.size();
    for (auto& ins : inputs_) {
      ins->dump(p);
      if (cnt < size - 1) IRPrinter::comma();
      cnt++;
    }
    IRPrinter::rparentheses();
  }

  // to
  IRPrinter::to();

  // outputs
  {
    IRPrinter::lparentheses();
    size_t cnt = 0;
    auto size = outputs_.size();
    for (auto& ous : outputs_) {
      ous->dump(p);
      if (cnt < size - 1) IRPrinter::comma();
      cnt++;
    }
    IRPrinter::rparentheses();
  }

  p.lbrace();

  // ops
  {
    size_t cnt = 0;
    auto size = ops_.size();
    for (auto& op : ops_) {
      op->dump(p);
      if (cnt < size - 1) p.newline();
      cnt++;
    }
  }

  p.rbrace();
}

Op::~Op() = default;

Op::Op() : Node(RK_Op) {};

Op::Op(const NodeKind& kind) : Node(kind) {};

void Op::dump(IRPrinter& p) {
  // inputs
  {
    IRPrinter::lparentheses();
    size_t cnt = 0;
    auto size = inputs().size();
    for (auto& ins : inputs()) {
      ins->dump(p);
      if (cnt < size - 1) IRPrinter::comma();
      cnt++;
    }
    IRPrinter::rparentheses();
  }

  // to
  IRPrinter::to();

  // outputs
  {
    IRPrinter::lparentheses();
    size_t cnt = 0;
    auto size = outputs().size();
    for (auto& ous : outputs()) {
      ous->dump(p);
      if (cnt < size - 1) IRPrinter::comma();
      cnt++;
    }
    IRPrinter::rparentheses();
  }
}

std::shared_ptr<Region> Op::createRegionAtTop() {
  auto reg = std::make_shared<Region>(cast<Op>(shared_from_this()));
  regions_.push_back(reg);
  return reg;
}

std::list<std::shared_ptr<Region>>& Op::regions() { return regions_; }

std::shared_ptr<Region> Op::getTopRegion() { return regions_.back(); }

Attr::~Attr() = default;

Attr::Attr() : Node(RK_Attr) {};

Attr::Attr(const NodeKind& kind) : Node(kind) {}

void Attr::dump(IRPrinter& p) { p.print("<InvalidAttribute NYI>"); }

Val::~Val() = default;

Val::Val() : Node(RK_Val) {};

Val::Val(const NodeKind& kind) : Node(kind) {}

void Val::dump(IRPrinter& p) { p.print("%{}:", name()); }

std::string& Val::name() { return name_; }

IRContext::IRContext(const node_ptr_t& module, const region_ptr_t& region)
    : top_level_op_(module), cur_insert_region_(region) {}

void IRContext::setValueName(const val_ptr_t& val, const std::string& name) {
  value_names_[val] = name;
}

std::string IRContext::getAutoIndexedValueName() {
  return std::to_string(auto_indexed_value_name_cnt_++);
}

void IRContext::resetRegion(const region_ptr_t& region) { cur_insert_region_ = region; }

const IRContext::region_ptr_t& IRContext::getCurInsertRegion() { return cur_insert_region_; }

node_ptr_t& IRContext::topLevelOp() { return top_level_op_; }

void IRContext::addToSymbolTable(const node_ptr_t& node, const std::string& name) {
  symbol_table_[name] = node;
}

node_ptr_t IRContext::lookupSymbolTable(const std::string& name) {
  return symbol_table_.count(name) ? symbol_table_[name] : nullptr;
}

void IRContext::setDevice(const std::string& device_name) { device_name_ = device_name; }

std::string IRContext::getDevice() { return device_name_; }

IRWriterGuard::IRWriterGuard(const std::shared_ptr<IRContext>& ctx,
                             const std::shared_ptr<Region>& new_region)
    : ctx_(ctx), new_region_(new_region) {
  old_region_ = ctx->getCurInsertRegion();
  ctx->resetRegion(new_region_);
}

IRWriterGuard::~IRWriterGuard() { ctx_->resetRegion(old_region_); }

IRWriter::IRWriter(const std::shared_ptr<IRContext>& ctx, const std::shared_ptr<Region>& cur_region)
    : ctx_(ctx), cur_region_(cur_region) {}

void IRWriter::removeOp(const op_ptr_t& op) {
  auto& ops = cur_region_->ops();

  cur_op_iter_ = ops.erase(std::find(ops.begin(), ops.end(), op));

  is_iterator_modified_ = true;
}

void IRWriter::replaceOp(const op_ptr_t& old_op, const op_ptr_t& new_op) {
  auto& ops = cur_region_->ops();
  std::replace(ops.begin(), ops.end(), old_op, new_op->template cast_<Op>());
}

}  // namespace mllm::ir