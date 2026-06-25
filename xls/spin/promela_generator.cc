// Copyright 2026 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xls/spin/promela_generator.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/substitute.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/channel.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/proc_instantiation.h"
#include "xls/ir/source_location.h"
#include "xls/ir/state_element.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls::spin {
namespace {

bool IsUnitType(Type* type) {
  return type->IsTuple() && type->AsTupleOrDie()->size() == 0;
}

std::string_view PromelaType(Type* type) {
  if (type->IsToken() || IsUnitType(type)) {
    return "";
  }
  if (type->IsBits()) {
    const int64_t bits = type->AsBitsOrDie()->bit_count();
    if (bits == 1) {
      return "bit";
    }
    if (bits <= 8) {
      return "byte";
    }
    if (bits <= 16) {
      return "short";
    }
    return "int";
  }
  LOG(WARNING) << "Type '" << type->ToString()
               << "' has no Promela equivalent; approximated as int (lossy).";
  return "int";
}

std::string SanitizeName(std::string_view name) {
  std::string result;
  result.reserve(name.size());
  if (!result.empty() && absl::ascii_isdigit(result[0])) {
    result.insert(result.begin(), '_');
  }
  for (char c : name) {
    result.push_back((absl::ascii_isalnum(c) || c == '_') ? c : '_');
  }
  return result;
}

std::string BitsLiteralStr(const Value& value) {
  if (!value.IsBits()) {
    return "0";
  }
  auto uint_value = value.bits().ToUint64();
  return uint_value.ok() ? absl::StrCat(*uint_value) : "0 /* oversized */";
}

bool HasBodyOps(Proc* proc) {
  return absl::c_any_of(proc->nodes(),
                        [](Node* n) {
                          return n->op() == Op::kReceive ||
                                 n->op() == Op::kSend || n->op() == Op::kAssert;
                        }) ||
         absl::c_any_of(proc->StateElements(), [](StateElement* se) {
           return !se->type()->IsToken() && !IsUnitType(se->type());
         });
}

std::vector<std::string> ProcParams(Proc* proc) {
  std::vector<std::string> params;
  absl::flat_hash_set<std::string> seen;
  for (ChannelInterface* iface : proc->interface()) {
    if (seen.emplace(iface->name()).second) {
      params.push_back(std::string(iface->name()));
    }
  }
  return params;
}

}  // namespace

PromelaGenerator::PromelaGenerator(Package* package,
                                   const PromelaGeneratorOptions& options)
    : package_(package), options_(options), emit_(out_) {}

absl::StatusOr<std::string> PromelaGenerator::Generate(
    Package* package, const PromelaGeneratorOptions& options) {
  PromelaGenerator gen(package, options);

  LOG(INFO) << "Generating Promela for package '" << package->name()
            << "': " << package->functions().size() << " function(s), "
            << package->procs().size() << " proc(s)";

  if (!package->procs().empty() && !package->ChannelsAreProcScoped()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Package '", package->name(),
                     "' has procs with old-style (package-level) channels. "
                     "Re-generate the IR with proc-scoped channels enabled "
                     "(lower_to_proc_scoped_channels pass)."));
  }

  XLS_RETURN_IF_ERROR(gen.ValidateTypes());

  gen.emit_.Line("/* Promela model generated from XLS IR package: $0 */",
                 package->name());
  gen.emit_.Blank();

  if (options.emit_termination_hook) {
    gen.emit_.Line("bit __terminated = 0;");
    gen.emit_.Blank();
  }

  for (const auto& fn : package->functions()) {
    XLS_RETURN_IF_ERROR(gen.EmitFunction(fn.get()));
  }

  for (const auto& proc : package->procs()) {
    if (!proc->is_new_style_proc()) {
      continue;
    }
    if (!HasBodyOps(proc.get()) && proc->channels().empty() &&
        proc->proc_instantiations().empty()) {
      continue;
    }
    XLS_RETURN_IF_ERROR(gen.EmitProc(proc.get()));
  }

  std::vector<Proc*> roots = gen.FindRootProcs();
  if (!roots.empty()) {
    gen.EmitInit(roots);
  }

  LOG(INFO) << "Promela generation complete";
  return gen.out_;
}

// Rejects any node or channel type wider than 32 bits (Promela int limit).
absl::Status PromelaGenerator::ValidateTypes() {
  auto check = [](Type* type, std::string_view ctx) -> absl::Status {
    if (type->IsBits() && type->AsBitsOrDie()->bit_count() > 32) {
      return absl::InvalidArgumentError(
          absl::StrCat(ctx, " has bit width ", type->AsBitsOrDie()->bit_count(),
                       " which exceeds Promela's maximum of 32 (int)."));
    }
    return absl::OkStatus();
  };
  for (const auto& fn : package_->functions()) {
    for (Node* node : fn->nodes()) {
      XLS_RETURN_IF_ERROR(check(
          node->GetType(), absl::StrCat("node '", node->GetName(),
                                        "' in function '", fn->name(), "'")));
    }
  }
  for (const auto& proc : package_->procs()) {
    for (Node* node : proc->nodes()) {
      XLS_RETURN_IF_ERROR(check(
          node->GetType(), absl::StrCat("node '", node->GetName(),
                                        "' in proc '", proc->name(), "'")));
    }
    for (ChannelInterface* i : proc->interface()) {
      XLS_RETURN_IF_ERROR(
          check(i->type(), absl::StrCat("channel '", i->name(), "' in proc '",
                                        proc->name(), "'")));
    }
    for (Channel* c : proc->channels()) {
      XLS_RETURN_IF_ERROR(
          check(c->type(), absl::StrCat("channel '", c->name(), "' in proc '",
                                        proc->name(), "'")));
    }
  }
  return absl::OkStatus();
}

// Returns procs that are not instantiated by any other proc (top-level roots).
std::vector<Proc*> PromelaGenerator::FindRootProcs() const {
  absl::flat_hash_set<Proc*> is_child;
  for (const auto& proc : package_->procs()) {
    for (const auto& instantiation : proc->proc_instantiations()) {
      is_child.insert(instantiation->proc());
    }
  }
  std::vector<Proc*> roots;
  for (const auto& proc : package_->procs()) {
    if (proc->is_new_style_proc() && !is_child.contains(proc.get())) {
      roots.push_back(proc.get());
    }
  }
  LOG(INFO) << "Found " << roots.size() << " root proc(s)";
  return roots;
}

// Emits a Promela inline macro for `fn`; result is passed via an out-parameter.
absl::Status PromelaGenerator::EmitFunction(Function* fn) {
  LOG(INFO) << "Emitting function '" << fn->name() << "'";
  std::vector<std::string> param_parts;
  for (Param* param : fn->params()) {
    param_parts.push_back(SanitizeName(param->name()));
  }
  param_parts.push_back("_ret");

  emit_.Line("inline fn_$0($1) {", SanitizeName(fn->name()),
             absl::StrJoin(param_parts, ", "));
  emit_.Indent();

  next_nodes_.clear();
  tuple_components_.clear();
  XLS_RETURN_IF_ERROR(fn->Accept(this));

  std::string_view ret_var = Ref(fn->return_value());
  if (!ret_var.empty()) {
    emit_.Line("_ret = $0;", ret_var);
  }
  emit_.Dedent();
  emit_.Line("}");
  emit_.Blank();
  return absl::OkStatus();
}

// Emits a Promela proctype for `proc`, including channel decls, child spawns,
// xr/xs hints, state variables, and the do/od main loop.
absl::Status PromelaGenerator::EmitProc(Proc* proc) {
  LOG(INFO) << "Emitting proc '" << proc->name() << "' (" << proc->node_count()
            << " node(s))";
  absl::flat_hash_set<std::string> send_channels, recv_channels;
  for (Node* node : proc->nodes()) {
    if (node->op() == Op::kSend) {
      send_channels.insert(std::string(node->As<Send>()->channel_name()));
    } else if (node->op() == Op::kReceive) {
      recv_channels.insert(std::string(node->As<Receive>()->channel_name()));
    }
  }

  std::vector<std::string> param_parts;
  for (const std::string& param_name : ProcParams(proc)) {
    param_parts.push_back(absl::StrCat("chan ", SanitizeName(param_name)));
  }
  emit_.Line("proctype $0($1) {", SanitizeName(proc->name()),
             absl::StrJoin(param_parts, "; "));
  emit_.Indent();

  for (Channel* channel : proc->channels()) {
    std::string_view elem_type = PromelaType(channel->type());
    if (elem_type.empty()) {
      elem_type = "bit";
    }
    emit_.Line("chan $0 = [$1] of { $2 };", SanitizeName(channel->name()),
               options_.channel_depth, elem_type);
  }
  if (!proc->channels().empty()) {
    emit_.Blank();
  }

  for (const auto& instantiation : proc->proc_instantiations()) {
    std::vector<std::string> args;
    for (ChannelInterface* channel_interface : instantiation->channel_args()) {
      args.push_back(SanitizeName(channel_interface->name()));
    }
    emit_.Line("run $0($1);", SanitizeName(instantiation->proc()->name()),
               absl::StrJoin(args, ", "));
  }
  if (!proc->proc_instantiations().empty()) {
    emit_.Blank();
  }

  // TODO: emit xs for send channels that are never polled (non-blocking receive)
  // by another proc; safe xs improves POR but requires proc-graph analysis.
  for (const std::string& channel_name : recv_channels) {
    emit_.Line("xr $0;", SanitizeName(channel_name));
  }
  if (!recv_channels.empty()) {
    emit_.Blank();
  }

  if (!HasBodyOps(proc)) {
    emit_.Dedent();
    emit_.Line("}");
    emit_.Blank();
    return absl::OkStatus();
  }

  for (StateElement* state_element : proc->StateElements()) {
    const std::string_view state_type = PromelaType(state_element->type());
    if (state_type.empty()) {
      continue;
    }
    emit_.Line("$0 s_$1 = $2;", state_type, SanitizeName(state_element->name()),
               BitsLiteralStr(state_element->initial_value()));
  }

  const std::string proc_key = SanitizeName(proc->name());
  auto throughput_it = options_.worst_case_throughput.find(proc_key);
  const int64_t throughput =
      (throughput_it != options_.worst_case_throughput.end())
          ? throughput_it->second
          : 1;
  const bool use_throughput = throughput > 1;

  if (use_throughput) {
    emit_.Line("$0 __thr = 0;", (throughput <= 256) ? "byte" : "short");
  }
  if (proc->GetStateElementCount() > 0 || use_throughput) {
    emit_.Blank();
  }

  emit_.Line("do");
  if (options_.emit_termination_hook) {
    emit_.Line(":: (__terminated) -> break");
  }
  if (use_throughput) {
    emit_.Line(":: (__thr > 0) -> __thr--;");
    emit_.Line(":: (__thr == 0) ->");
    emit_.Line("__thr = $0;", throughput - 1);
  } else {
    emit_.Line("::");
  }

  next_nodes_.clear();
  tuple_components_.clear();
  XLS_RETURN_IF_ERROR(proc->Accept(this));

  for (Node* node : next_nodes_) {
    auto* next_node = node->As<Next>();
    if (next_node->state_element()->type()->IsToken() ||
        IsUnitType(next_node->state_element()->type())) {
      continue;
    }
    const std::string state_var =
        absl::StrCat("s_", SanitizeName(next_node->state_element()->name()));
    std::string_view next_value = Ref(next_node->value());
    if (next_value.empty()) {
      next_value = "0";
    }
    if (next_node->predicate().has_value()) {
      std::string_view predicate = Ref(*next_node->predicate());
      if (predicate.empty()) {
        predicate = "0";
      }
      emit_.Line("if");
      emit_.Line(":: ($0 != 0) -> $1 = $2;", predicate, state_var, next_value);
      emit_.Line(":: else -> skip;");
      emit_.Line("fi");
    } else {
      emit_.Line("$0 = $1;", state_var, next_value);
    }
  }

  emit_.Line("od");
  emit_.Dedent();
  emit_.Line("}");
  emit_.Blank();
  return absl::OkStatus();
}

// Emits the Promela init block: declares interface channels and spawns roots.
void PromelaGenerator::EmitInit(const std::vector<Proc*>& root_procs) {
  LOG(INFO) << "Emitting init block for " << root_procs.size()
            << " root proc(s)";
  emit_.Line("init {");
  emit_.Indent();

  // absl::btree_map gives deterministic (sorted) iteration for channel decls.
  absl::btree_map<std::string, Type*> interface_channels;
  for (Proc* root_proc : root_procs) {
    for (ChannelInterface* i : root_proc->interface()) {
      interface_channels.emplace(SanitizeName(i->name()), i->type());
    }
  }
  for (const auto& [name, type] : interface_channels) {
    std::string_view elem_type = PromelaType(type);
    if (elem_type.empty()) {
      elem_type = "bit";
    }
    emit_.Line("chan $0 = [$1] of { $2 };", name, options_.channel_depth,
               elem_type);
  }
  if (!interface_channels.empty()) {
    emit_.Blank();
  }

  for (Proc* root_proc : root_procs) {
    std::vector<std::string> args;
    for (const std::string& param_name : ProcParams(root_proc)) {
      args.push_back(SanitizeName(param_name));
    }
    emit_.Line("run $0($1);", SanitizeName(root_proc->name()),
               absl::StrJoin(args, ", "));
  }

  if (options_.emit_termination_hook) {
    for (const auto& [name, unused] : interface_channels) {
      if (absl::StrContains(name, "terminator")) {
        emit_.Line("bit __term_val;");
        emit_.Line("$0 ? __term_val;", name);
        emit_.Line("__terminated = 1;");
        break;
      }
    }
  }

  emit_.Dedent();
  emit_.Line("}");
}

// Associates `name` with `node` for later Ref() lookup.
absl::Status PromelaGenerator::SetName(Node* node, std::string name) {
  names_[node] = std::move(name);
  return absl::OkStatus();
}

// Returns the Promela variable name recorded for `node`, or "" if none.
std::string_view PromelaGenerator::Ref(Node* node) const {
  auto it = names_.find(node);
  return (it != names_.end()) ? std::string_view(it->second) : "";
}

// Returns the canonical Promela variable name for `node` (v_<sanitized_name>).
std::string PromelaGenerator::Var(Node* node) const {
  return absl::StrCat("v_", SanitizeName(node->GetName()));
}

// Declares a typed Promela variable initialised to `expr`; masks narrow ints.
absl::Status PromelaGenerator::Assign(Node* node, std::string_view expr) {
  MaybeEmitLocComment(node);
  MaybeEmitIrHintComment(node);
  const std::string var = Var(node);
  if (node->GetType()->IsBits()) {
    const int64_t bit_count = node->GetType()->AsBitsOrDie()->bit_count();
    if (bit_count > 1 && bit_count < 32) {
      const uint64_t mask = (uint64_t{1} << bit_count) - 1;
      Emit("int $0 = ($1) & $2;", var, expr, mask);
      return SetName(node, var);
    }
  }
  Emit("$0 $1 = $2;", PromelaType(node->GetType()), var, expr);
  return SetName(node, var);
}

// Emits an if/fi block that sets a bit variable to 1 if `lhs cmp_op rhs`.
absl::Status PromelaGenerator::EmitCompare(CompareOp* node,
                                           std::string_view cmp_op) {
  MaybeEmitLocComment(node);
  MaybeEmitIrHintComment(node);
  const std::string var = Var(node);
  Emit("bit $0;", var);
  Emit("if");
  Emit(":: ($0 $1 $2) -> $3 = 1;", Ref(node->operand(0)), cmp_op,
       Ref(node->operand(1)), var);
  Emit(":: else -> $0 = 0;", var);
  Emit("fi");
  return SetName(node, var);
}

// Emits an n-ary bitwise expression joined by `op_sym`, optionally inverted.
absl::Status PromelaGenerator::EmitNaryBitwise(NaryOp* node,
                                               std::string_view op_sym,
                                               bool invert) {
  std::vector<std::string> parts;
  for (Node* operand : node->operands()) {
    parts.emplace_back(Ref(operand));
  }
  const std::string expr = absl::StrJoin(parts, absl::StrCat(" ", op_sym, " "));
  if (invert) {
    bool is_bit = node->GetType()->IsBits() &&
                  node->GetType()->AsBitsOrDie()->bit_count() == 1;
    return Assign(node, absl::StrCat(is_bit ? "!" : "~", "(", expr, ")"));
  }
  return Assign(node, expr);
}

// Emits an if/fi that sets a bit to 1 when `operand cmp_op sentinel`.
absl::Status PromelaGenerator::EmitBitwiseReduce(BitwiseReductionOp* node,
                                                 std::string_view cmp_op,
                                                 std::string_view sentinel) {
  MaybeEmitLocComment(node);
  MaybeEmitIrHintComment(node);
  const std::string var = Var(node);
  Emit("bit $0;", var);
  Emit("if");
  Emit(":: ($0 $1 $2) -> $3 = 1;", Ref(node->operand(0)), cmp_op, sentinel,
       var);
  Emit(":: else -> $0 = 0;", var);
  Emit("fi");
  return SetName(node, var);
}

// Emits a multiply; shared by HandleUMul and HandleSMul.
absl::Status PromelaGenerator::EmitMul(ArithOp* mul) {
  return Assign(
      mul, absl::StrCat(Ref(mul->operand(0)), " * ", Ref(mul->operand(1))));
}

// Returns "progress_<dir>_<chan>: " when emit_progress_labels is set, else "".
std::string PromelaGenerator::ProgressLabel(std::string_view dir,
                                            std::string_view chan) const {
  if (!options_.emit_progress_labels) {
    return "";
  }
  return absl::StrCat("progress_", dir, "_", chan, ": ");
}

// Emits /* file:line:col */ when emit_source_locations is set and node has loc.
void PromelaGenerator::MaybeEmitLocComment(Node* node) {
  if (!options_.emit_source_locations) {
    return;
  }
  const SourceInfo& info = node->loc();
  if (info.Empty()) {
    return;
  }
  const SourceLocation& loc = info.locations.front();
  std::optional<std::string> filename = package_->GetFilename(loc.fileno());
  const std::string file_part =
      filename.has_value() ? *filename
                           : absl::StrCat("file:", loc.fileno().value());
  Emit("/* $0:$1:$2 */", file_part, loc.lineno().value(), loc.colno().value());
}

// Emits /* ir: <node-expr> */ when emit_source_hints is set.
void PromelaGenerator::MaybeEmitIrHintComment(Node* node) {
  if (!options_.emit_source_hints) {
    return;
  }
  std::string repr = node->ToString();
  for (char& c : repr) {
    if (c == '\n' || c == '\r') {
      c = ' ';
    }
  }
  Emit("/* ir: $0 */", repr);
}

absl::Status PromelaGenerator::DefaultHandler(Node* node) {
  if (node->GetType()->IsToken() || IsUnitType(node->GetType())) {
    return SetName(node, "");
  }
  return absl::UnimplementedError(
      absl::StrCat("unsupported op: ", OpToString(node->op())));
}

absl::Status PromelaGenerator::HandleAfterAll(AfterAll* after_all) {
  return SetName(after_all, "");
}
absl::Status PromelaGenerator::HandleMinDelay(MinDelay* min_delay) {
  return SetName(min_delay, "");
}
absl::Status PromelaGenerator::HandleTrace(Trace* trace_op) {
  return SetName(trace_op, "");
}
absl::Status PromelaGenerator::HandleCover(Cover* cover) {
  return SetName(cover, "");
}
absl::Status PromelaGenerator::HandleNewChannel(NewChannel* new_channel) {
  return SetName(new_channel, "");
}
absl::Status PromelaGenerator::HandleRecvChannelEnd(RecvChannelEnd* rce) {
  return SetName(rce, "");
}
absl::Status PromelaGenerator::HandleSendChannelEnd(SendChannelEnd* sce) {
  return SetName(sce, "");
}
absl::Status PromelaGenerator::HandleParam(Param* param) {
  return SetName(param, SanitizeName(param->name()));
}

absl::Status PromelaGenerator::HandleStateRead(StateRead* state_read) {
  if (state_read->GetType()->IsToken() || IsUnitType(state_read->GetType())) {
    return SetName(state_read, "");
  }
  return SetName(
      state_read,
      absl::StrCat("s_", SanitizeName(state_read->state_element()->name())));
}

absl::Status PromelaGenerator::HandleNext(Next* next) {
  next_nodes_.push_back(next);
  return SetName(next, "");
}

absl::Status PromelaGenerator::HandleLiteral(Literal* literal) {
  if (literal->GetType()->IsToken() || IsUnitType(literal->GetType())) {
    return SetName(literal, "");
  }
  return Assign(literal, literal->value().IsBits()
                             ? BitsLiteralStr(literal->value())
                             : "0");
}

absl::Status PromelaGenerator::HandleAdd(BinOp* add) {
  return Assign(
      add, absl::StrCat(Ref(add->operand(0)), " + ", Ref(add->operand(1))));
}
absl::Status PromelaGenerator::HandleSub(BinOp* sub) {
  return Assign(
      sub, absl::StrCat(Ref(sub->operand(0)), " - ", Ref(sub->operand(1))));
}
absl::Status PromelaGenerator::HandleUMul(ArithOp* mul) { return EmitMul(mul); }
absl::Status PromelaGenerator::HandleSMul(ArithOp* mul) { return EmitMul(mul); }
absl::Status PromelaGenerator::HandleUDiv(BinOp* div) {
  return Assign(
      div, absl::StrCat(Ref(div->operand(0)), " / ", Ref(div->operand(1))));
}
absl::Status PromelaGenerator::HandleSDiv(BinOp* div) {
  return Assign(
      div, absl::StrCat(Ref(div->operand(0)), " / ", Ref(div->operand(1))));
}
absl::Status PromelaGenerator::HandleUMod(BinOp* mod) {
  return Assign(
      mod, absl::StrCat(Ref(mod->operand(0)), " % ", Ref(mod->operand(1))));
}
absl::Status PromelaGenerator::HandleSMod(BinOp* mod) {
  return Assign(
      mod, absl::StrCat(Ref(mod->operand(0)), " % ", Ref(mod->operand(1))));
}
absl::Status PromelaGenerator::HandleNeg(UnOp* neg) {
  return Assign(neg, absl::StrCat("-", Ref(neg->operand(0))));
}
absl::Status PromelaGenerator::HandleIdentity(UnOp* identity) {
  return Assign(identity, Ref(identity->operand(0)));
}

absl::Status PromelaGenerator::HandleNot(UnOp* not_op) {
  const int64_t bit_count = not_op->GetType()->AsBitsOrDie()->bit_count();
  std::string_view arg = Ref(not_op->operand(0));
  if (bit_count == 1) {
    return Assign(not_op, absl::StrCat("!(", arg, ")"));
  }
  return Assign(not_op, absl::StrCat("~(", arg, ")"));
}

absl::Status PromelaGenerator::HandleNaryAnd(NaryOp* and_op) {
  return EmitNaryBitwise(and_op, "&");
}
absl::Status PromelaGenerator::HandleNaryOr(NaryOp* or_op) {
  return EmitNaryBitwise(or_op, "|");
}
absl::Status PromelaGenerator::HandleNaryXor(NaryOp* xor_op) {
  return EmitNaryBitwise(xor_op, "^");
}
absl::Status PromelaGenerator::HandleNaryNand(NaryOp* nand_op) {
  return EmitNaryBitwise(nand_op, "&", /*invert=*/true);
}
absl::Status PromelaGenerator::HandleNaryNor(NaryOp* nor_op) {
  return EmitNaryBitwise(nor_op, "|", /*invert=*/true);
}

absl::Status PromelaGenerator::HandleAndReduce(BitwiseReductionOp* and_reduce) {
  const int64_t source_bit_count =
      and_reduce->operand(0)->GetType()->AsBitsOrDie()->bit_count();
  const std::string sentinel =
      (source_bit_count >= 32)
          ? "-1"
          : absl::StrCat((uint64_t{1} << source_bit_count) - 1);
  return EmitBitwiseReduce(and_reduce, "==", sentinel);
}
absl::Status PromelaGenerator::HandleOrReduce(BitwiseReductionOp* or_reduce) {
  return EmitBitwiseReduce(or_reduce, "!=", "0");
}
absl::Status PromelaGenerator::HandleShll(BinOp* shll) {
  return Assign(shll, absl::StrCat("(", Ref(shll->operand(0)), " << ",
                                   Ref(shll->operand(1)), ")"));
}
absl::Status PromelaGenerator::HandleShrl(BinOp* shrl) {
  return Assign(shrl, absl::StrCat("(", Ref(shrl->operand(0)), " >> ",
                                   Ref(shrl->operand(1)), ")"));
}
absl::Status PromelaGenerator::HandleShra(BinOp* shra) {
  return Assign(shra, absl::StrCat("(", Ref(shra->operand(0)), " >> ",
                                   Ref(shra->operand(1)), ")"));
}

absl::Status PromelaGenerator::HandleEq(CompareOp* eq) {
  return EmitCompare(eq, "==");
}
absl::Status PromelaGenerator::HandleNe(CompareOp* ne) {
  return EmitCompare(ne, "!=");
}
absl::Status PromelaGenerator::HandleULt(CompareOp* lt) {
  return EmitCompare(lt, "<");
}
absl::Status PromelaGenerator::HandleULe(CompareOp* le) {
  return EmitCompare(le, "<=");
}
absl::Status PromelaGenerator::HandleUGt(CompareOp* gt) {
  return EmitCompare(gt, ">");
}
absl::Status PromelaGenerator::HandleUGe(CompareOp* ge) {
  return EmitCompare(ge, ">=");
}
absl::Status PromelaGenerator::HandleSLt(CompareOp* lt) {
  return EmitCompare(lt, "<");
}
absl::Status PromelaGenerator::HandleSLe(CompareOp* le) {
  return EmitCompare(le, "<=");
}
absl::Status PromelaGenerator::HandleSGt(CompareOp* gt) {
  return EmitCompare(gt, ">");
}
absl::Status PromelaGenerator::HandleSGe(CompareOp* ge) {
  return EmitCompare(ge, ">=");
}

absl::Status PromelaGenerator::HandleBitSlice(BitSlice* bit_slice) {
  const int64_t width = bit_slice->width();
  const uint64_t mask =
      (width >= 64) ? ~uint64_t{0} : ((uint64_t{1} << width) - 1);
  return Assign(bit_slice,
                absl::StrFormat("(%s >> %d) & %u", Ref(bit_slice->operand(0)),
                                bit_slice->start(), mask));
}

absl::Status PromelaGenerator::HandleDynamicBitSlice(
    DynamicBitSlice* dynamic_bit_slice) {
  const int64_t width = dynamic_bit_slice->width();
  const uint64_t mask =
      (width >= 64) ? ~uint64_t{0} : ((uint64_t{1} << width) - 1);
  return Assign(
      dynamic_bit_slice,
      absl::StrFormat("(%s >> %s) & %u", Ref(dynamic_bit_slice->operand(0)),
                      Ref(dynamic_bit_slice->operand(1)), mask));
}

absl::Status PromelaGenerator::HandleSignExtend(ExtendOp* sign_ext) {
  MaybeEmitLocComment(sign_ext);
  MaybeEmitIrHintComment(sign_ext);
  const int64_t src_bits =
      sign_ext->operand(0)->GetType()->AsBitsOrDie()->bit_count();
  const uint64_t sign_bit = uint64_t{1} << (src_bits - 1);
  const uint64_t low_mask = sign_bit - 1;
  std::string_view arg = Ref(sign_ext->operand(0));
  const std::string var = Var(sign_ext);
  Emit("int $0;", var);
  Emit("if");
  Emit(":: (($0 & $1) != 0) -> $2 = $0 | (~$3);", arg, sign_bit, var, low_mask);
  Emit(":: else -> $0 = $1;", var, arg);
  Emit("fi");
  return SetName(sign_ext, var);
}

absl::Status PromelaGenerator::HandleZeroExtend(ExtendOp* zero_ext) {
  return Assign(zero_ext, Ref(zero_ext->operand(0)));
}

absl::Status PromelaGenerator::HandleConcat(Concat* concat) {
  const std::string var = Var(concat);
  Emit("$0 $1 = 0;", PromelaType(concat->GetType()), var);
  int64_t shift = 0;
  for (int64_t i = concat->operand_count() - 1; i >= 0; --i) {
    Node* operand = concat->operand(i);
    const int64_t width = operand->GetType()->IsBits()
                              ? operand->GetType()->AsBitsOrDie()->bit_count()
                              : 8;
    const uint64_t mask =
        (width >= 64) ? ~uint64_t{0} : ((uint64_t{1} << width) - 1);
    Emit("$0 = $0 | (($1 & $2) << $3);", var, Ref(operand), mask, shift);
    shift += width;
  }
  return SetName(concat, var);
}

absl::Status PromelaGenerator::HandleSel(Select* sel) {
  MaybeEmitLocComment(sel);
  MaybeEmitIrHintComment(sel);
  const std::string var = Var(sel);
  Emit("$0 $1;", PromelaType(sel->GetType()), var);
  Emit("if");
  for (int64_t i = 0; i < sel->cases().size(); ++i) {
    Emit(":: ($0 == $1) -> $2 = $3;", Ref(sel->selector()), i, var,
         Ref(sel->get_case(i)));
  }
  if (sel->default_value().has_value()) {
    Emit(":: else -> $0 = $1;", var, Ref(*sel->default_value()));
  } else if (!sel->cases().empty()) {
    Emit(":: else -> $0 = $1;", var,
         Ref(sel->get_case(sel->cases().size() - 1)));
  }
  Emit("fi");
  return SetName(sel, var);
}

absl::Status PromelaGenerator::HandleOneHotSel(OneHotSelect* sel) {
  const std::string var = Var(sel);
  Emit("$0 $1 = 0;", PromelaType(sel->GetType()), var);
  for (int64_t i = 0; i < sel->cases().size(); ++i) {
    Emit("if");
    Emit(":: ((($0 >> $1) & 1) == 1) -> $2 = $2 | $3;", Ref(sel->selector()), i,
         var, Ref(sel->get_case(i)));
    Emit(":: else -> skip;");
    Emit("fi");
  }
  return SetName(sel, var);
}

absl::Status PromelaGenerator::HandlePrioritySel(PrioritySelect* sel) {
  const std::string var = Var(sel);
  Emit("$0 $1 = $2;", PromelaType(sel->GetType()), var,
       Ref(sel->default_value()));
  for (int64_t i = sel->cases().size() - 1; i >= 0; --i) {
    Emit("if");
    Emit(":: ((($0 >> $1) & 1) == 1) -> $2 = $3;", Ref(sel->selector()), i, var,
         Ref(sel->get_case(i)));
    Emit(":: else -> skip;");
    Emit("fi");
  }
  return SetName(sel, var);
}

absl::Status PromelaGenerator::HandleGate(Gate* gate) {
  MaybeEmitLocComment(gate);
  MaybeEmitIrHintComment(gate);
  const std::string var = Var(gate);
  Emit("$0 $1;", PromelaType(gate->GetType()), var);
  Emit("if");
  Emit(":: ($0 != 0) -> $1 = $2;", Ref(gate->condition()), var,
       Ref(gate->data()));
  Emit(":: else -> $0 = 0;", var);
  Emit("fi");
  return SetName(gate, var);
}

absl::Status PromelaGenerator::HandleTuple(Tuple* tuple) {
  if (IsUnitType(tuple->GetType())) {
    return SetName(tuple, "");
  }
  std::vector<std::string> comps;
  comps.reserve(tuple->operand_count());
  for (Node* op : tuple->operands()) {
    comps.emplace_back(Ref(op));
  }
  tuple_components_[tuple] = std::move(comps);
  return SetName(tuple, "");
}

absl::Status PromelaGenerator::HandleTupleIndex(TupleIndex* index) {
  if (index->GetType()->IsToken() || IsUnitType(index->GetType())) {
    return SetName(index, "");
  }
  Node* src = index->operand(0);
  if (src->op() == Op::kReceive) {
    if (index->index() == 1) {
      return SetName(index, std::string(Ref(src)));
    }
    if (index->index() == 2 && !src->As<Receive>()->is_blocking()) {
      return SetName(index, absl::StrCat(Ref(src), "_valid"));
    }
    return SetName(index, "");
  }
  auto it = tuple_components_.find(src);
  if (it != tuple_components_.end()) {
    const auto& comps = it->second;
    if (index->index() < comps.size()) {
      return SetName(index, comps[index->index()]);
    }
    return SetName(index, "");
  }
  return Assign(index, absl::StrCat(Ref(src), " /* tuple_index[",
                                    index->index(), "] */"));
}

absl::Status PromelaGenerator::HandleReceive(Receive* receive) {
  MaybeEmitLocComment(receive);
  MaybeEmitIrHintComment(receive);
  Type* payload = receive->GetPayloadType();
  const std::string_view payload_type = PromelaType(payload);
  const std::string channel_name = SanitizeName(receive->channel_name());
  const std::string recv_progress_label = ProgressLabel("recv", channel_name);

  if (payload_type.empty()) {
    if (receive->predicate().has_value()) {
      Emit("if");
      Emit(":: ($0 != 0) -> $1$2 ? eval(0);", Ref(*receive->predicate()),
           recv_progress_label, channel_name);
      Emit(":: else -> skip;");
      Emit("fi");
    } else {
      Emit("$0$1 ? eval(0);", recv_progress_label, channel_name);
    }
    return SetName(receive, "");
  }

  const std::string data_var = absl::StrCat(Var(receive), "_data");
  Emit("$0 $1;", payload_type, data_var);

  if (!receive->is_blocking()) {
    const std::string valid_var = data_var + "_valid";
    Emit("bit $0;", valid_var);
    Emit("atomic {");
    Emit("if");
    if (receive->predicate().has_value()) {
      std::string_view pred = Ref(*receive->predicate());
      Emit(":: ($0 != 0) && $1?[$2] -> $3$1 ? $2; $4 = 1;", pred, channel_name,
           data_var, recv_progress_label, valid_var);
    } else {
      Emit(":: $0?[$1] -> $2$0 ? $1; $3 = 1;", channel_name, data_var,
           recv_progress_label, valid_var);
    }
    Emit(":: else -> $0 = 0;", valid_var);
    Emit("fi");
    Emit("}");
    return SetName(receive, data_var);
  }

  if (receive->predicate().has_value()) {
    Emit("if");
    Emit(":: ($0 != 0) -> $1$2 ? $3;", Ref(*receive->predicate()),
         recv_progress_label, channel_name, data_var);
    Emit(":: else -> skip;");
    Emit("fi");
  } else {
    Emit("$0$1 ? $2;", recv_progress_label, channel_name, data_var);
  }
  return SetName(receive, data_var);
}

absl::Status PromelaGenerator::HandleSend(Send* send) {
  MaybeEmitLocComment(send);
  MaybeEmitIrHintComment(send);
  const std::string channel_name = SanitizeName(send->channel_name());
  const std::string send_progress_label = ProgressLabel("send", channel_name);
  if (send->predicate().has_value()) {
    Emit("if");
    if (options_.assert_send_on_full_channel) {
      Emit(":: ($0 != 0) -> assert(len($1) < $2); $3$1 ! $4;",
           Ref(*send->predicate()), channel_name, options_.channel_depth,
           send_progress_label, Ref(send->data()));
    } else {
      Emit(":: ($0 != 0) -> $1$2 ! $3;", Ref(*send->predicate()),
           send_progress_label, channel_name, Ref(send->data()));
    }
    Emit(":: else -> skip;");
    Emit("fi");
  } else {
    if (options_.assert_send_on_full_channel) {
      Emit("assert(len($0) < $1);", channel_name, options_.channel_depth);
    }
    Emit("$0$1 ! $2;", send_progress_label, channel_name, Ref(send->data()));
  }
  return SetName(send, "");
}

absl::Status PromelaGenerator::HandleAssert(Assert* assert_op) {
  // Print the IR label only when the condition is false so the spin runner can
  // identify which assertion fired without noise on every passing iteration.
  if (assert_op->label().has_value() && !assert_op->label()->empty()) {
    std::string label = *assert_op->label();
    absl::StrReplaceAll({{"\\", "\\\\"}, {"\"", "\\\""}}, &label);
    Emit("if");
    Emit(absl::StrCat(":: (", Ref(assert_op->condition()),
                      " == 0) -> printf(\"XLS_ASSERT:", label, "\\n\");"));
    Emit(":: else -> skip;");
    Emit("fi");
  }
  Emit("assert($0 != 0);", Ref(assert_op->condition()));
  return SetName(assert_op, "");
}

absl::Status PromelaGenerator::HandleInvoke(Invoke* invoke) {
  const std::string var = Var(invoke);
  Emit("$0 $1;", PromelaType(invoke->GetType()), var);
  std::vector<std::string> arg_parts;
  for (Node* operand : invoke->operands()) {
    arg_parts.emplace_back(Ref(operand));
  }
  arg_parts.push_back(var);
  Emit("fn_$0($1);", SanitizeName(invoke->to_apply()->name()),
       absl::StrJoin(arg_parts, ", "));
  return SetName(invoke, var);
}

}  // namespace xls::spin
