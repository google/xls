// Copyright 2020 The XLS Authors
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

#include "xls/dslx/interpreter.h"

#include <memory>

#include "xls/common/status/ret_check.h"
#include "xls/dslx/ast_utils.h"
#include "xls/dslx/builtins.h"
#include "xls/dslx/evaluate.h"
#include "xls/dslx/evaluate_sym.h"
#include "xls/dslx/mangle.h"

namespace xls::dslx {

Interpreter::RunningProc::RunningProc(
    Proc* proc, const absl::flat_hash_map<std::string, InterpValue>& members,
    std::unique_ptr<InterpBindings> bindings, std::vector<RunningProc> children)
    : proc_(proc),
      members_(members),
      bindings_(std::move(bindings)),
      children_(std::move(children)) {}

std::string Interpreter::RunningProc::ToString(int indent) const {
  std::vector<std::string> output;
  std::string indent_str(indent * 2, ' ');
  output.push_back(absl::StrCat(indent_str, "Name: ", proc_->identifier()));
  output.push_back(absl::StrCat(indent_str, "Members:"));
  for (const auto& [k, v] : members_) {
    output.push_back(absl::StrCat(indent_str, " - ", k, " : ", v.ToString()));
  }
  output.push_back(absl::StrCat(indent_str, "Children:"));
  for (const auto& child : children_) {
    output.push_back(child.ToString(indent + 1));
  }
  return absl::StrJoin(output, "\n");
}

class Evaluator : public ExprVisitor {
 public:
  Evaluator(Interpreter* parent, InterpBindings* bindings,
            ConcreteType* type_context, AbstractInterpreter* interp,
            bool run_concolic)
      : parent_(parent),
        bindings_(bindings),
        type_context_(type_context),
        interp_(interp),
        run_concolic_(run_concolic) {}

#define DISPATCH(__expr_type)                                                 \
  void Handle##__expr_type(__expr_type* expr) override {                      \
    value_ = run_concolic_ ? EvaluateSym##__expr_type(expr, bindings_,        \
                                                      type_context_, interp_) \
                           : Evaluate##__expr_type(expr, bindings_,           \
                                                   type_context_, interp_);   \
  }

  DISPATCH(Array)
  DISPATCH(Attr)
  DISPATCH(Binop)
  DISPATCH(Carry)
  DISPATCH(Cast)
  DISPATCH(ColonRef)
  DISPATCH(ConstRef)
  DISPATCH(For)
  DISPATCH(Index)
  DISPATCH(Let)
  DISPATCH(Match)
  DISPATCH(NameRef)
  DISPATCH(Number)
  DISPATCH(SplatStructInstance)
  DISPATCH(String)
  DISPATCH(StructInstance)
  DISPATCH(Ternary)
  DISPATCH(Unop)
  DISPATCH(While)
  DISPATCH(XlsTuple)

#undef DISPATCH

  void HandleChannelDecl(ChannelDecl* expr) override {
    // All send/recvs check their argument channels for proper directionality,
    // so we don't need to re-do that check here (or downstream).
    auto channel = InterpValue::MakeChannel();
    value_ = InterpValue::MakeTuple({channel, channel});
  }
  void HandleFormatMacro(FormatMacro* expr) override {
    value_ = parent_->EvaluateFormatMacro(expr, bindings_, type_context_);
  }
  void HandleInvocation(Invocation* expr) override {
    value_ = parent_->EvaluateInvocation(expr, bindings_, type_context_);
  }
  void HandleRecv(Recv* expr) override {
    // TODO(rspringer): 2021-11-08: Correctly implement recv. This will require
    // returning some sort of "BLOCKED" InterpValue if the recv queue is empty.
    value_ = InterpValue::MakeTuple(
        {InterpValue::MakeToken(), InterpValue::MakeU32(0xbeef)});
  }
  void HandleRecvIf(RecvIf* expr) override {
    value_ = absl::UnimplementedError(absl::StrFormat(
        "RecvIf expression is unhandled @ %s", expr->span().ToString()));
  }

  void HandleSend(Send* expr) override { HandleSendInternal(expr); }

  void HandleSendIf(SendIf* expr) override {
    auto status_or_predicate =
        parent_->Evaluate(expr->condition(), bindings_, type_context_);
    if (!status_or_predicate.ok()) {
      value_ = status_or_predicate.status();
      return;
    }

    if (status_or_predicate.value().GetBitValueUint64().value() == 1) {
      HandleSendInternal(expr);
    } else {
      value_ = InterpValue::MakeToken();
    }
  }

  void HandleJoin(Join* expr) override {
    value_ = absl::UnimplementedError(absl::StrFormat(
        "Join expression is unhandled @ %s", expr->span().ToString()));
  }
  void HandleSpawn(Spawn* expr) override {
    absl::Status status =
        parent_->EvaluateSpawn(expr, bindings_, type_context_);
    if (!status.ok()) {
      value_ = status;
      return;
    }

    value_ = parent_->Evaluate(expr->body(), bindings_, type_context_);
  }

  absl::StatusOr<InterpValue>& value() { return value_; }

 private:
  // Common actual sending logic for send & sendif.
  template <typename SendT>
  void HandleSendInternal(SendT* send) {
    value_ = absl::UnimplementedError(absl::StrFormat(
        "Send expression is unhandled @ %s", send->span().ToString()));
    auto status_or_channel = bindings_->ResolveValue(send->channel());
    if (!status_or_channel.ok()) {
      value_ = status_or_channel.status();
      return;
    }

    if (!status_or_channel.value().IsChannel()) {
      value_ = absl::InvalidArgumentError(
          absl::StrCat("Send channel argument was not a channel: ",
                       status_or_channel.value().ToString()));
      return;
    }
    std::shared_ptr<InterpValue::Channel> channel =
        status_or_channel.value().GetChannelOrDie();

    auto status_or_payload =
        parent_->Evaluate(send->payload(), bindings_, type_context_);
    if (!status_or_payload.ok()) {
      value_ = status_or_payload.status();
      return;
    }

    channel->push_back(status_or_payload.value());
    value_ = InterpValue::MakeToken();
    XLS_VLOG(3) << "send[if]: " << send->span()
                << " : payload:" << status_or_payload.value().ToString();
  }

  Interpreter* parent_;
  InterpBindings* bindings_;
  ConcreteType* type_context_;
  AbstractInterpreter* interp_;
  absl::StatusOr<InterpValue> value_;
  bool run_concolic_;
};

// Adapts a concrete interpreter to present as an abstract one.
class AbstractInterpreterAdapter : public AbstractInterpreter {
 public:
  explicit AbstractInterpreterAdapter(Interpreter* interp) : interp_(interp) {}

  // TODO(leary): 2020-03-01 Porting artifact -- try to remove the unique_ptr
  // from the signature in AbstractInterpreter, since we just pass on the
  // ConcreteType's pointer here.
  absl::StatusOr<InterpValue> Eval(
      Expr* expr, InterpBindings* bindings,
      std::unique_ptr<ConcreteType> type_context) override {
    return interp_->Evaluate(expr, bindings, type_context.get());
  }
  absl::StatusOr<InterpValue> CallValue(
      const InterpValue& value, absl::Span<const InterpValue> args,
      const Span& invocation_span, Invocation* invocation,
      const SymbolicBindings* sym_bindings) override {
    return interp_->CallFnValue(value, args, invocation_span, invocation,
                                sym_bindings);
  }
  TypecheckFn GetTypecheckFn() override { return interp_->typecheck_; }
  bool IsWip(AstNode* node) override { return interp_->IsWip(node); }
  absl::optional<InterpValue> NoteWip(
      AstNode* node, absl::optional<InterpValue> value) override {
    return interp_->NoteWip(node, value);
  }
  TypeInfo* GetCurrentTypeInfo() override {
    return interp_->current_type_info_;
  }
  void SetCurrentTypeInfo(TypeInfo& updated) override {
    interp_->current_type_info_ = &updated;
  }
  ImportData* GetImportData() override { return interp_->import_data_; }
  FormatPreference GetTraceFormatPreference() const override {
    return interp_->trace_format_preference();
  }

 private:
  Interpreter* interp_;
};

Interpreter::Interpreter(Module* entry_module, TypecheckFn typecheck,
                         ImportData* import_data, bool trace_all,
                         bool run_concolic,
                         FormatPreference trace_format_preference,
                         PostFnEvalHook post_fn_eval_hook)
    : entry_module_(entry_module),
      current_type_info_(import_data->GetRootTypeInfo(entry_module).value()),
      post_fn_eval_hook_(std::move(post_fn_eval_hook)),
      typecheck_(std::move(typecheck)),
      import_data_(import_data),
      trace_all_(trace_all),
      run_concolic_(run_concolic),
      trace_format_preference_(trace_format_preference),
      abstract_adapter_(std::make_unique<AbstractInterpreterAdapter>(this)) {}

absl::StatusOr<InterpValue> Interpreter::RunFunction(
    absl::string_view name, absl::Span<const InterpValue> args,
    SymbolicBindings symbolic_bindings) {
  XLS_ASSIGN_OR_RETURN(Function * f, entry_module_->GetFunctionOrError(name));
  Pos fake_pos("<fake>", 0, 0);
  Span fake_span(fake_pos, fake_pos);
  XLS_ASSIGN_OR_RETURN(TypeInfo * type_info,
                       import_data_->GetRootTypeInfoForNode(f));
  TypeInfoSwap tis(this, type_info);
  return EvaluateAndCompareInternal(f, args, fake_span, /*invocation=*/nullptr,
                                    &symbolic_bindings);
}

absl::Status Interpreter::RunTest(absl::string_view name) {
  XLS_ASSIGN_OR_RETURN(TestFunction * test, entry_module_->GetTest(name));
  XLS_ASSIGN_OR_RETURN(
      const InterpBindings* top_level_bindings,
      GetOrCreateTopLevelBindings(entry_module_, abstract_adapter_.get()));
  InterpBindings bindings(/*parent=*/top_level_bindings);
  bindings.set_fn_ctx(
      FnCtx{entry_module_->name(), absl::StrFormat("%s__test", name)});
  absl::StatusOr<InterpValue> result_or =
      Evaluate(test->body(), &bindings, /*type_context=*/nullptr);
  if (!result_or.status().ok()) {
    XLS_LOG(ERROR) << result_or.status();
    return result_or.status();
  }
  if (!result_or.value().IsUnit()) {
    return absl::InternalError(absl::StrFormat(
        "EvaluateError: Want test %s to return nil tuple; got: %s",
        test->identifier(), result_or.value().ToString()));
  }
  XLS_VLOG(2) << "Ran test " << name << " successfully.";
  return absl::OkStatus();
}

absl::Status Interpreter::RunTestProc(absl::string_view name) {
  XLS_ASSIGN_OR_RETURN(TestProc * tp, entry_module_->GetTestProc(name));
  Proc* proc = tp->proc();
  XLS_ASSIGN_OR_RETURN(
      const InterpBindings* top_level_bindings,
      GetOrCreateTopLevelBindings(entry_module_, abstract_adapter_.get()));
  auto bindings =
      std::make_unique<InterpBindings>(/*parent=*/top_level_bindings);

  // Type checking guarantees the terminator channel exists.
  InterpValue terminator = InterpValue::MakeChannel();
  bindings->AddValue(tp->proc()->config()->params()[0]->identifier(),
                     terminator);

  XLS_RET_CHECK(current_type_info_ != nullptr);

  spawn_stack_.push_back(SpawnContext());
  Evaluator evaluator(this, bindings.get(), /*type_context=*/nullptr,
                      abstract_adapter_.get(), run_concolic_);
  tp->proc()->config()->body()->AcceptExpr(&evaluator);
  absl::StatusOr<InterpValue> result_or = std::move(evaluator.value());
  if (!result_or.ok()) {
    return result_or.status();
  }

  absl::flat_hash_map<std::string, InterpValue> members;
  XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* member_values,
                       result_or.value().GetValues());

  for (int i = 0; i < proc->members().size(); i++) {
    members.insert({proc->members()[i]->identifier(), member_values->at(i)});
    bindings->AddValue(proc->members()[i]->identifier(), member_values->at(i));
    XLS_VLOG(3) << "Setting member \"" << proc->members()[i]->identifier()
                << "\" to " << member_values->at(i).ToString();
  }

  std::vector<Param*> state_elems = proc->next()->params();
  bindings->AddValue(state_elems[0]->identifier(), InterpValue::MakeToken());
  for (int i = 0; i < tp->next_args().size(); i++) {
    tp->next_args()[i]->AcceptExpr(&evaluator);
    XLS_ASSIGN_OR_RETURN(InterpValue next_arg, evaluator.value());
    XLS_VLOG(3) << "Updating \"" << state_elems[i + 1]->identifier() << "\" to "
                << next_arg.ToString();
    bindings->AddValue(state_elems[i + 1]->identifier(), next_arg);
  }

  RunningProc root = RunningProc(tp->proc(), members, std::move(bindings),
                                 std::move(spawn_stack_.back().child_procs));
  XLS_RETURN_IF_ERROR(RunTestProc(&root, terminator.GetChannelOrDie()));

  XLS_VLOG(2) << "Ran test " << name << " successfully.";
  return absl::OkStatus();
}

absl::Status Interpreter::TickProc(RunningProc* rp) {
  // Tick us first.
  Evaluator evaluator(this, rp->bindings(), /*type_context=*/nullptr,
                      abstract_adapter_.get(), run_concolic_);
  rp->proc()->next()->body()->AcceptExpr(&evaluator);
  XLS_ASSIGN_OR_RETURN(InterpValue result, evaluator.value());

  std::vector<Param*> state_elems = rp->proc()->next()->params();
  XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* result_elements,
                       result.GetValues());
  // Update bindings with the new state values.
  for (int i = 0; i < result_elements->size(); i++) {
    XLS_VLOG(3) << "Updating \"" << state_elems[i + 1]->identifier() << "\" to "
                << result_elements->at(i).ToString();
    rp->bindings()->AddValue(state_elems[i + 1]->identifier(),
                             result_elements->at(i));
  }

  // ...then recursively tick our kids.
  for (auto& child : rp->children()) {
    XLS_RETURN_IF_ERROR(TickProc(&child));
  }
  return absl::OkStatus();
}

absl::Status Interpreter::RunTestProc(
    RunningProc* rp, std::shared_ptr<InterpValue::Channel> terminator) {
  // Keep ticking the network until there's data in the terminator channel.
  // The value placed therein is the result of the test.
  while (terminator->empty()) {
    XLS_RETURN_IF_ERROR(TickProc(rp));
  }

  // TODO(rspringer): 2021-11-09: Need to support actual failure
  // messages/errors. How to capture AssertEq failures from inside the network?
  if (terminator->back().GetBitValueUint64().value() == false) {
    return FailureErrorStatus(rp->proc()->next()->span(),
                              "Terminator returned false/failure.");
  }

  return absl::OkStatus();
}

absl::StatusOr<InterpValue> Interpreter::EvaluateLiteral(Expr* expr) {
  InterpBindings bindings(/*parent=*/nullptr);
  return Evaluate(expr, &bindings, /*type_context=*/nullptr);
}

absl::StatusOr<InterpValue> Interpreter::Evaluate(Expr* expr,
                                                  InterpBindings* bindings,
                                                  ConcreteType* type_context) {
  XLS_RET_CHECK(current_type_info_ != nullptr);
  XLS_RET_CHECK_EQ(expr->owner(), current_type_info_->module())
      << expr->span() << " vs " << current_type_info_->module()->name();
  Evaluator evaluator(this, bindings, type_context, abstract_adapter_.get(),
                      run_concolic_);
  expr->AcceptExpr(&evaluator);
  absl::StatusOr<InterpValue> result_or = std::move(evaluator.value());
  if (!result_or.ok()) {
    return result_or;
  }
  InterpValue result = std::move(result_or).value();
  if (trace_all_) {
    OptionalTrace(expr, result, abstract_adapter_.get());
  }
  return result;
}

/* static */ absl::StatusOr<InterpValue> Interpreter::InterpretExpr(
    Module* entry_module, TypeInfo* type_info, TypecheckFn typecheck,
    ImportData* import_data,
    const absl::flat_hash_map<std::string, InterpValue>& env, Expr* expr,
    const FnCtx* fn_ctx, ConcreteType* type_context) {
  XLS_RET_CHECK_EQ(entry_module, type_info->module());
  XLS_RET_CHECK_EQ(expr->owner(), entry_module);

  auto env_formatter = [](std::string* out,
                          const std::pair<std::string, InterpValue>& p) {
    out->append(absl::StrCat(p.first, ":", p.second.ToString()));
  };
  XLS_VLOG(3) << "InterpretExpr: " << expr->ToString() << " env: {"
              << absl::StrJoin(env, ", ", env_formatter) << "}";

  Interpreter interp(entry_module, typecheck, import_data);
  XLS_ASSIGN_OR_RETURN(const InterpBindings* top_level_bindings,
                       GetOrCreateTopLevelBindings(
                           entry_module, interp.abstract_adapter_.get()));
  InterpBindings bindings(/*parent=*/top_level_bindings);
  if (fn_ctx != nullptr) {
    bindings.set_fn_ctx(*fn_ctx);
  }
  for (const auto& [identifier, value] : env) {
    XLS_VLOG(3) << "Adding to bindings; identifier: " << identifier
                << " value: " << value.ToString();
    bindings.AddValue(identifier, value);
  }

  XLS_ASSIGN_OR_RETURN(TypeInfo * expr_root_type_info,
                       import_data->GetRootTypeInfoForNode(expr));
  TypeInfoSwap tis(&interp, expr_root_type_info);
  return interp.Evaluate(expr, &bindings, /*type_context=*/type_context);
}

/* static */ absl::StatusOr<Bits> Interpreter::InterpretExprToBits(
    Module* entry_module, TypeInfo* type_info, TypecheckFn typecheck,
    ImportData* import_data,
    const absl::flat_hash_map<std::string, InterpValue>& env, Expr* expr,
    const FnCtx* fn_ctx, ConcreteType* type_context) {
  XLS_ASSIGN_OR_RETURN(
      InterpValue value,
      InterpretExpr(entry_module, type_info, typecheck, import_data, env, expr,
                    fn_ctx, type_context));
  switch (value.tag()) {
    case InterpValueTag::kUBits:
    case InterpValueTag::kSBits:
      return value.GetBits();
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Expression %s @ %s did not evaluate to an integral value",
          expr->ToString(), expr->span().ToString()));
  }
}

absl::StatusOr<InterpValue> Interpreter::RunBuiltin(
    Builtin builtin, absl::Span<InterpValue const> args, const Span& span,
    Invocation* invocation, const SymbolicBindings* symbolic_bindings) {
  switch (builtin) {
#define CASE(__name)       \
  case Builtin::k##__name: \
    return Builtin##__name(args, span, invocation, symbolic_bindings)
    CASE(AddWithCarry);
    CASE(AssertEq);
    CASE(AssertLt);
    CASE(BitSlice);
    CASE(BitSliceUpdate);
    CASE(Clz);
    CASE(Cover);
    CASE(Ctz);
    CASE(Enumerate);
    CASE(Fail);
    CASE(Gate);
    CASE(OneHot);
    CASE(OneHotSel);
    CASE(Range);
    CASE(Rev);
    CASE(Signex);
    CASE(Slice);
    CASE(Update);
    // Reductions.
    CASE(AndReduce);
    CASE(OrReduce);
    CASE(XorReduce);
#undef CASE
    case Builtin::kMap:  // Needs callbacks.
      return BuiltinMap(args, span, invocation, symbolic_bindings,
                        abstract_adapter_.get());
    case Builtin::kTrace:  // Needs callbacks.
      return BuiltinTrace(args, span, invocation, symbolic_bindings,
                          abstract_adapter_.get());
    default:
      return absl::UnimplementedError("Unhandled builtin: " +
                                      BuiltinToString(builtin));
  }
}

absl::StatusOr<InterpValue> Interpreter::CallFnValue(
    const InterpValue& fv, absl::Span<InterpValue const> args, const Span& span,
    Invocation* invocation, const SymbolicBindings* symbolic_bindings) {
  XLS_RET_CHECK(current_type_info_ != nullptr);
  if (fv.IsBuiltinFunction()) {
    auto builtin = absl::get<Builtin>(fv.GetFunctionOrDie());
    return RunBuiltin(builtin, args, span, invocation, symbolic_bindings);
  }
  const auto& fn_data =
      absl::get<InterpValue::UserFnData>(fv.GetFunctionOrDie());
  XLS_RET_CHECK_EQ(fn_data.function->owner(), current_type_info_->module())
      << fn_data.function->owner()->name() << " vs "
      << current_type_info_->module()->name();
  return EvaluateAndCompareInternal(fn_data.function, args, span, invocation,
                                    symbolic_bindings);
}

absl::StatusOr<InterpValue> Interpreter::EvaluateAndCompareInternal(
    Function* f, absl::Span<const InterpValue> args, const Span& span,
    Invocation* invocation, const SymbolicBindings* symbolic_bindings) {
  XLS_ASSIGN_OR_RETURN(
      InterpValue interpreter_value,
      run_concolic_
          ? EvaluateSymFunction(f, args, span,
                                symbolic_bindings == nullptr
                                    ? SymbolicBindings()
                                    : *symbolic_bindings,
                                abstract_adapter_.get())
          : EvaluateFunction(f, args, span,
                             symbolic_bindings == nullptr ? SymbolicBindings()
                                                          : *symbolic_bindings,
                             abstract_adapter_.get()));

  if (post_fn_eval_hook_ != nullptr) {
    XLS_RETURN_IF_ERROR(
        post_fn_eval_hook_(f, args, symbolic_bindings, interpreter_value));
  }

  return interpreter_value;
}

absl::StatusOr<InterpValue> Interpreter::EvaluateFormatMacro(
    FormatMacro* expr, InterpBindings* bindings, ConcreteType* type_context) {
  XLS_VLOG(3) << absl::StreamFormat("EvaluateFormatMacro: `%s` @ %s",
                                    expr->ToString(), expr->span().ToString());

  // Evaluate all the arguments we want to pass to the macro.
  std::vector<InterpValue> arg_values;
  for (Expr* arg : expr->args()) {
    XLS_ASSIGN_OR_RETURN(InterpValue arg_value,
                         Evaluate(arg, bindings, /*type_context=*/nullptr));
    arg_values.push_back(std::move(arg_value));
  }

  if (expr->macro() == "trace_fmt!") {
    return BuiltinTraceFmt(arg_values, expr);
  } else {
    return absl::UnimplementedError(
        absl::StrFormat("Unimplemented format macro %s at %s", expr->macro(),
                        expr->span().ToString()));
  }
}

absl::StatusOr<InterpValue> Interpreter::EvaluateInvocation(
    Invocation* expr, InterpBindings* bindings, ConcreteType* type_context) {
  XLS_VLOG(3) << absl::StreamFormat("EvaluateInvocation: `%s` @ %s",
                                    expr->ToString(), expr->span().ToString());

  // Evaluate all the argument values we want to pass to the function.
  std::vector<InterpValue> arg_values;
  for (Expr* arg : expr->args()) {
    XLS_ASSIGN_OR_RETURN(InterpValue arg_value,
                         Evaluate(arg, bindings, /*type_context=*/nullptr));
    arg_values.push_back(std::move(arg_value));
  }

  // Evaluate the callee value.
  XLS_ASSIGN_OR_RETURN(
      InterpValue callee_value,
      Evaluate(expr->callee(), bindings, /*type_context=*/nullptr));
  if (!callee_value.IsFunction()) {
    return absl::InternalError(absl::StrFormat(
        "EvaluateError: %s Callee value is not a function; should have been "
        "determined during type inference; got %s",
        expr->callee()->span().ToString(), callee_value.ToString()));
  }

  if (trace_all_ && callee_value.IsTraceBuiltin()) {
    // TODO(leary): 2020-11-19 This was the previous behavior, but I'm pretty
    // sure it's not right to skip traces, because they're supposed to result in
    // their (traced) argument.
    return InterpValue::MakeUnit();
  }

  absl::optional<SymbolicBindings> owned_symbolic_bindings;
  const SymbolicBindings* fn_symbolic_bindings = nullptr;
  if (bindings->fn_ctx().has_value()) {
    // The symbolic bindings of this invocation were already computed during
    // typechecking.
    absl::optional<const SymbolicBindings*> callee_bindings =
        current_type_info_->GetInstantiationCalleeBindings(
            expr, bindings->fn_ctx()->sym_bindings);
    if (callee_bindings.has_value()) {
      fn_symbolic_bindings = callee_bindings.value();
      XLS_VLOG(5) << "Found callee symbolic bindings: " << *fn_symbolic_bindings
                  << " @ " << expr->span();
    } else {
      owned_symbolic_bindings.emplace();
      fn_symbolic_bindings = &*owned_symbolic_bindings;
    }
  } else {
    // Note, when there's no function context we may be in a ConstantDef doing
    // e.g. a parametric invocation.
    XLS_VLOG(5) << "EvaluateInvocation; getting callee bindings without "
                   "function context"
                << "; type_info: " << current_type_info_ << "; node: " << expr
                << "; expr: `" << expr->ToString() << "`";
    absl::optional<const SymbolicBindings*> callee_bindings =
        current_type_info_->GetInstantiationCalleeBindings(expr,
                                                           SymbolicBindings());
    if (callee_bindings.has_value()) {
      XLS_RET_CHECK(callee_bindings.value() != nullptr);
      fn_symbolic_bindings = callee_bindings.value();
    } else {
      owned_symbolic_bindings.emplace();
      fn_symbolic_bindings = &*owned_symbolic_bindings;
    }
  }

  XLS_RET_CHECK(fn_symbolic_bindings != nullptr);
  TypeInfo* invocation_type_info;
  if (current_type_info_->HasInstantiation(expr, *fn_symbolic_bindings)) {
    invocation_type_info =
        current_type_info_
            ->GetInstantiationTypeInfo(expr, *fn_symbolic_bindings)
            .value();
    XLS_VLOG(5) << "Instantiation exists for " << expr->ToString()
                << " sym_bindings: " << *fn_symbolic_bindings
                << " invocation_type_info: " << invocation_type_info;
  } else {
    absl::optional<Module*> callee_module = GetFunctionValueOwner(callee_value);
    if (callee_module) {
      XLS_ASSIGN_OR_RETURN(invocation_type_info,
                           import_data_->GetRootTypeInfo(*callee_module));
    } else {
      invocation_type_info = current_type_info_;
    }
  }
  absl::optional<TypeInfoSwap> tis;
  if (invocation_type_info != nullptr) {  // Builtins have no type info.
    tis.emplace(this, invocation_type_info);
  }
  absl::StatusOr<InterpValue> result = CallFnValue(
      callee_value, arg_values, expr->span(), expr, fn_symbolic_bindings);
  if (!result.ok()) {
    Invocation* invocation = dynamic_cast<Invocation*>(expr);

    std::string invoking_fn_name;
    if (bindings->fn_ctx().has_value()) {
      invoking_fn_name = absl::StrCat("::", bindings->fn_ctx()->fn_name);
    } else {
      invoking_fn_name = " @ <top>";
    }
    std::string function_name =
        absl::StrCat(invocation->owner()->name(), invoking_fn_name);
    return absl::Status(
        result.status().code(),
        absl::StrCat(result.status().message(), "\n  via ", function_name,
                     " @ ", expr->span().ToString(), " : ", expr->ToString()));
  }
  return result;
}

absl::Status Interpreter::EvaluateSpawn(Spawn* expr, InterpBindings* bindings,
                                        ConcreteType* type_context) {
  XLS_VLOG(3) << absl::StreamFormat("EvaluateSpawn: `%s` @ %s",
                                    expr->ToString(), expr->span().ToString());

  // Evaluate all the argument values we want to pass to the function.
  std::vector<Expr*> config_args;
  for (Expr* arg : expr->config()->args()) {
    config_args.push_back(arg);
  }

  std::vector<InterpValue> next_arg_values;
  for (Expr* arg : expr->next()->args()) {
    XLS_ASSIGN_OR_RETURN(InterpValue arg_value,
                         Evaluate(arg, bindings, /*type_context=*/nullptr));
    next_arg_values.push_back(std::move(arg_value));
  }

  // Add a new element to the stack so we can collect this Spawn's children.
  spawn_stack_.push_back(SpawnContext());
  XLS_ASSIGN_OR_RETURN(
      InterpValue member_tuple,
      EvaluateInvocation(expr->config(), bindings, type_context));
  SpawnContext ctx = std::move(spawn_stack_.back());
  spawn_stack_.pop_back();

  absl::flat_hash_map<std::string, InterpValue> members;
  XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* member_values,
                       member_tuple.GetValues());

  XLS_ASSIGN_OR_RETURN(
      const InterpBindings* top_level_bindings,
      GetOrCreateTopLevelBindings(entry_module_, abstract_adapter_.get()));
  auto child_bindings =
      std::make_unique<InterpBindings>(/*parent=*/top_level_bindings);

  XLS_ASSIGN_OR_RETURN(Proc * proc,
                       ResolveProc(expr->callee(), current_type_info_));
  for (int64_t i = 0; i < proc->members().size(); i++) {
    members.insert({proc->members()[i]->identifier(), member_values->at(i)});
    child_bindings->AddValue(proc->members()[i]->identifier(),
                             member_values->at(i));
  }

  spawn_stack_.back().child_procs.push_back(
      RunningProc(proc, std::move(members), std::move(child_bindings),
                  std::move(ctx.child_procs)));
  return absl::OkStatus();
}

bool Interpreter::IsWip(AstNode* node) const {
  auto it = wip_.find(node);
  // If it's in the "work in progress" mapping but doesn't have a completed
  // value associated with it, it's work in progress.
  bool marked_wip = it != wip_.end() && !it->second.has_value();
  return marked_wip ||
         import_data_->GetTypecheckWorkInProgress(node->owner()) == node;
}

absl::optional<InterpValue> Interpreter::NoteWip(
    AstNode* node, absl::optional<InterpValue> value) {
  if (!value.has_value()) {
    // Implicitly value-initializes wip_ entry with absl::nullopt
    // marking as WIP if not already present. Otherwise returns the
    // cached value.
    return wip_[node];
  }

  wip_[node] = value;
  return value;
}

absl::StatusOr<InterpValue> SignConvertValue(const ConcreteType& concrete_type,
                                             const InterpValue& value) {
  if (auto* tuple_type = dynamic_cast<const TupleType*>(&concrete_type)) {
    XLS_RET_CHECK(value.IsTuple()) << value.ToString();
    const int64_t tuple_size = value.GetValuesOrDie().size();
    std::vector<InterpValue> results;
    for (int64_t i = 0; i < tuple_size; ++i) {
      const InterpValue& e = value.GetValuesOrDie()[i];
      const ConcreteType& t = tuple_type->GetMemberType(i);
      XLS_ASSIGN_OR_RETURN(InterpValue converted, SignConvertValue(t, e));
      results.push_back(converted);
    }
    return InterpValue::MakeTuple(std::move(results));
  }
  if (auto* array_type = dynamic_cast<const ArrayType*>(&concrete_type)) {
    XLS_RET_CHECK(value.IsArray()) << value.ToString();
    const ConcreteType& t = array_type->element_type();
    int64_t array_size = value.GetValuesOrDie().size();
    std::vector<InterpValue> results;
    for (int64_t i = 0; i < array_size; ++i) {
      const InterpValue& e = value.GetValuesOrDie()[i];
      XLS_ASSIGN_OR_RETURN(InterpValue converted, SignConvertValue(t, e));
      results.push_back(converted);
    }
    return InterpValue::MakeArray(std::move(results));
  }
  if (auto* bits_type = dynamic_cast<const BitsType*>(&concrete_type)) {
    XLS_RET_CHECK(value.IsBits()) << value.ToString();
    if (bits_type->is_signed()) {
      return InterpValue::MakeBits(InterpValueTag::kSBits,
                                   value.GetBitsOrDie());
    }
    return value;
  }
  if (auto* enum_type = dynamic_cast<const EnumType*>(&concrete_type)) {
    XLS_RET_CHECK(value.IsBits()) << value.ToString();
    XLS_RET_CHECK(enum_type->signedness().has_value());
    if (*enum_type->signedness()) {
      return InterpValue::MakeBits(InterpValueTag::kSBits,
                                   value.GetBitsOrDie());
    }
    return value;
  }
  return absl::UnimplementedError("Cannot sign convert type: " +
                                  concrete_type.ToString());
}

absl::StatusOr<std::vector<InterpValue>> SignConvertArgs(
    const FunctionType& fn_type, absl::Span<const InterpValue> args) {
  absl::Span<const std::unique_ptr<ConcreteType>> params = fn_type.params();
  XLS_RET_CHECK_EQ(params.size(), args.size());
  std::vector<InterpValue> converted;
  converted.reserve(args.size());
  for (int64_t i = 0; i < args.size(); ++i) {
    XLS_ASSIGN_OR_RETURN(InterpValue value,
                         SignConvertValue(*params[i], args[i]));
    converted.push_back(value);
  }
  return converted;
}

}  // namespace xls::dslx
