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

#include "xls/common/status/ret_check.h"
#include "xls/dslx/builtins.h"
#include "xls/dslx/evaluate.h"
#include "xls/dslx/mangle.h"
#include "xls/jit/ir_jit.h"

namespace xls::dslx {

class Evaluator : public ExprVisitor {
 public:
  Evaluator(Interpreter* parent, InterpBindings* bindings,
            ConcreteType* type_context, AbstractInterpreter* interp)
      : parent_(parent),
        bindings_(bindings),
        type_context_(type_context),
        interp_(interp) {}

#define DISPATCH(__expr_type)                                                \
  void Handle##__expr_type(__expr_type* expr) override {                     \
    value_ = Evaluate##__expr_type(expr, bindings_, type_context_, interp_); \
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
  DISPATCH(StructInstance)
  DISPATCH(Ternary)
  DISPATCH(Unop)
  DISPATCH(While)
  DISPATCH(XlsTuple)

#undef DISPATCH

  void HandleInvocation(Invocation* expr) override {
    value_ = parent_->EvaluateInvocation(expr, bindings_, type_context_);
  }
  void HandleNext(Next* expr) override {
    value_ = absl::UnimplementedError(absl::StrFormat(
        "Next expression is unhandled @ %s", expr->span().ToString()));
  }

  absl::StatusOr<InterpValue>& value() { return value_; }

 private:
  Interpreter* parent_;
  InterpBindings* bindings_;
  ConcreteType* type_context_;
  AbstractInterpreter* interp_;
  absl::StatusOr<InterpValue> value_;
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
  void SetCurrentTypeInfo(TypeInfo* updated) override {
    interp_->current_type_info_ = updated;
  }
  ImportData* GetImportData() override { return interp_->import_data_; }
  absl::Span<std::string const> GetAdditionalSearchPaths() override {
    return interp_->additional_search_paths_;
  }

 private:
  Interpreter* interp_;
};

Interpreter::Interpreter(Module* entry_module, TypecheckFn typecheck,
                         absl::Span<std::string const> additional_search_paths,
                         ImportData* import_data, bool trace_all,
                         Package* ir_package)
    : entry_module_(entry_module),
      current_type_info_(import_data->GetRootTypeInfo(entry_module).value()),
      typecheck_(std::move(typecheck)),
      additional_search_paths_(additional_search_paths.begin(),
                               additional_search_paths.end()),
      import_data_(import_data),
      trace_all_(trace_all),
      ir_package_(ir_package),
      abstract_adapter_(absl::make_unique<AbstractInterpreterAdapter>(this)) {}

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
  XLS_ASSIGN_OR_RETURN(InterpValue result, Evaluate(test->body(), &bindings,
                                                    /*type_context=*/nullptr));
  if (!result.IsNilTuple()) {
    return absl::InternalError(absl::StrFormat(
        "EvaluateError: Want test %s to return nil tuple; got: %s",
        test->identifier(), result.ToString()));
  }
  XLS_VLOG(2) << "Ran test " << name << " successfully.";
  return absl::OkStatus();
}

absl::StatusOr<InterpValue> Interpreter::EvaluateLiteral(Expr* expr) {
  InterpBindings bindings(/*parent=*/nullptr);
  return Evaluate(expr, &bindings, /*type_context=*/nullptr);
}

absl::StatusOr<InterpValue> Interpreter::Evaluate(Expr* expr,
                                                  InterpBindings* bindings,
                                                  ConcreteType* type_context) {
  XLS_RET_CHECK_EQ(expr->owner(), current_type_info_->module())
      << expr->span() << " vs " << current_type_info_->module()->name();
  Evaluator evaluator(this, bindings, type_context, abstract_adapter_.get());
  expr->AcceptExpr(&evaluator);
  absl::StatusOr<InterpValue> result_or = std::move(evaluator.value());
  if (!result_or.ok()) {
    if (result_or.status().code() != absl::StatusCode::kNotFound) {
      XLS_LOG(ERROR) << "error @ " << expr->span() << ": "
                     << result_or.status();
    }
    return result_or;
  }
  InterpValue result = std::move(result_or).value();
  if (trace_all_) {
    OptionalTrace(expr, result);
  }
  return result;
}

/* static */ absl::StatusOr<int64_t> Interpreter::InterpretExprToInt(
    Module* entry_module, TypeInfo* type_info, TypecheckFn typecheck,
    absl::Span<std::string const> additional_search_paths,
    ImportData* import_data,
    const absl::flat_hash_map<std::string, int64_t>& env,
    const absl::flat_hash_map<std::string, int64_t>& bit_widths, Expr* expr,
    const FnCtx* fn_ctx, ConcreteType* type_context) {
  XLS_VLOG(3) << "InterpretExpr: " << expr->ToString() << " env: {"
              << absl::StrJoin(env, ", ", absl::PairFormatter(":")) << "}";

  Interpreter interp(entry_module, typecheck, additional_search_paths,
                     import_data);
  XLS_ASSIGN_OR_RETURN(const InterpBindings* top_level_bindings,
                       GetOrCreateTopLevelBindings(
                           entry_module, interp.abstract_adapter_.get()));
  InterpBindings bindings(/*parent=*/top_level_bindings);
  if (fn_ctx != nullptr) {
    bindings.set_fn_ctx(*fn_ctx);
  }
  for (const auto& [identifier, value] : env) {
    XLS_VLOG(3) << "Adding to bindings; identifier: " << identifier
                << " value: " << value;
    auto it = bit_widths.find(identifier);
    if (it == bit_widths.end()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Could not find bitwidth for identifier %s; env: {%s}; bit_widths: "
          "{%s}",
          identifier, absl::StrJoin(env, ", ", absl::PairFormatter(":")),
          absl::StrJoin(bit_widths, ", ", absl::PairFormatter(":"))));
    }
    bindings.AddValue(
        identifier,
        InterpValue::MakeUBits(/*bit_count=*/it->second, /*value=*/value));
  }

  XLS_ASSIGN_OR_RETURN(TypeInfo * expr_root_type_info,
                       import_data->GetRootTypeInfoForNode(expr));
  TypeInfoSwap tis(&interp, expr_root_type_info);
  XLS_ASSIGN_OR_RETURN(
      InterpValue result,
      interp.Evaluate(expr, &bindings, /*type_context=*/type_context));
  switch (result.tag()) {
    case InterpValueTag::kUBits: {
      XLS_ASSIGN_OR_RETURN(uint64_t result, result.GetBitValueUint64());
      return result;
    }
    case InterpValueTag::kSBits:
      return result.GetBitValueInt64();
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
    CASE(Ctz);
    CASE(Enumerate);
    CASE(Fail);
    CASE(OneHot);
    CASE(OneHotSel);
    CASE(Range);
    CASE(Rev);
    CASE(Signex);
    CASE(Slice);
    CASE(Trace);
    CASE(Update);
    // Reductions.
    CASE(AndReduce);
    CASE(OrReduce);
    CASE(XorReduce);
#undef CASE
    case Builtin::kSLt:
      return BuiltinScmp(SignedCmp::kLt, args, span, invocation,
                         symbolic_bindings);
    case Builtin::kSLe:
      return BuiltinScmp(SignedCmp::kLe, args, span, invocation,
                         symbolic_bindings);
    case Builtin::kSGt:
      return BuiltinScmp(SignedCmp::kGt, args, span, invocation,
                         symbolic_bindings);
    case Builtin::kSGe:
      return BuiltinScmp(SignedCmp::kGe, args, span, invocation,
                         symbolic_bindings);
    case Builtin::kMap:  // Needs callbacks.
      return BuiltinMap(args, span, invocation, symbolic_bindings,
                        abstract_adapter_.get());
    default:
      return absl::UnimplementedError("Unhandled builtin: " +
                                      BuiltinToString(builtin));
  }
}

// Retrieves the module associated with the function_value if it is user
// defined.
//
// Check-fails if function_value is not a function-typed value.
static absl::optional<Module*> GetFunctionValueOwner(
    const InterpValue& function_value) {
  if (function_value.IsBuiltinFunction()) {
    return absl::nullopt;
  }
  const auto& fn_data =
      absl::get<InterpValue::UserFnData>(function_value.GetFunctionOrDie());
  return fn_data.function->owner();
}

absl::StatusOr<InterpValue> Interpreter::CallFnValue(
    const InterpValue& fv, absl::Span<InterpValue const> args, const Span& span,
    Invocation* invocation, const SymbolicBindings* symbolic_bindings) {
  if (fv.IsBuiltinFunction()) {
    auto builtin = absl::get<Builtin>(fv.GetFunctionOrDie());
    return RunBuiltin(builtin, args, span, invocation, symbolic_bindings);
  }
  const auto& fn_data =
      absl::get<InterpValue::UserFnData>(fv.GetFunctionOrDie());
  XLS_RET_CHECK_EQ(fn_data.function->owner(), current_type_info_->module());
  return EvaluateAndCompareInternal(fn_data.function, args, span, invocation,
                                    symbolic_bindings);
}

absl::Status Interpreter::RunJitComparison(
    Function* f, absl::Span<InterpValue const> args,
    const SymbolicBindings* symbolic_bindings,
    const InterpValue& expected_value) {
  if (ir_package_ != nullptr) {
    XLS_ASSIGN_OR_RETURN(
        std::string ir_name,
        MangleDslxName(f->identifier(), f->GetFreeParametricKeySet(),
                       f->owner(), symbolic_bindings));

    auto get_result = ir_package_->GetFunction(ir_name);

    // ir_package_ does not include specializations of parametric functions
    // that are only called from test code, so not finding the function
    // may be benign.
    // TODO(amfv): 2021-03-18 Extend IR conversion to include those functions.
    if (!get_result.ok()) {
      XLS_LOG(WARNING) << "Could not find " << ir_name
                       << " function for JIT comparison";
      return absl::OkStatus();
    }

    xls::Function* ir_function = get_result.value();

    XLS_ASSIGN_OR_RETURN(std::vector<Value> ir_args,
                         InterpValue::ConvertValuesToIr(args));

    // TODO(leary): 2020-11-19 Cache JIT function so we don't have to create it
    // every time.
    XLS_ASSIGN_OR_RETURN(Value jit_value, CreateAndRun(ir_function, ir_args));

    XLS_ASSIGN_OR_RETURN(Value expected_ir, expected_value.ConvertToIr());
    XLS_RET_CHECK_EQ(expected_ir, jit_value) << absl::StreamFormat(
        "\n%s\n vs \n%s", f->ToString(), ir_function->DumpIr());
  }
  return absl::OkStatus();
}

absl::StatusOr<InterpValue> Interpreter::EvaluateAndCompareInternal(
    Function* f, absl::Span<const InterpValue> args, const Span& span,
    Invocation* invocation, const SymbolicBindings* symbolic_bindings) {
  XLS_ASSIGN_OR_RETURN(
      InterpValue interpreter_value,
      EvaluateFunction(f, args, span,
                       symbolic_bindings == nullptr ? SymbolicBindings()
                                                    : *symbolic_bindings,
                       abstract_adapter_.get()));

  XLS_RETURN_IF_ERROR(
      RunJitComparison(f, args, symbolic_bindings, interpreter_value));

  return interpreter_value;
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
    return InterpValue::MakeNil();
  }

  const SymbolicBindings* fn_symbolic_bindings = nullptr;
  if (bindings->fn_ctx().has_value()) {
    // The symbolic bindings of this invocation were already computed during
    // typechecking.
    absl::optional<const SymbolicBindings*> callee_bindings =
        current_type_info_->GetInvocationSymbolicBindings(
            expr, bindings->fn_ctx()->sym_bindings);
    if (!callee_bindings.has_value()) {
      return absl::NotFoundError(
          absl::StrFormat("Could not find callee bindings in type info for "
                          "FnCtx: %s expr: %s @ %s",
                          bindings->fn_ctx()->ToString(), expr->ToString(),
                          expr->span().ToString()));
    }
    XLS_RET_CHECK(callee_bindings.value() != nullptr);
    fn_symbolic_bindings = callee_bindings.value();
    XLS_VLOG(5) << "Found callee symbolic bindings: " << *fn_symbolic_bindings
                << " @ " << expr->span();
  } else {
    // Note, when there's no function context we may be in a ConstantDef doing
    // e.g. a parametric invocation.
    XLS_VLOG(5) << "EvaluateInvocation; getting callee bindings without "
                   "function context"
                << "; type_info: " << current_type_info_ << "; node: " << expr
                << "; expr: `" << expr->ToString() << "`";
    absl::optional<const SymbolicBindings*> callee_bindings =
        current_type_info_->GetInvocationSymbolicBindings(expr,
                                                          SymbolicBindings());
    XLS_RET_CHECK(callee_bindings.has_value()) << absl::StreamFormat(
        "current_type_info: %p invocation: %p `%s` @ %s", current_type_info_,
        expr, expr->ToString(), expr->span().ToString());
    XLS_RET_CHECK(callee_bindings.value() != nullptr);
    fn_symbolic_bindings = callee_bindings.value();
  }

  XLS_RET_CHECK(fn_symbolic_bindings != nullptr);
  TypeInfo* invocation_type_info;
  if (current_type_info_->HasInstantiation(expr, *fn_symbolic_bindings)) {
    invocation_type_info =
        current_type_info_->GetInstantiation(expr, *fn_symbolic_bindings)
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
  TypeInfoSwap tis(this, invocation_type_info);
  return CallFnValue(callee_value, arg_values, expr->span(), expr,
                     fn_symbolic_bindings);
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
    // Starting evaluation, attempting to mark as WIP.
    auto it = wip_.find(node);
    if (it != wip_.end() && it->second.has_value()) {
      return it->second;  // Already computed.
    }
    wip_[node] = absl::nullopt;  // Mark as WIP.
    return absl::nullopt;
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
