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
  bool IsWip(ConstantDef* constant_def) override {
    return interp_->IsWip(constant_def);
  }
  absl::optional<InterpValue> NoteWip(
      ConstantDef* constant_def, absl::optional<InterpValue> value) override {
    return interp_->NoteWip(constant_def, value);
  }
  TypeInfo* GetCurrentTypeInfo() override { return interp_->type_info_; }
  ImportCache* GetImportCache() override { return interp_->import_cache_; }
  absl::Span<std::string const> GetAdditionalSearchPaths() override {
    return interp_->additional_search_paths_;
  }

 private:
  Interpreter* interp_;
};

Interpreter::Interpreter(Module* module, TypeInfo* type_info,
                         TypecheckFn typecheck,
                         absl::Span<std::string const> additional_search_paths,
                         ImportCache* import_cache, bool trace_all,
                         Package* ir_package)
    : module_(module),
      type_info_(type_info),
      typecheck_(std::move(typecheck)),
      additional_search_paths_(additional_search_paths.begin(),
                               additional_search_paths.end()),
      import_cache_(import_cache),
      trace_all_(trace_all),
      ir_package_(ir_package),
      abstract_adapter_(absl::make_unique<AbstractInterpreterAdapter>(this)) {}

absl::StatusOr<InterpValue> Interpreter::RunFunction(
    absl::string_view name, absl::Span<const InterpValue> args,
    SymbolicBindings symbolic_bindings) {
  XLS_ASSIGN_OR_RETURN(Function * f, module_->GetFunctionOrError(name));
  Pos fake_pos("<fake>", 0, 0);
  Span fake_span(fake_pos, fake_pos);
  return EvaluateAndCompare(f, args, fake_span, /*expr=*/nullptr,
                            &symbolic_bindings);
}

absl::Status Interpreter::RunTest(absl::string_view name) {
  XLS_ASSIGN_OR_RETURN(InterpBindings bindings,
                       MakeTopLevelBindings(module_, abstract_adapter_.get()));
  XLS_ASSIGN_OR_RETURN(TestFunction * test, module_->GetTest(name));
  bindings.set_fn_ctx(
      FnCtx{module_->name(), absl::StrFormat("%s__test", name)});
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

/* static */ absl::StatusOr<int64> Interpreter::InterpretExpr(
    Module* entry_module, TypeInfo* type_info, TypecheckFn typecheck,
    absl::Span<std::string const> additional_search_paths,
    ImportCache* import_cache,
    const absl::flat_hash_map<std::string, int64>& env,
    const absl::flat_hash_map<std::string, int64>& bit_widths, Expr* expr,
    const FnCtx& fn_ctx) {
  XLS_VLOG(3) << "InterpretExpr: " << expr->ToString() << " env: {"
              << absl::StrJoin(env, ", ", absl::PairFormatter(":")) << "}";

  Interpreter interp(entry_module, type_info, typecheck,
                     additional_search_paths, import_cache);
  XLS_ASSIGN_OR_RETURN(
      InterpBindings bindings,
      MakeTopLevelBindings(entry_module, interp.abstract_adapter_.get()));
  bindings.set_fn_ctx(fn_ctx);
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
  XLS_ASSIGN_OR_RETURN(
      InterpValue result,
      interp.Evaluate(expr, &bindings, /*type_context=*/nullptr));
  return result.GetBitValueInt64();
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

absl::StatusOr<InterpValue> Interpreter::CallFnValue(
    const InterpValue& fv, absl::Span<InterpValue const> args, const Span& span,
    Invocation* invocation, const SymbolicBindings* symbolic_bindings) {
  if (fv.IsBuiltinFunction()) {
    auto builtin = absl::get<Builtin>(fv.GetFunctionOrDie());
    return RunBuiltin(builtin, args, span, invocation, symbolic_bindings);
  }
  const auto& fn_data =
      absl::get<InterpValue::UserFnData>(fv.GetFunctionOrDie());
  return EvaluateAndCompare(fn_data.function, args, span, invocation,
                            symbolic_bindings);
}

absl::Status Interpreter::RunJitComparison(
    Function* f, absl::Span<InterpValue const> args,
    const SymbolicBindings* symbolic_bindings) {
  // TODO(leary): 2020-11-20 Implement this, for now we lie and say the JIT
  // execution matched even though we don't run anything.
#if 0
  std::string ir_name = MangleDslxName(
      f->identifier(), f->GetFreeParametricKeys(), f->GetContainingModule(),
      symbolic_bindings);

  xls::Function* ir_function = ir_package_->GetFunction(ir_name);
  std::vector<Value> ir_args = ConvertArgsToIr(args);
  // TODO(leary): 2020-11-19 Cache JIT function so we don't have to create it
  // every time.
  XLS_ASSIGN_OR_RETURN(Value jit_value, IrJitRun(ir_function, ir_args));
  XLS_RETURN_IF_ERROR(CompareValues(interpreter_value, jit_value));
  return absl::OkStatus();
#endif
  return absl::OkStatus();
}

absl::StatusOr<InterpValue> Interpreter::EvaluateAndCompare(
    Function* f, absl::Span<const InterpValue> args, const Span& span,
    Invocation* expr, const SymbolicBindings* symbolic_bindings) {
  bool has_child_type_info =
      expr != nullptr &&
      type_info_->HasInstantiation(expr, symbolic_bindings == nullptr
                                             ? SymbolicBindings()
                                             : *symbolic_bindings);
  absl::optional<TypeInfo*> invocation_type_info;
  if (has_child_type_info) {
    invocation_type_info =
        type_info_->GetInstantiation(expr, *symbolic_bindings);
  } else {
    invocation_type_info = type_info_;
  }

  TypeInfoSwap tis(this, invocation_type_info);

  XLS_ASSIGN_OR_RETURN(
      InterpValue interpreter_value,
      EvaluateFunction(f, args, span,
                       symbolic_bindings == nullptr ? SymbolicBindings()
                                                    : *symbolic_bindings,
                       abstract_adapter_.get()));

  XLS_RETURN_IF_ERROR(RunJitComparison(f, args, symbolic_bindings));

  return interpreter_value;
}

absl::StatusOr<InterpValue> Interpreter::EvaluateInvocation(
    Invocation* expr, InterpBindings* bindings, ConcreteType* type_context) {
  XLS_VLOG(3) << absl::StreamFormat("EvaluateInvocation: `%s` @ %s",
                                    expr->ToString(), expr->span().ToString());
  std::vector<InterpValue> arg_values;
  for (Expr* arg : expr->args()) {
    XLS_ASSIGN_OR_RETURN(InterpValue arg_value,
                         Evaluate(arg, bindings, /*type_context=*/nullptr));
    arg_values.push_back(std::move(arg_value));
  }

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
        type_info_->GetInvocationSymbolicBindings(
            expr, bindings->fn_ctx()->sym_bindings);
    if (!callee_bindings.has_value()) {
      return absl::NotFoundError(
          absl::StrFormat("Could not find callee bindings in type info for "
                          "FnCtx: %s expr: %s @ %s",
                          bindings->fn_ctx()->ToString(), expr->ToString(),
                          expr->span().ToString()));
    }
    fn_symbolic_bindings = callee_bindings.value();
  } else {
    // Note, when there's no function context we may be in a ConstantDef doing
    // e.g. a parametric invocation.
    absl::optional<const SymbolicBindings*> callee_bindings =
        type_info_->GetInvocationSymbolicBindings(expr, SymbolicBindings());
    if (callee_bindings.has_value()) {
      fn_symbolic_bindings = callee_bindings.value();
    }
  }
  return CallFnValue(callee_value, arg_values, expr->span(), expr,
                     fn_symbolic_bindings);
}

bool Interpreter::IsWip(ConstantDef* c) const {
  auto it = wip_.find(c);
  return it != wip_.end() && !it->second.has_value();
}

absl::optional<InterpValue> Interpreter::NoteWip(
    ConstantDef* c, absl::optional<InterpValue> value) {
  if (!value.has_value()) {
    // Starting evaluation, attempting to mark as WIP.
    auto it = wip_.find(c);
    if (it != wip_.end() && it->second.has_value()) {
      return it->second;  // Already computed.
    }
    wip_[c] = absl::nullopt;  // Mark as WIP.
    return absl::nullopt;
  }

  wip_[c] = value;
  return value;
}

absl::StatusOr<InterpValue> SignConvertValue(const ConcreteType& concrete_type,
                                             const InterpValue& value) {
  if (auto* tuple_type = dynamic_cast<const TupleType*>(&concrete_type)) {
    XLS_RET_CHECK(value.IsTuple()) << value.ToString();
    const int64 tuple_size = value.GetValuesOrDie().size();
    std::vector<InterpValue> results;
    for (int64 i = 0; i < tuple_size; ++i) {
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
    int64 array_size = value.GetValuesOrDie().size();
    std::vector<InterpValue> results;
    for (int64 i = 0; i < array_size; ++i) {
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
  for (int64 i = 0; i < args.size(); ++i) {
    XLS_ASSIGN_OR_RETURN(InterpValue value,
                         SignConvertValue(*params[i], args[i]));
    converted.push_back(value);
  }
  return converted;
}

}  // namespace xls::dslx
