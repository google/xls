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

#include "xls/dslx/builtins.h"

#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/interp_value_helpers.h"
#include "xls/ir/bits_ops.h"

namespace xls::dslx {
namespace {

// Simple fluent object for checking args up front and creating error statuses
// on precondition violations.
class ArgChecker {
 public:
  ArgChecker(std::string name, absl::Span<const InterpValue> args)
      : name_(std::move(name)), args_(args) {}

  ArgChecker& size(int64_t target) {
    if (args_.size() != target) {
      status_.Update(absl::InvalidArgumentError(
          absl::StrFormat("Expect %d argument(s) to %s(); got %d", target,
                          name_, args_.size())));
    }
    return *this;
  }

  ArgChecker& size_ge(int64_t target) {
    if (args_.size() < target) {
      status_.Update(absl::InvalidArgumentError(
          absl::StrFormat("Expect >= %d argument(s) to %s(); got %d", target,
                          name_, args_.size())));
    }
    return *this;
  }

  ArgChecker& array(int64_t argno) {
    if (!args_[argno].IsArray()) {
      status_.Update(absl::InvalidArgumentError(
          absl::StrFormat("Expect argument %d to %s to be an array; got: %s",
                          argno, name_, TagToString(args_[argno].tag()))));
    }
    return *this;
  }

  ArgChecker& bits(int64_t argno) {
    if (!args_[argno].IsBits()) {
      status_.Update(absl::InvalidArgumentError(
          absl::StrFormat("Expect argument %d to %s to be bits; got: %s", argno,
                          name_, TagToString(args_[argno].tag()))));
    }
    return *this;
  }

  ArgChecker& ubits(int64_t argno, int64_t bit_count) {
    const InterpValue& arg = args_[argno];
    if (!arg.IsUBits() || arg.GetBitCount().value() != bit_count) {
      status_.Update(absl::InvalidArgumentError(absl::StrFormat(
          "Expect argument %d to %s to be uN[%d]; got: %s", argno, name_,
          bit_count, TagToString(args_[argno].tag()))));
    }
    return *this;
  }

  const absl::Status& status() const { return status_; }

 private:
  std::string name_;
  absl::Span<const InterpValue> args_;
  absl::Status status_;
};

// Helper that returns a failure error if pred (predicated) is false.
absl::StatusOr<InterpValue> FailUnless(const InterpValue& pred,
                                       std::string message, const Span& span) {
  if (pred.IsFalse()) {
    return FailureErrorStatus(span, message);
  }
  return InterpValue::MakeUnit();
}

}  // namespace

static void PerformTrace(absl::string_view text, const Span& span,
                         const InterpValue& value,
                         AbstractInterpreter* interp) {
  FormatPreference format = interp->GetTraceFormatPreference();
  XLS_LOG(INFO) << absl::StreamFormat("trace of %s @ %s: %s", text,
                                      span.ToString(),
                                      value.ToString(/*humanize=*/true,
                                                     /*format=*/format));
}

constexpr absl::string_view kTraceFmtErrorTemplate = R"(
trace_fmt! error: %s
Expression: %s @ %s
Actual arguments: (%s))";

static absl::Status PerformTraceFmt(FormatMacro* expr,
                                    absl::Span<const InterpValue> arg_values) {
  auto arg_value = arg_values.begin();
  auto make_error = [expr, &arg_values](std::string msg) -> absl::Status {
    std::string args_str = absl::StrJoin(
        arg_values, ", ", [](std::string* out, const InterpValue& v) {
          absl::StrAppend(out,
                          v.ToString(/*humanize=*/true,
                                     /*format=*/FormatPreference::kDefault));
        });
    return absl::InternalError(
        absl::StrFormat(kTraceFmtErrorTemplate, msg, expr->ToString(),
                        expr->span().ToString(), args_str));
  };

  std::string trace_output;
  for (auto step : expr->format()) {
    if (absl::holds_alternative<FormatPreference>(step)) {
      if (arg_value != arg_values.end()) {
        // Print via IR so that formatted traces will have the same output in
        // both interpreted and JIT execution.
        XLS_ASSIGN_OR_RETURN(Value ir_value, arg_value->ConvertToIr());
        trace_output +=
            ir_value.ToHumanString(absl::get<FormatPreference>(step));
        arg_value++;
      } else {
        return make_error("Not enough arguments for format string");
      }
    } else {
      trace_output += absl::get<std::string>(step);
    }
  }

  XLS_LOG(INFO) << trace_output;

  if (arg_value != arg_values.end()) {
    return make_error("Too many arguments for format string");
  }
  return absl::OkStatus();
}

void OptionalTrace(Expr* expr, const InterpValue& result,
                   AbstractInterpreter* interp) {
  // Implementation note: We don't need to trace the 'trace' invocation, or Let
  // nodes -- we just want to see the non-Let bodies.
  //
  // NameRefs and ColonRefs/ModRefs also add a lot of noise without a lot of
  // value.

  auto query_is_trace_instance = [expr] {
    auto* invocation = dynamic_cast<Invocation*>(expr);
    if (invocation == nullptr) {
      return false;
    }
    auto* callee = dynamic_cast<NameRef*>(invocation->callee());
    if (callee == nullptr) {
      return false;
    }
    return callee->identifier() == "trace!";
  };

  bool is_trace_instance = query_is_trace_instance();
  bool is_let_instance = dynamic_cast<Let*>(expr) != nullptr;

  if (!is_trace_instance && !is_let_instance && !result.IsFunction()) {
    PerformTrace(expr->ToString(), expr->span(), result, interp);
  }
}

absl::StatusOr<InterpValue> BuiltinTrace(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings, AbstractInterpreter* interp) {
  XLS_RETURN_IF_ERROR(ArgChecker("trace!", args).size(1).status());
  PerformTrace(expr->FormatArgs(), span, args[0], interp);
  return args[0];
}

absl::StatusOr<InterpValue> BuiltinTraceFmt(absl::Span<const InterpValue> args,
                                            FormatMacro* expr) {
  XLS_RETURN_IF_ERROR(PerformTraceFmt(expr, args));
  return InterpValue::MakeToken();
}

absl::StatusOr<InterpValue> BuiltinMap(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings, AbstractInterpreter* interp) {
  XLS_RETURN_IF_ERROR(ArgChecker("map", args).size(2).status());
  const InterpValue& inputs = args[0];

  // Note: this `mapped_fn` may not be from the current module (where the `expr`
  // -- i.e. the map invocation -- lives).
  const InterpValue& mapped_fn = args[1];

  XLS_VLOG(5) << "mapping; inputs: " << inputs.ToString()
              << "; mapped_fn: " << mapped_fn.ToString() << " @ " << span
              << "; current type info module: "
              << interp->GetCurrentTypeInfo()->module()->name();

  // On entry to BuiltinMap, EvaluateInvocation has set up the type information
  // if we're invoking a parametric. If it's not a parametric, but a plain
  // function in another module, we still need to set up the appropriate type
  // info here.
  //
  // TODO(leary): 2021-07-20 It'd be better to store the type info that should
  // be established for any given invocation in the TypeInfo map, but we treat
  // maps specially since they're currently our only higher order function.
  // Introduction of reduce (google/xls#331) will provide impetus to clean this
  // up more.
  absl::optional<AbstractInterpreter::ScopedTypeInfoSwap> stis;
  absl::optional<Module*> mapped_module = GetFunctionValueOwner(mapped_fn);
  if (mapped_module.has_value() &&
      mapped_module.value() != interp->GetCurrentTypeInfo()->module()) {
    XLS_ASSIGN_OR_RETURN(TypeInfo * mapped_type_info,
                         interp->GetRootTypeInfo(*mapped_module));
    XLS_RET_CHECK(mapped_type_info != nullptr);
    stis.emplace(interp, *mapped_type_info);
  }

  // Call "mapped_fn" on each of the input elements.
  std::vector<InterpValue> outputs;
  for (const InterpValue& v : inputs.GetValuesOrDie()) {
    std::vector<InterpValue> args = {v};
    XLS_ASSIGN_OR_RETURN(
        InterpValue result,
        interp->CallValue(mapped_fn, args, span, expr, symbolic_bindings));
    outputs.push_back(result);
  }
  return InterpValue::MakeArray(std::move(outputs));
}

absl::StatusOr<InterpValue> BuiltinFail(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings) {
  XLS_RETURN_IF_ERROR(ArgChecker("fail!", args).size(1).status());
  return FailureErrorStatus(span, args[0].ToString());
}

absl::StatusOr<InterpValue> BuiltinCover(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings) {
  XLS_RETURN_IF_ERROR(ArgChecker("cover!", args).size(2).array(0).status());
  return InterpValue::MakeUnit();
}

absl::StatusOr<InterpValue> BuiltinGate(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings) {
  XLS_RETURN_IF_ERROR(
      ArgChecker("gate!", args).size(2).ubits(0, 1).bits(1).status());
  const InterpValue& p = args[0];
  const InterpValue& value = args[1];
  if (p.IsTrue()) {
    return value;
  }
  return InterpValue::MakeBits(value.tag(), Bits(value.GetBitCount().value()));
}

absl::StatusOr<InterpValue> BuiltinUpdate(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings) {
  XLS_RETURN_IF_ERROR(
      ArgChecker("update", args).size(3).array(0).bits(1).status());
  const InterpValue& array = args[0];
  const InterpValue& index = args[1];
  const InterpValue& value = args[2];
  return array.Update(index, value);
}

absl::StatusOr<InterpValue> BuiltinAssertEq(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings) {
  XLS_RETURN_IF_ERROR(ArgChecker("assert_eq", args).size(2).status());
  const InterpValue& lhs = args[0];
  const InterpValue& rhs = args[1];
  if (lhs.Eq(rhs)) {
    return InterpValue::MakeUnit();
  }

  std::string message =
      absl::StrFormat("\n  lhs: %s\n  rhs: %s\n  were not equal",
                      lhs.ToHumanString(), rhs.ToHumanString());

  if (lhs.IsArray() && rhs.IsArray()) {
    XLS_ASSIGN_OR_RETURN(
        absl::optional<int64_t> i,
        FindFirstDifferingIndex(lhs.GetValuesOrDie(), rhs.GetValuesOrDie()));
    XLS_RET_CHECK(i.has_value());
    const auto& lhs_values = lhs.GetValuesOrDie();
    const auto& rhs_values = rhs.GetValuesOrDie();
    message += absl::StrFormat("; first differing index: %d :: %s vs %s", *i,
                               lhs_values[*i].ToHumanString(),
                               rhs_values[*i].ToHumanString());
  }

  return FailureErrorStatus(span, message);
}

absl::StatusOr<InterpValue> BuiltinAssertLt(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings) {
  XLS_RETURN_IF_ERROR(ArgChecker("assert_lt", args).size(2).status());
  const InterpValue& lhs = args[0];
  const InterpValue& rhs = args[1];
  XLS_ASSIGN_OR_RETURN(InterpValue pred, lhs.Lt(rhs));

  std::string message = absl::StrFormat(
      "\n  want: %s < %s", lhs.ToHumanString(), rhs.ToHumanString());
  return FailUnless(pred, message, span);
}

absl::StatusOr<InterpValue> BuiltinAndReduce(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings) {
  XLS_RETURN_IF_ERROR(ArgChecker("and_reduce", args).size(1).status());
  XLS_ASSIGN_OR_RETURN(Bits bits, args[0].GetBits());
  return InterpValue::MakeBits(InterpValueTag::kUBits,
                               bits_ops::AndReduce(bits));
}

absl::StatusOr<InterpValue> BuiltinOrReduce(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings) {
  XLS_RETURN_IF_ERROR(ArgChecker("or_reduce", args).size(1).status());
  XLS_ASSIGN_OR_RETURN(Bits bits, args[0].GetBits());
  return InterpValue::MakeBits(InterpValueTag::kUBits,
                               bits_ops::OrReduce(bits));
}

absl::StatusOr<InterpValue> BuiltinXorReduce(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings) {
  XLS_RETURN_IF_ERROR(ArgChecker("xor_reduce", args).size(1).status());
  XLS_ASSIGN_OR_RETURN(Bits bits, args[0].GetBits());
  return InterpValue::MakeBits(InterpValueTag::kUBits,
                               bits_ops::XorReduce(bits));
}

absl::StatusOr<InterpValue> BuiltinRev(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings) {
  XLS_RETURN_IF_ERROR(ArgChecker("rev", args).size(1).status());
  XLS_ASSIGN_OR_RETURN(Bits bits, args[0].GetBits());
  return InterpValue::MakeBits(InterpValueTag::kUBits, bits_ops::Reverse(bits));
}

absl::StatusOr<InterpValue> BuiltinEnumerate(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings) {
  XLS_RETURN_IF_ERROR(ArgChecker("enumerate", args).size(1).array(0).status());
  auto& values = args[0].GetValuesOrDie();
  std::vector<InterpValue> results;
  for (int64_t i = 0; i < values.size(); ++i) {
    auto ordinal = InterpValue::MakeUBits(/*bit_count=*/32, /*value=*/i);
    auto tuple = InterpValue::MakeTuple({ordinal, values[i]});
    results.push_back(tuple);
  }
  return InterpValue::MakeArray(results);
}

absl::StatusOr<InterpValue> BuiltinRange(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings) {
  XLS_RETURN_IF_ERROR(ArgChecker("range", args).size_ge(1).bits(0).status());
  absl::optional<InterpValue> start;
  absl::optional<InterpValue> limit;
  if (args.size() == 1) {
    limit = args[0];
    start = InterpValue::MakeUBits(/*bit_count=*/limit->GetBitCount().value(),
                                   /*value=*/0);
  } else if (args.size() == 2) {
    start = args[0];
    limit = args[1];
  } else {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Expected 1 or 2 arguments to range; got: %d", args.size()));
  }

  std::vector<InterpValue> elements;
  while (start->Lt(*limit).value().IsTrue()) {
    elements.push_back(*start);
    start = start
                ->Add(InterpValue::MakeUBits(
                    /*bit_count=*/limit->GetBitCount().value(), /*value=*/1))
                .value();
  }
  return InterpValue::MakeArray(std::move(elements));
}

absl::StatusOr<InterpValue> BuiltinBitSlice(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings) {
  XLS_RETURN_IF_ERROR(ArgChecker("bit_slice", args).size(3).status());
  const InterpValue& subject = args[0];
  const InterpValue& start = args[1];
  const InterpValue& width = args[2];
  XLS_ASSIGN_OR_RETURN(Bits subject_bits, subject.GetBits());
  XLS_ASSIGN_OR_RETURN(Bits start_bits, start.GetBits());
  XLS_ASSIGN_OR_RETURN(uint64_t start_index, start_bits.ToUint64());
  if (start_index >= subject_bits.bit_count()) {
    start_index = subject_bits.bit_count();
  }
  // Note: output size has to be derived from the types of the input argument,
  // so the bitwidth of the "width" argument is what determines the output size,
  // not the value. This is the "forcing a known-const to be available via the
  // type system" kind of trick before we can have explicit parametrics for
  // builtins.
  XLS_ASSIGN_OR_RETURN(int64_t bit_count, width.GetBitCount());
  return InterpValue::MakeBits(InterpValueTag::kUBits,
                               subject_bits.Slice(start_index, bit_count));
}

absl::StatusOr<InterpValue> BuiltinBitSliceUpdate(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings) {
  XLS_RETURN_IF_ERROR(ArgChecker("bit_slice_update", args)
                          .size(3)
                          .bits(0)
                          .bits(1)
                          .bits(2)
                          .status());
  const InterpValue& subject = args[0];
  const InterpValue& start = args[1];
  const InterpValue& update_value = args[2];
  XLS_ASSIGN_OR_RETURN(Bits subject_bits, subject.GetBits());
  XLS_ASSIGN_OR_RETURN(Bits start_bits, start.GetBits());
  XLS_ASSIGN_OR_RETURN(Bits update_value_bits, update_value.GetBits());
  if (bits_ops::UGreaterThanOrEqual(start_bits, subject_bits.bit_count())) {
    // Update is entirely out of bounds so no bits of the subject are updated.
    return InterpValue::MakeBits(InterpValueTag::kUBits, subject_bits);
  }
  XLS_ASSIGN_OR_RETURN(int64_t start_index, start_bits.ToUint64());
  return InterpValue::MakeBits(
      InterpValueTag::kUBits,
      bits_ops::BitSliceUpdate(subject_bits, start_index, update_value_bits));
}

absl::StatusOr<InterpValue> BuiltinSlice(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings) {
  XLS_RETURN_IF_ERROR(ArgChecker("slice", args).size(3).status());
  const InterpValue& array = args[0];
  const InterpValue& start = args[1];
  const InterpValue& length = args[2];
  return array.Slice(start, length);
}

absl::StatusOr<InterpValue> BuiltinAddWithCarry(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings) {
  XLS_RETURN_IF_ERROR(ArgChecker("add_with_carry", args).size(2).status());
  const InterpValue& lhs = args[0];
  const InterpValue& rhs = args[1];
  return lhs.AddWithCarry(rhs);
}

absl::StatusOr<InterpValue> BuiltinClz(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings) {
  XLS_RETURN_IF_ERROR(ArgChecker("clz", args).size(1).status());
  const InterpValue& arg = args[0];
  XLS_ASSIGN_OR_RETURN(Bits bits, arg.GetBits());
  return InterpValue::MakeUBits(/*bit_count=*/bits.bit_count(),
                                /*value=*/bits.CountLeadingZeros());
}

absl::StatusOr<InterpValue> BuiltinCtz(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings) {
  XLS_RETURN_IF_ERROR(ArgChecker("ctz", args).size(1).status());
  const InterpValue& arg = args[0];
  XLS_ASSIGN_OR_RETURN(Bits bits, arg.GetBits());
  return InterpValue::MakeUBits(/*bit_count=*/bits.bit_count(),
                                /*value=*/bits.CountTrailingZeros());
}

absl::StatusOr<InterpValue> BuiltinOneHot(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings) {
  XLS_RETURN_IF_ERROR(ArgChecker("one_hot", args).size(2).status());
  const InterpValue& arg = args[0];
  const InterpValue& lsb_prio = args[1];
  XLS_ASSIGN_OR_RETURN(Bits bits, arg.GetBits());
  XLS_ASSIGN_OR_RETURN(Bits lsb_prio_bits, lsb_prio.GetBits());
  auto f = lsb_prio_bits.IsAllOnes() ? bits_ops::OneHotLsbToMsb
                                     : bits_ops::OneHotMsbToLsb;
  Bits result = f(bits);
  return InterpValue::MakeBits(InterpValueTag::kUBits, result);
}

absl::StatusOr<InterpValue> BuiltinOneHotSel(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings) {
  XLS_RETURN_IF_ERROR(ArgChecker("one_hot_sel", args).size(2).status());
  const InterpValue& selector = args[0];
  const InterpValue& cases = args[1];
  XLS_ASSIGN_OR_RETURN(Bits selector_bits, selector.GetBits());
  XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* values,
                       cases.GetValues());
  if (values->empty()) {
    return absl::InvalidArgumentError(
        "At least one value to select is required.");
  }
  XLS_ASSIGN_OR_RETURN(int64_t result_bit_count, (*values)[0].GetBitCount());
  Bits accum(result_bit_count);
  for (int64_t i = 0; i < values->size(); ++i) {
    if (!selector_bits.Get(i)) {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(Bits i_bits, (*values)[i].GetBits());
    accum = bits_ops::Or(i_bits, accum);
  }
  return InterpValue::MakeBits((*values)[0].tag(), accum);
}

absl::StatusOr<InterpValue> BuiltinSignex(
    absl::Span<const InterpValue> args, const Span& span, Invocation* expr,
    const SymbolicBindings* symbolic_bindings) {
  XLS_RETURN_IF_ERROR(ArgChecker("signex", args).size(2).status());
  const InterpValue& lhs = args[0];
  const InterpValue& rhs = args[1];
  XLS_ASSIGN_OR_RETURN(int64_t new_bit_count, rhs.GetBitCount());
  XLS_ASSIGN_OR_RETURN(Bits lhs_bits, lhs.GetBits());
  return InterpValue::MakeBits(rhs.tag(),
                               bits_ops::SignExtend(lhs_bits, new_bit_count));
}

absl::Status FailureErrorStatus(const Span& span, absl::string_view message) {
  return absl::InternalError(absl::StrFormat(
      "FailureError: %s The program being interpreted failed! %s",
      span.ToString(), message));
}

}  // namespace xls::dslx
