// Copyright 2021 The XLS Authors
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

#include "xls/dslx/bytecode/bytecode_interpreter.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <ios>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/die_if_null.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/bytecode/builtins.h"
#include "xls/dslx/bytecode/bytecode.h"
#include "xls/dslx/bytecode/bytecode_cache_interface.h"
#include "xls/dslx/bytecode/bytecode_interpreter_options.h"
#include "xls/dslx/bytecode/frame.h"
#include "xls/dslx/bytecode/interpreter_stack.h"
#include "xls/dslx/dslx_builtins.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/proc_id.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/interp_value_utils.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system/unwrap_meta_type.h"
#include "xls/dslx/value_format_descriptor.h"
#include "xls/ir/big_int.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/format_strings.h"

namespace xls::dslx {
namespace {

// Returns the given InterpValue formatted using the given format descriptor (if
// it is not null).
absl::StatusOr<std::string> ToStringMaybeFormatted(
    const InterpValue& value,
    std::optional<ValueFormatDescriptor> value_fmt_desc,
    int64_t indentation = 0) {
  if (value_fmt_desc.has_value()) {
    XLS_ASSIGN_OR_RETURN(std::string value_str,
                         value.ToFormattedString(*value_fmt_desc, indentation));
    return std::string(indentation, ' ') + value_str;
  }
  return std::string(indentation, ' ') +
         value.ToString(/*humanize=*/false, FormatPreference::kDefault);
}

// Casts an InterpValue representable as Bits to a new InterpValue
// with the given BitsType.
absl::StatusOr<InterpValue> ResizeBitsValue(const InterpValue& from,
                                            const BitsLikeProperties& to_bits,
                                            const Type& to_type,
                                            bool is_checked,
                                            const Span& source_span,
                                            const FileTable& file_table) {
  VLOG(3) << "ResizeBitsValue; from: " << from << " to: " << to_type << " @ "
          << source_span.ToString(file_table);
  XLS_ASSIGN_OR_RETURN(int64_t to_bit_count, to_bits.size.GetAsInt64());
  XLS_ASSIGN_OR_RETURN(bool is_signed, to_bits.is_signed.GetAsBool());

  // Check if it fits.
  // Case A: to unsigned of N-bits
  //   Be within [0, 2^N)
  // Case B: to signed of N-bits
  //   Be within [-2^(N-1), 2^(N-1))
  if (is_checked) {
    bool does_fit = false;

    if (!is_signed) {
      does_fit = !from.IsNegative() && from.FitsInNBitsUnsigned(to_bit_count);
    } else if (is_signed && !from.IsNegative()) {
      does_fit = from.FitsInNBitsUnsigned(to_bit_count - 1);
    } else {  // to_bits->is_signed() && from.IsNegative()
      does_fit = from.FitsInNBitsSigned(to_bit_count);
    }

    if (!does_fit) {
      return CheckedCastErrorStatus(source_span, from, &to_type, file_table);
    }
  }

  Bits result_bits;

  int64_t from_bit_count = from.GetBits().value().bit_count();
  if (from_bit_count == to_bit_count) {
    result_bits = from.GetBitsOrDie();
  } else {
    if (from.IsSigned()) {
      // Despite the name, InterpValue::SignExt also shrinks.
      XLS_ASSIGN_OR_RETURN(InterpValue tmp, from.SignExt(to_bit_count));
      result_bits = tmp.GetBitsOrDie();
    } else {
      // Same for ZeroExt.
      XLS_ASSIGN_OR_RETURN(InterpValue tmp, from.ZeroExt(to_bit_count));
      result_bits = tmp.GetBitsOrDie();
    }
  }

  return InterpValue::MakeBits(is_signed, result_bits);
}

}  // namespace

// How much to indent the data value in the trace emitted when sending/receiving
// over a channel.
constexpr int64_t kChannelTraceIndentation = 2;

/* static */ absl::StatusOr<InterpValue> BytecodeInterpreter::Interpret(
    ImportData* import_data, BytecodeFunction* bf,
    const std::vector<InterpValue>& args,
    std::optional<InterpValueChannelManager*> channel_manager,
    const BytecodeInterpreterOptions& options) {
  BytecodeInterpreter interpreter(import_data,
                                  /*proc_id=*/std::nullopt, channel_manager,
                                  options);
  XLS_RETURN_IF_ERROR(interpreter.InitFrame(bf, args, bf->type_info()));
  XLS_RETURN_IF_ERROR(interpreter.Run());
  if (options.validate_final_stack_depth()) {
    XLS_RET_CHECK_EQ(interpreter.stack_.size(), 1);
  }
  return interpreter.stack_.PeekOrDie();
}

BytecodeInterpreter::BytecodeInterpreter(
    ImportData* import_data, const std::optional<ProcId>& proc_id,
    std::optional<InterpValueChannelManager*> channel_manager,
    const BytecodeInterpreterOptions& options)
    : import_data_(ABSL_DIE_IF_NULL(import_data)),
      proc_id_(proc_id),
      stack_(import_data_->file_table()),
      channel_manager_(channel_manager),
      options_(options) {}

absl::Status BytecodeInterpreter::InitFrame(BytecodeFunction* bf,
                                            absl::Span<const InterpValue> args,
                                            const TypeInfo* type_info) {
  XLS_RET_CHECK(frames_.empty());

  // In "mission mode" we expect type_info to be non-null in the frame, but for
  // bytecode-level testing we may not have an AST.
  if (type_info == nullptr && bf->owner() != nullptr) {
    type_info = import_data_->GetRootTypeInfo(bf->owner()).value();
  }
  frames_.push_back(Frame(bf,
                          std::vector<InterpValue>(args.begin(), args.end()),
                          type_info, std::nullopt, /*initial_args=*/{}));
  return absl::OkStatus();
}

/* static */ absl::StatusOr<std::unique_ptr<BytecodeInterpreter>>
BytecodeInterpreter::CreateUnique(
    ImportData* import_data, const std::optional<ProcId>& proc_id,
    BytecodeFunction* bf, const std::vector<InterpValue>& args,
    std::optional<InterpValueChannelManager*> channel_manager,
    const BytecodeInterpreterOptions& options) {
  auto interp = absl::WrapUnique(
      new BytecodeInterpreter(import_data, proc_id, channel_manager, options));
  XLS_RETURN_IF_ERROR(interp->InitFrame(bf, args, bf->type_info()));
  return interp;
}

absl::Status BytecodeInterpreter::Run(bool* progress_made) {
  blocked_channel_info_ = std::nullopt;
  while (!frames_.empty()) {
    Frame* frame = &frames_.back();
    while (frame->pc() < frame->bf()->bytecodes().size()) {
      const std::vector<Bytecode>& bytecodes = frame->bf()->bytecodes();
      const Bytecode& bytecode = bytecodes.at(frame->pc());
      VLOG(2) << "Bytecode: " << bytecode.ToString(file_table());
      VLOG(2) << std::hex << "PC: " << frame->pc() << " : "
              << bytecode.ToString(file_table());
      VLOG(3) << absl::StreamFormat(" - stack depth %d [%s]", stack_.size(),
                                    stack_.ToString());
      int64_t old_pc = frame->pc();
      XLS_RETURN_IF_ERROR(EvalNextInstruction());
      VLOG(3) << absl::StreamFormat(" - stack depth %d [%s]", stack_.size(),
                                    stack_.ToString());

      if (bytecode.op() == Bytecode::Op::kCall) {
        frame = &frames_.back();
      } else if (frame->pc() != old_pc + 1) {
        XLS_RET_CHECK(bytecodes.at(frame->pc()).op() == Bytecode::Op::kJumpDest)
            << "Jumping from PC " << old_pc << " to PC: " << frame->pc()
            << " bytecode: " << bytecodes.at(frame->pc()).ToString(file_table())
            << " not a jump_dest or old bytecode: "
            << bytecode.ToString(file_table()) << " was not a call op.";
      }
      if (progress_made != nullptr) {
        *progress_made = true;
      }
    }

    // Run the post-fn eval hook, if our function has a return value and if we
    // actually have a hook.
    const Function* source_fn = frame->bf()->source_fn();
    if (source_fn != nullptr) {
      std::optional<Type*> fn_return = frame->type_info()->GetItem(source_fn);
      if (fn_return.has_value()) {
        bool fn_returns_value = *fn_return.value() != *Type::MakeUnit();
        if (options_.post_fn_eval_hook() != nullptr && fn_returns_value) {
          ParametricEnv holder;
          const ParametricEnv* bindings = &holder;
          if (frame->bindings().has_value()) {
            bindings = &frame->bindings().value();
          }
          XLS_RETURN_IF_ERROR(options_.post_fn_eval_hook()(
              source_fn, frame->initial_args(), bindings, stack_.PeekOrDie()));
        }
      }
    }

    // We've reached the end of a function. Time to load the next frame up!
    frames_.pop_back();
  }

  return absl::OkStatus();
}

absl::StatusOr<std::vector<InterpValue>>
BytecodeInterpreter::PopArgsRightToLeft(size_t count) {
  std::vector<InterpValue> args(count, InterpValue::MakeToken());
  for (int i = 0; i < count; i++) {
    XLS_ASSIGN_OR_RETURN(InterpValue arg, Pop());
    args[count - i - 1] = arg;
  }
  return args;
}

absl::Status BytecodeInterpreter::EvalNextInstruction() {
  Frame* frame = &frames_.back();
  const std::vector<Bytecode>& bytecodes = frame->bf()->bytecodes();
  if (frame->pc() >= bytecodes.size()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Frame PC exceeds bytecode length: %d vs %d.",
                        frame->pc(), bytecodes.size()));
  }
  const Bytecode& bytecode = bytecodes.at(frame->pc());
  VLOG(10) << "Running bytecode: " << bytecode.ToString(file_table())
           << " depth before: " << stack_.size();
  switch (bytecode.op()) {
    case Bytecode::Op::kUAdd: {
      XLS_RETURN_IF_ERROR(EvalAdd(bytecode, /*is_signed=*/false));
      break;
    }
    case Bytecode::Op::kSAdd: {
      XLS_RETURN_IF_ERROR(EvalAdd(bytecode, /*is_signed=*/true));
      break;
    }
    case Bytecode::Op::kAnd: {
      XLS_RETURN_IF_ERROR(EvalAnd(bytecode));
      break;
    }
    case Bytecode::Op::kCall: {
      XLS_RETURN_IF_ERROR(EvalCall(bytecode));
      return absl::OkStatus();
    }
    case Bytecode::Op::kCast: {
      XLS_RETURN_IF_ERROR(EvalCast(bytecode));
      break;
    }
    case Bytecode::Op::kCheckedCast: {
      XLS_RETURN_IF_ERROR(EvalCast(bytecode, /*is_checked=*/true));
      break;
    }
    case Bytecode::Op::kConcat: {
      XLS_RETURN_IF_ERROR(EvalConcat(bytecode));
      break;
    }
    case Bytecode::Op::kCreateArray: {
      XLS_RETURN_IF_ERROR(EvalCreateArray(bytecode));
      break;
    }
    case Bytecode::Op::kCreateTuple: {
      XLS_RETURN_IF_ERROR(EvalCreateTuple(bytecode));
      break;
    }
    case Bytecode::Op::kDecode: {
      XLS_RETURN_IF_ERROR(EvalDecode(bytecode));
      break;
    }
    case Bytecode::Op::kDiv: {
      XLS_RETURN_IF_ERROR(EvalDiv(bytecode));
      break;
    }
    case Bytecode::Op::kDup: {
      XLS_RETURN_IF_ERROR(EvalDup(bytecode));
      break;
    }
    case Bytecode::Op::kEq: {
      XLS_RETURN_IF_ERROR(EvalEq(bytecode));
      break;
    }
    case Bytecode::Op::kExpandTuple: {
      XLS_RETURN_IF_ERROR(EvalExpandTuple(bytecode));
      break;
    }
    case Bytecode::Op::kFail: {
      XLS_RETURN_IF_ERROR(EvalFail(bytecode));
      break;
    }
    case Bytecode::Op::kGe: {
      XLS_RETURN_IF_ERROR(EvalGe(bytecode));
      break;
    }
    case Bytecode::Op::kGt: {
      XLS_RETURN_IF_ERROR(EvalGt(bytecode));
      break;
    }
    case Bytecode::Op::kIndex: {
      XLS_RETURN_IF_ERROR(EvalIndex(bytecode));
      break;
    }
    case Bytecode::Op::kTupleIndex: {
      XLS_RETURN_IF_ERROR(EvalTupleIndex(bytecode));
      break;
    }
    case Bytecode::Op::kInvert: {
      XLS_RETURN_IF_ERROR(EvalInvert(bytecode));
      break;
    }
    case Bytecode::Op::kJumpDest:
      break;
    case Bytecode::Op::kJumpRel: {
      XLS_ASSIGN_OR_RETURN(Bytecode::JumpTarget target, bytecode.jump_target());
      frame->set_pc(frame->pc() + target.value());
      return absl::OkStatus();
    }
    case Bytecode::Op::kJumpRelIf: {
      XLS_ASSIGN_OR_RETURN(std::optional<int64_t> new_pc,
                           EvalJumpRelIf(frame->pc(), bytecode));
      if (new_pc.has_value()) {
        frame->set_pc(new_pc.value());
        return absl::OkStatus();
      }
      break;
    }
    case Bytecode::Op::kLe: {
      XLS_RETURN_IF_ERROR(EvalLe(bytecode));
      break;
    }
    case Bytecode::Op::kLoad: {
      XLS_RETURN_IF_ERROR(EvalLoad(bytecode));
      break;
    }
    case Bytecode::Op::kLiteral: {
      XLS_RETURN_IF_ERROR(EvalLiteral(bytecode));
      break;
    }
    case Bytecode::Op::kLogicalAnd: {
      XLS_RETURN_IF_ERROR(EvalLogicalAnd(bytecode));
      break;
    }
    case Bytecode::Op::kLogicalOr: {
      XLS_RETURN_IF_ERROR(EvalLogicalOr(bytecode));
      break;
    }
    case Bytecode::Op::kLt: {
      XLS_RETURN_IF_ERROR(EvalLt(bytecode));
      break;
    }
    case Bytecode::Op::kMatchArm: {
      XLS_RETURN_IF_ERROR(EvalMatchArm(bytecode));
      break;
    }
    case Bytecode::Op::kSMul: {
      XLS_RETURN_IF_ERROR(EvalMul(bytecode, /*is_signed=*/true));
      break;
    }
    case Bytecode::Op::kUMul: {
      XLS_RETURN_IF_ERROR(EvalMul(bytecode, /*is_signed=*/false));
      break;
    }
    case Bytecode::Op::kMod: {
      XLS_RETURN_IF_ERROR(EvalMod(bytecode));
      break;
    }
    case Bytecode::Op::kNe: {
      XLS_RETURN_IF_ERROR(EvalNe(bytecode));
      break;
    }
    case Bytecode::Op::kNegate: {
      XLS_RETURN_IF_ERROR(EvalNegate(bytecode));
      break;
    }
    case Bytecode::Op::kOr: {
      XLS_RETURN_IF_ERROR(EvalOr(bytecode));
      break;
    }
    case Bytecode::Op::kPop: {
      XLS_RETURN_IF_ERROR(EvalPop(bytecode));
      break;
    }
    case Bytecode::Op::kRange: {
      XLS_RETURN_IF_ERROR(EvalRange(bytecode));
      break;
    }
    case Bytecode::Op::kRecv: {
      XLS_RETURN_IF_ERROR(EvalRecv(bytecode));
      break;
    }
    case Bytecode::Op::kRecvNonBlocking: {
      XLS_RETURN_IF_ERROR(EvalRecvNonBlocking(bytecode));
      break;
    }
    case Bytecode::Op::kSend: {
      XLS_RETURN_IF_ERROR(EvalSend(bytecode));
      break;
    }
    case Bytecode::Op::kShl: {
      XLS_RETURN_IF_ERROR(EvalShl(bytecode));
      break;
    }
    case Bytecode::Op::kShr: {
      XLS_RETURN_IF_ERROR(EvalShr(bytecode));
      break;
    }
    case Bytecode::Op::kSlice: {
      XLS_RETURN_IF_ERROR(EvalSlice(bytecode));
      break;
    }
    case Bytecode::Op::kSpawn: {
      XLS_RETURN_IF_ERROR(EvalSpawn(bytecode));
      stack_.Push(InterpValue::MakeUnit());
      break;
    }
    case Bytecode::Op::kStore: {
      XLS_RETURN_IF_ERROR(EvalStore(bytecode));
      break;
    }
    case Bytecode::Op::kUSub: {
      XLS_RETURN_IF_ERROR(EvalSub(bytecode, /*is_signed=*/false));
      break;
    }
    case Bytecode::Op::kSSub: {
      XLS_RETURN_IF_ERROR(EvalSub(bytecode, /*is_signed=*/true));
      break;
    }
    case Bytecode::Op::kSwap: {
      XLS_RETURN_IF_ERROR(EvalSwap(bytecode));
      break;
    }
    case Bytecode::Op::kTrace: {
      XLS_RETURN_IF_ERROR(EvalTrace(bytecode));
      break;
    }
    case Bytecode::Op::kWidthSlice: {
      XLS_RETURN_IF_ERROR(EvalWidthSlice(bytecode));
      break;
    }
    case Bytecode::Op::kXor: {
      XLS_RETURN_IF_ERROR(EvalXor(bytecode));
      break;
    }
  }
  frame->IncrementPc();
  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::EvalBinop(
    const std::function<absl::StatusOr<InterpValue>(
        const InterpValue& lhs, const InterpValue& rhs)>& op) {
  XLS_RET_CHECK_GE(stack_.size(), 2);
  XLS_ASSIGN_OR_RETURN(InterpValue rhs, Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue lhs, Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue result, op(lhs, rhs));
  stack_.Push(std::move(result));
  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::EvalAdd(const Bytecode& bytecode,
                                          bool is_signed) {
  return EvalBinop([&](const InterpValue& lhs,
                       const InterpValue& rhs) -> absl::StatusOr<InterpValue> {
    XLS_ASSIGN_OR_RETURN(InterpValue output, lhs.Add(rhs));

    // Slow path: when rollover warning hook is enabled.
    if (options_.rollover_hook() != nullptr) {
      auto make_big_int = [is_signed](const Bits& bits) {
        return is_signed ? BigInt::MakeSigned(bits)
                         : BigInt::MakeUnsigned(bits);
      };
      bool rollover =
          make_big_int(lhs.GetBitsOrDie()) + make_big_int(rhs.GetBitsOrDie()) !=
          make_big_int(output.GetBitsOrDie());
      if (rollover) {
        options_.rollover_hook()(bytecode.source_span());
      }
    }

    return output;
  });
}

absl::Status BytecodeInterpreter::EvalAnd(const Bytecode& bytecode) {
  return EvalBinop([](const InterpValue& lhs, const InterpValue& rhs) {
    return lhs.BitwiseAnd(rhs);
  });
}

absl::StatusOr<BytecodeFunction*> BytecodeInterpreter::GetBytecodeFn(
    Function& f, const Invocation* invocation,
    const ParametricEnv& caller_bindings) {
  const Frame& frame = frames_.back();
  const TypeInfo* caller_type_info = frame.type_info();

  BytecodeCacheInterface* cache = import_data_->bytecode_cache();
  XLS_RET_CHECK(cache != nullptr);

  std::optional<ParametricEnv> callee_bindings;

  TypeInfo* callee_type_info = nullptr;
  if (f.IsParametric() || f.tag() == FunctionTag::kProcInit) {
    if (!caller_type_info->GetRootInvocations().contains(invocation)) {
      return absl::InternalError(absl::StrFormat(
          "BytecodeInterpreter::GetBytecodeFn; could not find information for "
          "invocation `%s` "
          "callee: %s (tag: %v), caller_bindings: %s span: %s",
          invocation->ToString(), f.identifier(), f.tag(),
          caller_bindings.ToString(),
          invocation->span().ToString(file_table())));
    }

    const InvocationData& invocation_data =
        caller_type_info->GetRootInvocations().at(invocation);
    XLS_RET_CHECK(
        invocation_data.env_to_callee_data().contains(caller_bindings))
        << "invocation: `" << invocation_data.node()->ToString() << "` @ "
        << invocation_data.node()->span().ToString(file_table()) << " caller: `"
        << invocation_data.caller()->identifier() << "`"
        << " caller_bindings: " << caller_bindings;

    const InvocationCalleeData& callee_data =
        invocation_data.env_to_callee_data().at(caller_bindings);
    callee_type_info = callee_data.derived_type_info;
    callee_bindings = callee_data.callee_bindings;
  } else {
    // If it's NOT parametric, then we need the root TypeInfo for the new
    // module.
    XLS_ASSIGN_OR_RETURN(callee_type_info,
                         import_data_->GetRootTypeInfo(f.owner()));
  }

  return cache->GetOrCreateBytecodeFunction(f, callee_type_info,
                                            callee_bindings);
}

absl::Status BytecodeInterpreter::EvalCall(const Bytecode& bytecode) {
  VLOG(3) << "BytecodeInterpreter::EvalCall: "
          << bytecode.ToString(file_table());
  XLS_ASSIGN_OR_RETURN(InterpValue callee, Pop());
  if (callee.IsBuiltinFunction()) {
    frames_.back().IncrementPc();
    return RunBuiltinFn(bytecode, std::get<Builtin>(callee.GetFunctionOrDie()));
  }

  XLS_ASSIGN_OR_RETURN(const InterpValue::FnData* fn_data,
                       callee.GetFunction());
  InterpValue::UserFnData user_fn_data =
      std::get<InterpValue::UserFnData>(*fn_data);
  XLS_ASSIGN_OR_RETURN(Bytecode::InvocationData data,
                       bytecode.invocation_data());

  const ParametricEnv& caller_bindings = data.caller_bindings().has_value()
                                             ? data.caller_bindings().value()
                                             : ParametricEnv();
  XLS_ASSIGN_OR_RETURN(BytecodeFunction * bf,
                       GetBytecodeFn(*user_fn_data.function, data.invocation(),
                                     caller_bindings));

  // Store the _return_ PC.
  frames_.back().IncrementPc();

  // If `user_fn` is a method (first arg is `self`), then the first arg will be
  // the most recent value pushed. Handle that case first.
  std::optional<InterpValue> first_arg;
  int remaining_args = user_fn_data.function->params().size();
  if (user_fn_data.function->IsMethod()) {
    XLS_ASSIGN_OR_RETURN(first_arg, Pop());
    remaining_args--;
  }
  XLS_ASSIGN_OR_RETURN(std::vector<InterpValue> args,
                       PopArgsRightToLeft(remaining_args));
  if (first_arg.has_value()) {
    args.insert(args.begin(), *first_arg);
  }

  std::vector<InterpValue> args_copy = args;
  frames_.push_back(Frame(bf, std::move(args), bf->type_info(),
                          data.callee_bindings(), std::move(args_copy)));

  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::EvalCast(const Bytecode& bytecode,
                                           bool is_checked) {
  if (!bytecode.data().has_value() ||
      !std::holds_alternative<std::unique_ptr<Type>>(bytecode.data().value())) {
    return absl::InternalError("Cast op requires Type data.");
  }

  XLS_ASSIGN_OR_RETURN(InterpValue from_value, Pop());

  if (!bytecode.data().has_value()) {
    return absl::InternalError("Cast op is missing its data element!");
  }

  const Type& to = *std::get<std::unique_ptr<Type>>(bytecode.data().value());
  if (from_value.IsArray()) {
    // From array to bits.
    std::optional<BitsLikeProperties> to_bits_like = GetBitsLike(to);
    if (!to_bits_like.has_value()) {
      return absl::InvalidArgumentError(
          "Array types can only be cast to bits.");
    }
    XLS_ASSIGN_OR_RETURN(InterpValue converted, from_value.Flatten());

    // Soundness check that the "to" type has the same number of bits as our new
    // flattened value.
    XLS_RET_CHECK_EQ(converted.GetBitCount().value(),
                     to_bits_like->size.GetAsInt64().value());

    stack_.Push(converted);
    return absl::OkStatus();
  }

  if (from_value.IsEnum()) {
    // From enum to bits.
    std::optional<BitsLikeProperties> to_bits_like = GetBitsLike(to);
    if (!to_bits_like.has_value()) {
      return absl::InvalidArgumentError("Enum types can only be cast to bits.");
    }

    XLS_ASSIGN_OR_RETURN(
        InterpValue result,
        ResizeBitsValue(from_value, to_bits_like.value(), to, is_checked,
                        bytecode.source_span(), file_table()));
    stack_.Push(result);
    return absl::OkStatus();
  }

  if (!from_value.IsBits()) {
    return absl::InvalidArgumentError(
        "Bytecode interpreter only supports casts from arrays, enums, and "
        "bits; got: " +
        from_value.ToString());
  }

  int64_t from_bit_count = from_value.GetBits().value().bit_count();

  // If the thing we're casting from is bits like, and the thing we're casting
  // to is bits, like, we use the `ResizeBitsValue` helper.
  if (std::optional<BitsLikeProperties> to_bits_like = GetBitsLike(to);
      to_bits_like.has_value()) {
    XLS_ASSIGN_OR_RETURN(
        InterpValue result,
        ResizeBitsValue(from_value, to_bits_like.value(), to, is_checked,
                        bytecode.source_span(), file_table()));
    stack_.Push(result);
    return absl::OkStatus();
  }

  // From bits to array.
  if (const ArrayType* to_array = dynamic_cast<const ArrayType*>(&to);
      to_array != nullptr) {
    XLS_ASSIGN_OR_RETURN(int64_t to_bit_count,
                         to_array->GetTotalBitCount().value().GetAsInt64());
    if (from_bit_count != to_bit_count) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Cast to array had mismatching bit counts: from %d to %d.",
          from_bit_count, to_bit_count));
    }
    XLS_ASSIGN_OR_RETURN(InterpValue casted,
                         CastBitsToArray(from_value, *to_array));
    stack_.Push(casted);
    return absl::OkStatus();
  }

  // From bits to enum.
  if (const EnumType* to_enum = dynamic_cast<const EnumType*>(&to);
      to_enum != nullptr) {
    XLS_ASSIGN_OR_RETURN(InterpValue converted,
                         CastBitsToEnum(from_value, *to_enum));
    stack_.Push(converted);
    return absl::OkStatus();
  }

  return absl::UnimplementedError(absl::StrFormat(
      "BytecodeInterpreter; cast of value %s to type %s is not yet implemented",
      from_value.ToString(), to.ToString()));
}

absl::Status BytecodeInterpreter::EvalConcat(const Bytecode& bytecode) {
  return EvalBinop([](const InterpValue& lhs, const InterpValue& rhs) {
    return lhs.Concat(rhs);
  });
}

absl::Status BytecodeInterpreter::EvalCreateArray(const Bytecode& bytecode) {
  XLS_ASSIGN_OR_RETURN(Bytecode::NumElements array_size,
                       bytecode.num_elements());
  XLS_RET_CHECK_GE(stack_.size(), array_size.value());

  std::vector<InterpValue> elements;
  elements.reserve(array_size.value());
  for (int64_t i = 0; i < array_size.value(); i++) {
    XLS_ASSIGN_OR_RETURN(InterpValue value, Pop());
    elements.push_back(value);
  }

  std::reverse(elements.begin(), elements.end());
  XLS_ASSIGN_OR_RETURN(InterpValue array, InterpValue::MakeArray(elements));
  stack_.Push(array);
  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::EvalCreateTuple(const Bytecode& bytecode) {
  XLS_ASSIGN_OR_RETURN(Bytecode::NumElements tuple_size,
                       bytecode.num_elements());
  XLS_RET_CHECK_GE(stack_.size(), tuple_size.value());

  std::vector<InterpValue> elements;
  elements.reserve(tuple_size.value());
  for (int64_t i = 0; i < tuple_size.value(); i++) {
    XLS_ASSIGN_OR_RETURN(InterpValue value, Pop());
    elements.push_back(value);
  }

  std::reverse(elements.begin(), elements.end());

  stack_.Push(InterpValue::MakeTuple(elements));
  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::EvalDecode(const Bytecode& bytecode) {
  if (!bytecode.data().has_value() ||
      !std::holds_alternative<std::unique_ptr<Type>>(bytecode.data().value())) {
    return absl::InternalError("Decode op requires Type data.");
  }

  XLS_ASSIGN_OR_RETURN(InterpValue from, Pop());
  if (!from.IsBits() || from.IsSigned()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Decode op requires UBits-type input, was: ", from.ToString()));
  }

  Type* to = std::get<std::unique_ptr<Type>>(bytecode.data().value()).get();
  std::optional<BitsLikeProperties> to_bits_like = GetBitsLike(*to);
  if (!to_bits_like.has_value()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Decode op requires bits-like output type, was: ", to->ToString()));
  }
  XLS_ASSIGN_OR_RETURN(bool is_signed, to_bits_like->is_signed.GetAsBool());
  XLS_ASSIGN_OR_RETURN(int64_t new_bit_count, to_bits_like->size.GetAsInt64());
  if (is_signed) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Decode op requires UBits-type output, was: ", to->ToString()));
  }

  XLS_ASSIGN_OR_RETURN(InterpValue decoded, from.Decode(new_bit_count));
  stack_.Push(std::move(decoded));
  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::EvalDiv(const Bytecode& bytecode) {
  return EvalBinop([](const InterpValue& lhs, const InterpValue& rhs) {
    return lhs.FloorDiv(rhs);
  });
}

absl::Status BytecodeInterpreter::EvalMod(const Bytecode& bytecode) {
  return EvalBinop([](const InterpValue& lhs, const InterpValue& rhs) {
    return lhs.FloorMod(rhs);
  });
}

absl::Status BytecodeInterpreter::EvalDup(const Bytecode& bytecode) {
  XLS_RET_CHECK(!stack_.empty());
  stack_.Push(stack_.PeekOrDie());
  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::EvalEq(const Bytecode& bytecode) {
  return EvalBinop([](const InterpValue& lhs, const InterpValue& rhs) {
    return InterpValue::MakeBool(lhs.Eq(rhs));
  });
}

absl::Status BytecodeInterpreter::EvalExpandTuple(const Bytecode& bytecode) {
  XLS_ASSIGN_OR_RETURN(InterpValue tuple, stack_.Pop());
  if (!tuple.IsTuple()) {
    return FailureErrorStatus(
        bytecode.source_span(),
        absl::StrCat("Stack top for ExpandTuple was not a tuple, was: ",
                     TagToString(tuple.tag())),
        file_table());
  }

  // Note that we destructure the tuple in "reverse" order, with the first
  // element on top of the stack.
  XLS_ASSIGN_OR_RETURN(int64_t tuple_size, tuple.GetLength());
  for (int64_t i = tuple_size - 1; i >= 0; i--) {
    XLS_ASSIGN_OR_RETURN(InterpValue element,
                         tuple.Index(InterpValue::MakeUBits(64, i)));
    stack_.Push(element);
  }

  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::EvalFail(const Bytecode& bytecode) {
  XLS_ASSIGN_OR_RETURN(const Bytecode::TraceData* trace_data,
                       bytecode.trace_data());
  XLS_ASSIGN_OR_RETURN(std::string message,
                       TraceDataToString(*trace_data, stack_));
  return FailureErrorStatus(bytecode.source_span(), message, file_table());
}

absl::Status BytecodeInterpreter::EvalGe(const Bytecode& bytecode) {
  return EvalBinop([](const InterpValue& lhs, const InterpValue& rhs) {
    return lhs.Ge(rhs);
  });
}

absl::Status BytecodeInterpreter::EvalGt(const Bytecode& bytecode) {
  return EvalBinop([](const InterpValue& lhs, const InterpValue& rhs) {
    return lhs.Gt(rhs);
  });
}

absl::Status BytecodeInterpreter::EvalTupleIndex(const Bytecode& bytecode) {
  XLS_ASSIGN_OR_RETURN(InterpValue index, Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue basis, Pop());

  if (!basis.IsTuple()) {
    return absl::InternalError(
        "BytecodeInterpreter type error: tuple_index bytecode can only index "
        "on tuple value; got: " +
        basis.ToString());
  }

  XLS_ASSIGN_OR_RETURN(
      InterpValue result, basis.Index(index),
      _ << " while processing " << bytecode.ToString(file_table()));
  stack_.Push(result);
  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::EvalIndex(const Bytecode& bytecode) {
  XLS_ASSIGN_OR_RETURN(InterpValue index, Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue basis, Pop());

  if (!basis.IsArray() && !basis.IsTuple()) {
    return absl::InternalError(
        "BytecodeInterpreter type error: can only index on array or tuple "
        "values; got: " +
        basis.ToString());
  }

  XLS_ASSIGN_OR_RETURN(
      InterpValue result, basis.Index(index),
      _ << " while processing "
        << bytecode.ToString(file_table(), /*source_locs=*/true));
  stack_.Push(result);
  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::EvalInvert(const Bytecode& bytecode) {
  XLS_ASSIGN_OR_RETURN(InterpValue operand, Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue result, operand.BitwiseNegate());
  stack_.Push(result);
  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::EvalLe(const Bytecode& bytecode) {
  return EvalBinop([](const InterpValue& lhs, const InterpValue& rhs) {
    return lhs.Le(rhs);
  });
}

absl::Status BytecodeInterpreter::EvalLiteral(const Bytecode& bytecode) {
  XLS_ASSIGN_OR_RETURN(InterpValue value, bytecode.value_data());
  stack_.PushFormattedValue(InterpreterStack::FormattedInterpValue{
      .value = std::move(value),
      .format_descriptor = bytecode.format_descriptor()});
  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::EvalLoad(const Bytecode& bytecode) {
  XLS_ASSIGN_OR_RETURN(Bytecode::SlotIndex slot, bytecode.slot_index());
  if (frames_.back().slots().size() <= slot.value()) {
    return absl::InternalError(absl::StrFormat(
        "Attempted to access local data in slot %d, which is out of range.",
        slot.value()));
  }
  stack_.Push(frames_.back().slots().at(slot.value()));
  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::EvalLogicalAnd(const Bytecode& bytecode) {
  XLS_ASSIGN_OR_RETURN(InterpValue rhs, Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue lhs, Pop());

  if (!lhs.HasBits() || lhs.GetBitsOrDie().bit_count() != 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("Logical AND arguments must be boolean (LHS had ",
                     lhs.GetBitsOrDie().bit_count(), " bits)."));
  }

  if (!rhs.HasBits() || rhs.GetBitsOrDie().bit_count() != 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("Logical AND arguments must be boolean (RHS had ",
                     rhs.GetBitsOrDie().bit_count(), " bits)."));
  }

  XLS_ASSIGN_OR_RETURN(InterpValue result, lhs.BitwiseAnd(rhs));
  stack_.Push(result);
  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::EvalLogicalOr(const Bytecode& bytecode) {
  XLS_ASSIGN_OR_RETURN(InterpValue rhs, Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue lhs, Pop());

  if (!lhs.HasBits() || lhs.GetBitsOrDie().bit_count() != 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("Logical OR arguments must be boolean (LHS had ",
                     lhs.GetBitsOrDie().bit_count(), " bits)."));
  }

  if (!rhs.HasBits() || rhs.GetBitsOrDie().bit_count() != 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("Logical OR arguments must be boolean (RHS had ",
                     rhs.GetBitsOrDie().bit_count(), " bits)."));
  }

  XLS_ASSIGN_OR_RETURN(InterpValue result, lhs.BitwiseOr(rhs));
  stack_.Push(result);
  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::EvalLt(const Bytecode& bytecode) {
  return EvalBinop([](const InterpValue& lhs, const InterpValue& rhs) {
    return lhs.Lt(rhs);
  });
}

absl::StatusOr<bool> BytecodeInterpreter::MatchArmEqualsInterpValue(
    Frame* frame, const Bytecode::MatchArmItem& item,
    const InterpValue& value) {
  using Kind = Bytecode::MatchArmItem::Kind;
  switch (item.kind()) {
    case Kind::kInterpValue: {
      XLS_ASSIGN_OR_RETURN(InterpValue arm_value, item.interp_value());
      return arm_value.Eq(value);
    }
    case Kind::kRange: {
      XLS_ASSIGN_OR_RETURN(Bytecode::MatchArmItem::RangeData range,
                           item.range());
      VLOG(10) << "value: " << value.ToString()
               << " start: " << range.start.ToString()
               << " limit: " << range.limit.ToString();
      XLS_ASSIGN_OR_RETURN(InterpValue val_ge_start, value.Ge(range.start));
      XLS_ASSIGN_OR_RETURN(InterpValue val_lt_limit, value.Lt(range.limit));
      XLS_ASSIGN_OR_RETURN(InterpValue conjunction,
                           val_ge_start.BitwiseAnd(val_lt_limit));
      VLOG(10) << "val_ge_start: " << val_ge_start.ToString()
               << " val_lt_limit: " << val_lt_limit.ToString()
               << " conjunction: " << conjunction.ToString();
      return conjunction.IsTrue();
    }
    case Kind::kLoad: {
      XLS_ASSIGN_OR_RETURN(Bytecode::SlotIndex slot_index, item.slot_index());
      if (frame->slots().size() <= slot_index.value()) {
        return absl::InternalError(absl::StrCat(
            "MatchArm load item index was OOB: ", slot_index.value(), " vs. ",
            frame->slots().size(), "."));
      }
      InterpValue arm_value = frame->slots().at(slot_index.value());
      return arm_value.Eq(value);
    }

    case Kind::kStore: {
      XLS_ASSIGN_OR_RETURN(Bytecode::SlotIndex slot_index, item.slot_index());
      frame->StoreSlot(slot_index, value);
      return true;
    }

    case Kind::kWildcard:
      return true;

    case Kind::kRestOfTuple:
      return true;

    case Kind::kTuple: {
      // Otherwise, we're a tuple. Recurse.
      XLS_ASSIGN_OR_RETURN(auto item_elements, item.tuple_elements());
      XLS_ASSIGN_OR_RETURN(auto* value_elements, value.GetValues());
      if (item_elements.size() != value_elements->size()) {
        return absl::InternalError(
            absl::StrCat("Match arm item had a different number of elements "
                         "than the corresponding InterpValue: ",
                         item.ToString(), " vs. ", value.ToString()));
      }

      // We don't have to deal with rest-of-tuple processing because
      // the matcher will have already handled that.
      for (int i = 0; i < item_elements.size(); i++) {
        XLS_ASSIGN_OR_RETURN(bool equal, MatchArmEqualsInterpValue(
                                             &frames_.back(), item_elements[i],
                                             value_elements->at(i)));
        if (!equal) {
          return false;
        }
      }

      return true;
    }
  }
}

absl::Status BytecodeInterpreter::EvalMatchArm(const Bytecode& bytecode) {
  // Puts true on the stack if the items are equal and false otherwise.
  XLS_ASSIGN_OR_RETURN(const Bytecode::MatchArmItem* item,
                       bytecode.match_arm_item());
  XLS_ASSIGN_OR_RETURN(InterpValue matchee, Pop());
  XLS_ASSIGN_OR_RETURN(
      bool equal, MatchArmEqualsInterpValue(&frames_.back(), *item, matchee));
  stack_.Push(InterpValue::MakeBool(equal));
  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::EvalMul(const Bytecode& bytecode,
                                          bool is_signed) {
  return EvalBinop([&](const InterpValue& lhs,
                       const InterpValue& rhs) -> absl::StatusOr<InterpValue> {
    XLS_ASSIGN_OR_RETURN(InterpValue output, lhs.Mul(rhs));

    // Slow path: when rollover warning hook is enabled.
    if (options_.rollover_hook() != nullptr) {
      auto make_big_int = [is_signed](const Bits& bits) {
        return is_signed ? BigInt::MakeSigned(bits)
                         : BigInt::MakeUnsigned(bits);
      };
      bool rollover =
          make_big_int(lhs.GetBitsOrDie()) * make_big_int(rhs.GetBitsOrDie()) !=
          make_big_int(output.GetBitsOrDie());
      if (rollover) {
        options_.rollover_hook()(bytecode.source_span());
      }
    }

    return output;
  });
}

absl::Status BytecodeInterpreter::EvalNe(const Bytecode& bytecode) {
  return EvalBinop([](const InterpValue& lhs, const InterpValue& rhs) {
    return InterpValue::MakeBool(lhs.Ne(rhs));
  });
}

absl::Status BytecodeInterpreter::EvalNegate(const Bytecode& bytecode) {
  XLS_ASSIGN_OR_RETURN(InterpValue operand, Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue result, operand.ArithmeticNegate());
  stack_.Push(result);
  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::EvalOr(const Bytecode& bytecode) {
  return EvalBinop([](const InterpValue& lhs, const InterpValue& rhs) {
    return lhs.BitwiseOr(rhs);
  });
}

absl::Status BytecodeInterpreter::EvalPop(const Bytecode& bytecode) {
  return Pop().status();
}

absl::Status BytecodeInterpreter::EvalRange(const Bytecode& bytecode) {
  return BuiltinRangeInternal(stack_);
}

absl::Status BytecodeInterpreter::EvalRecvNonBlocking(
    const Bytecode& bytecode) {
  XLS_ASSIGN_OR_RETURN(InterpValue default_value, Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue condition, Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue channel_value, Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue::ChannelReference channel_reference,
                       channel_value.GetChannelReference());
  XLS_ASSIGN_OR_RETURN(InterpValue token, Pop());

  XLS_RET_CHECK(channel_reference.GetChannelId().has_value());
  int64_t channel_id = channel_reference.GetChannelId().value();
  XLS_RET_CHECK(channel_manager_.has_value());
  InterpValueChannel& channel = (*channel_manager_)->GetChannel(channel_id);

  XLS_ASSIGN_OR_RETURN(const Bytecode::ChannelData* channel_data,
                       bytecode.channel_data());
  if (condition.IsTrue() && !channel.IsEmpty()) {
    InterpValue value = channel.Read();
    if (options_.trace_channels() && options_.trace_hook()) {
      XLS_ASSIGN_OR_RETURN(
          std::string formatted_data,
          ToStringMaybeFormatted(value, channel_data->value_fmt_desc(),
                                 kChannelTraceIndentation));
      options_.trace_hook()(
          bytecode.source_span(),
          absl::StrFormat("Received data on channel `%s`:\n%s",
                          FormatChannelNameForTracing(*channel_data),
                          formatted_data));
    }
    stack_.Push(InterpValue::MakeTuple(
        {token, std::move(value), InterpValue::MakeBool(true)}));
  } else {
    stack_.Push(InterpValue::MakeTuple(
        {token, default_value, InterpValue::MakeBool(false)}));
  }

  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::EvalRecv(const Bytecode& bytecode) {
  XLS_ASSIGN_OR_RETURN(InterpValue default_value, Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue condition, Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue channel_value, Pop());
  XLS_ASSIGN_OR_RETURN(auto channel_reference,
                       channel_value.GetChannelReference());
  XLS_ASSIGN_OR_RETURN(const Bytecode::ChannelData* channel_data,
                       bytecode.channel_data());

  XLS_RET_CHECK(channel_reference.GetChannelId().has_value());
  int64_t channel_id = channel_reference.GetChannelId().value();
  XLS_RET_CHECK(channel_manager_.has_value());
  InterpValueChannel& channel = (*channel_manager_)->GetChannel(channel_id);

  if (condition.IsTrue()) {
    if (channel.IsEmpty()) {
      // Restore the stack!
      stack_.Push(channel_value);
      stack_.Push(condition);
      stack_.Push(default_value);
      blocked_channel_info_ = BlockedChannelInfo{
          .name = FormatChannelNameForTracing(*channel_data),
          .span = bytecode.source_span(),
      };
      return absl::UnavailableError("Channel is empty.");
    }

    XLS_ASSIGN_OR_RETURN(InterpValue token, Pop());
    InterpValue value = channel.Read();
    if (options_.trace_channels() && options_.trace_hook()) {
      XLS_ASSIGN_OR_RETURN(
          std::string formatted_data,
          ToStringMaybeFormatted(value, channel_data->value_fmt_desc(),
                                 kChannelTraceIndentation));
      options_.trace_hook()(
          bytecode.source_span(),
          absl::StrFormat("Received data on channel `%s`:\n%s",
                          FormatChannelNameForTracing(*channel_data),
                          formatted_data));
    }
    stack_.Push(InterpValue::MakeTuple({token, std::move(value)}));
  } else {
    XLS_ASSIGN_OR_RETURN(InterpValue token, Pop());
    stack_.Push(InterpValue::MakeTuple({token, default_value}));
  }

  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::EvalSend(const Bytecode& bytecode) {
  XLS_ASSIGN_OR_RETURN(InterpValue condition, Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue payload, Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue channel_value, Pop());
  XLS_ASSIGN_OR_RETURN(auto channel_reference,
                       channel_value.GetChannelReference());
  XLS_ASSIGN_OR_RETURN(InterpValue token, Pop());

  XLS_RET_CHECK(channel_reference.GetChannelId().has_value());
  int64_t channel_id = channel_reference.GetChannelId().value();
  XLS_RET_CHECK(channel_manager_.has_value());
  InterpValueChannel& channel = (*channel_manager_)->GetChannel(channel_id);

  if (condition.IsTrue()) {
    if (options_.trace_channels() && options_.trace_hook()) {
      XLS_ASSIGN_OR_RETURN(const Bytecode::ChannelData* channel_data,
                           bytecode.channel_data());
      XLS_ASSIGN_OR_RETURN(
          std::string formatted_data,
          ToStringMaybeFormatted(payload, channel_data->value_fmt_desc(),
                                 kChannelTraceIndentation));
      options_.trace_hook()(
          bytecode.source_span(),
          absl::StrFormat("Sent data on channel `%s`:\n%s",
                          FormatChannelNameForTracing(*channel_data),
                          formatted_data));
    }
    channel.Write(payload);
  }
  stack_.Push(token);
  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::EvalShl(const Bytecode& bytecode) {
  return EvalBinop([](const InterpValue& lhs, const InterpValue& rhs) {
    return lhs.Shl(rhs);
  });
}

absl::Status BytecodeInterpreter::EvalShr(const Bytecode& bytecode) {
  return EvalBinop([](const InterpValue& lhs, const InterpValue& rhs) {
    if (lhs.IsSigned()) {
      return lhs.Shra(rhs);
    }
    return lhs.Shrl(rhs);
  });
}

absl::Status BytecodeInterpreter::EvalSlice(const Bytecode& bytecode) {
  XLS_ASSIGN_OR_RETURN(InterpValue length, Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue start, Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue basis, Pop());
  XLS_RET_CHECK(length.IsUBits())
      << "Slice length is not unsigned bits: " << length.ToString();
  XLS_RET_CHECK(start.IsUBits())
      << "Slice start is not unsigned bits: " << start.ToString();
  XLS_RET_CHECK(basis.IsUBits())
      << "Slice basis is not unsigned bits: " << basis.ToString();
  XLS_ASSIGN_OR_RETURN(InterpValue result, basis.Slice(start, length));
  stack_.Push(result);
  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::EvalStore(const Bytecode& bytecode) {
  XLS_ASSIGN_OR_RETURN(Bytecode::SlotIndex slot, bytecode.slot_index());
  if (stack_.empty()) {
    return absl::InvalidArgumentError(
        "Attempted to store value from empty stack.");
  }

  XLS_ASSIGN_OR_RETURN(InterpValue value, Pop());
  frames_.back().StoreSlot(slot, value);
  return absl::OkStatus();
}

absl::StatusOr<std::optional<int64_t>> BytecodeInterpreter::EvalJumpRelIf(
    int64_t pc, const Bytecode& bytecode) {
  XLS_ASSIGN_OR_RETURN(InterpValue top, Pop());
  VLOG(2) << "jump_rel_if value: " << top.ToString();
  if (top.IsTrue()) {
    XLS_ASSIGN_OR_RETURN(Bytecode::JumpTarget target, bytecode.jump_target());
    return pc + target.value();
  }
  return std::nullopt;
}

absl::Status BytecodeInterpreter::EvalSub(const Bytecode& bytecode,
                                          bool is_signed) {
  return EvalBinop([&](const InterpValue& lhs,
                       const InterpValue& rhs) -> absl::StatusOr<InterpValue> {
    XLS_ASSIGN_OR_RETURN(InterpValue output, lhs.Sub(rhs));

    // Slow path: when rollover warning hook is enabled.
    if (options_.rollover_hook() != nullptr) {
      auto make_big_int = [is_signed](const Bits& bits) {
        return is_signed ? BigInt::MakeSigned(bits)
                         : BigInt::MakeUnsigned(bits);
      };
      bool rollover =
          make_big_int(lhs.GetBitsOrDie()) - make_big_int(rhs.GetBitsOrDie()) !=
          make_big_int(output.GetBitsOrDie());
      if (rollover) {
        options_.rollover_hook()(bytecode.source_span());
      }
    }

    return output;
  });
}

absl::Status BytecodeInterpreter::EvalSwap(const Bytecode& bytecode) {
  XLS_RET_CHECK_GE(stack_.size(), 2);
  XLS_ASSIGN_OR_RETURN(InterpValue tos0, Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue tos1, Pop());
  stack_.Push(tos0);
  stack_.Push(tos1);
  return absl::OkStatus();
}

/* static */ absl::StatusOr<std::string> BytecodeInterpreter::TraceDataToString(
    const Bytecode::TraceData& trace_data, InterpreterStack& stack) {
  XLS_RET_CHECK(!trace_data.steps().empty());

  size_t argc =
      std::count_if(trace_data.steps().begin(), trace_data.steps().end(),
                    [](const FormatStep& step) {
                      // The strings are literals.
                      return !std::holds_alternative<std::string>(step);
                    });

  std::vector<InterpValue> args;
  for (size_t i = 0; i < argc; ++i) {
    XLS_RET_CHECK(!stack.empty());
    XLS_ASSIGN_OR_RETURN(InterpValue value, stack.Pop());
    args.push_back(value);
  }

  std::reverse(args.begin(), args.end());

  // When we encounter an argument to format, we may want to resolve its struct
  // definition. We do this by tracking the argument number.
  size_t argno = 0;

  std::vector<std::string> pieces;
  for (int64_t i = 0; i < trace_data.steps().size(); ++i) {
    FormatStep trace_element = trace_data.steps().at(i);
    if (std::holds_alternative<std::string>(trace_element)) {
      pieces.push_back(std::get<std::string>(trace_element));
    } else {
      const InterpValue& value = args.at(argno);
      if (argno < trace_data.value_fmt_descs().size()) {
        XLS_ASSIGN_OR_RETURN(
            std::string formatted,
            value.ToFormattedString(trace_data.value_fmt_descs()[argno]));
        pieces.push_back(formatted);
      } else {
        pieces.push_back(value.ToString(
            /*humanize=*/true, std::get<FormatPreference>(trace_element)));
      }
      argno += 1;
      CHECK_LE(argno, argc);
    }
  }

  return absl::StrJoin(pieces, "");
}

absl::Status BytecodeInterpreter::EvalTrace(const Bytecode& bytecode) {
  XLS_ASSIGN_OR_RETURN(const Bytecode::TraceData* trace_data,
                       bytecode.trace_data());
  XLS_ASSIGN_OR_RETURN(std::string message,
                       TraceDataToString(*trace_data, stack_));
  if (options_.trace_hook()) {
    options_.trace_hook()(bytecode.source_span(), message);
  }
  stack_.Push(InterpValue::MakeToken());
  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::EvalWidthSlice(const Bytecode& bytecode) {
  XLS_ASSIGN_OR_RETURN(const Type* type, bytecode.type_data());
  XLS_ASSIGN_OR_RETURN(const Type* unwrapped_type, UnwrapMetaType(*type));

  // Width slice only works on bits-like types.
  std::optional<BitsLikeProperties> bits_like = GetBitsLike(*unwrapped_type);
  XLS_RET_CHECK(bits_like.has_value())
      << "Wide slice type is not bits-like: " << type->ToString();
  XLS_ASSIGN_OR_RETURN(bool is_signed, bits_like->is_signed.GetAsBool());
  XLS_ASSIGN_OR_RETURN(int64_t width_value, bits_like->size.GetAsInt64());

  InterpValue oob_value(InterpValue::MakeUBits(width_value, /*value=*/0));
  XLS_ASSIGN_OR_RETURN(InterpValue start, Pop());
  if (!start.FitsInUint64()) {
    XLS_RETURN_IF_ERROR(Pop().status());
    stack_.Push(oob_value);
    return absl::OkStatus();
  }
  XLS_ASSIGN_OR_RETURN(uint64_t start_index, start.GetBitValueUnsigned());

  XLS_ASSIGN_OR_RETURN(InterpValue basis, Pop());
  XLS_ASSIGN_OR_RETURN(Bits basis_bits, basis.GetBits());
  XLS_ASSIGN_OR_RETURN(int64_t basis_width, basis.GetBitCount());
  InterpValue width = InterpValue::MakeUBits(64, width_value);

  if (start_index >= basis_width) {
    stack_.Push(oob_value);
    return absl::OkStatus();
  }

  if (start_index + width_value > basis_width) {
    basis_bits = bits_ops::ZeroExtend(basis_bits, start_index + width_value);
  }

  Bits result_bits = basis_bits.Slice(start_index, width_value);
  InterpValueTag tag =
      is_signed ? InterpValueTag::kSBits : InterpValueTag::kUBits;
  XLS_ASSIGN_OR_RETURN(InterpValue result,
                       InterpValue::MakeBits(tag, result_bits));
  stack_.Push(result);
  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::EvalXor(const Bytecode& bytecode) {
  return EvalBinop([](const InterpValue& lhs, const InterpValue& rhs) {
    return lhs.BitwiseXor(rhs);
  });
}

absl::Status BytecodeInterpreter::RunBuiltinFn(const Bytecode& bytecode,
                                               Builtin builtin) {
  switch (builtin) {
    case Builtin::kAndReduce:
      return RunBuiltinAndReduce(bytecode, stack_);
    case Builtin::kAssertEq:
      return RunBuiltinAssertEq(bytecode, stack_, frames_.back(), options_);
    case Builtin::kAssertLt:
      return RunBuiltinAssertLt(bytecode, stack_, frames_.back(), options_);
    case Builtin::kBitSliceUpdate:
      return RunBuiltinBitSliceUpdate(bytecode, stack_);
    case Builtin::kClz:
      return RunBuiltinClz(bytecode, stack_);
    case Builtin::kCover:
      return RunBuiltinCover(bytecode, stack_);
    case Builtin::kCtz:
      return RunBuiltinCtz(bytecode, stack_);
    case Builtin::kEnumerate:
      return RunBuiltinEnumerate(bytecode, stack_);
    case Builtin::kFail: {
      XLS_ASSIGN_OR_RETURN(InterpValue value, Pop());
      return FailureErrorStatus(bytecode.source_span(), value.ToString(),
                                file_table());
    }
    case Builtin::kAssert: {
      XLS_ASSIGN_OR_RETURN(InterpValue label, Pop());
      XLS_ASSIGN_OR_RETURN(InterpValue predicate, Pop());

      XLS_ASSIGN_OR_RETURN(std::string label_as_string,
                           InterpValueAsString(label));
      if (predicate.IsFalse()) {
        return FailureErrorStatus(bytecode.source_span(), label_as_string,
                                  file_table());
      }

      stack_.Push(InterpValue::MakeUnit());
      return absl::OkStatus();
    }
    case Builtin::kGate:
      return RunBuiltinGate(bytecode, stack_);
    case Builtin::kMap:
      return RunBuiltinMap(bytecode);
    case Builtin::kZip:
      return RunBuiltinZip(bytecode, stack_);
    case Builtin::kEncode:
      return RunBuiltinEncode(bytecode, stack_);
    case Builtin::kOneHot:
      return RunBuiltinOneHot(bytecode, stack_);
    case Builtin::kOneHotSel:
      return RunBuiltinOneHotSel(bytecode, stack_);
    case Builtin::kPriorityhSel:
      return RunBuiltinPrioritySel(bytecode, stack_);
    case Builtin::kOrReduce:
      return RunBuiltinOrReduce(bytecode, stack_);
    case Builtin::kRange:
      return RunBuiltinRange(bytecode, stack_);
    case Builtin::kRev:
      return RunBuiltinRev(bytecode, stack_);
    case Builtin::kArrayRev:
      return RunBuiltinArrayRev(bytecode, stack_);
    case Builtin::kArraySize:
      return RunBuiltinArraySize(bytecode, stack_);
    case Builtin::kSignex:
      return RunBuiltinSignex(bytecode, stack_);
    case Builtin::kArraySlice:
      return RunBuiltinArraySlice(bytecode, stack_);
    case Builtin::kSMulp:
      return RunBuiltinSMulp(bytecode, stack_);
    case Builtin::kTrace:
      return absl::InternalError(
          "`trace!` builtins should be converted into kTrace opcodes.");
    case Builtin::kUMulp:
      return RunBuiltinUMulp(bytecode, stack_);
    case Builtin::kUpdate:
      return RunBuiltinUpdate(bytecode, stack_);
    case Builtin::kXorReduce:
      return RunBuiltinXorReduce(bytecode, stack_);
    // Implementation note: some of these operations are implemented via
    // bytecodes; e.g. see `BytecodeEmitter::HandleBuiltin*`
    case Builtin::kJoin:
    case Builtin::kToken:
    case Builtin::kSend:
    case Builtin::kSendIf:
    case Builtin::kRecv:
    case Builtin::kRecvIf:
    case Builtin::kRecvNonBlocking:
    case Builtin::kRecvIfNonBlocking:
    case Builtin::kDecode:
    case Builtin::kCheckedCast:
    case Builtin::kWideningCast:
      return absl::UnimplementedError(absl::StrFormat(
          "BytecodeInterpreter: builtin function \"%s\" not yet implemented.",
          BuiltinToString(builtin)));
  }
}

absl::Status BytecodeInterpreter::RunBuiltinMap(const Bytecode& bytecode) {
  XLS_RET_CHECK_GE(stack_.size(), 2);
  XLS_ASSIGN_OR_RETURN(Bytecode::InvocationData invocation_data,
                       bytecode.invocation_data());
  XLS_ASSIGN_OR_RETURN(InterpValue callee, Pop());
  XLS_RET_CHECK(callee.IsFunction());
  XLS_ASSIGN_OR_RETURN(InterpValue inputs, Pop());

  XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* elements,
                       inputs.GetValues());
  Span span = invocation_data.invocation()->span();

  // Rather than "unrolling" the map, we can implement a for-like loop here to
  // prevent the generated BytecodeFunction from being overlarge as the input
  // array scales in size.
  // Here, slot 0 is the function arg, i.e., the initial input array, and we
  // use slot 1 to hold the current loop and element index.
  // Initialize the loop index.
  std::vector<Bytecode> bytecodes;
  bytecodes.push_back(
      Bytecode(span, Bytecode::Op::kLiteral, InterpValue::MakeU32(0)));
  bytecodes.push_back(
      Bytecode(span, Bytecode::Op::kStore, Bytecode::SlotIndex(1)));

  // Top-of-loop marker.
  size_t top_of_loop_index = bytecodes.size();
  bytecodes.push_back(Bytecode(span, Bytecode::Op::kJumpDest));

  // Extract element N and call the mapping fn on that value.
  bytecodes.push_back(
      Bytecode(span, Bytecode::Op::kLoad, Bytecode::SlotIndex(0)));
  bytecodes.push_back(
      Bytecode(span, Bytecode::Op::kLoad, Bytecode::SlotIndex(1)));
  bytecodes.push_back(Bytecode(span, Bytecode::Op::kIndex));
  bytecodes.push_back(Bytecode(span, Bytecode::Op::kLiteral, callee));
  bytecodes.push_back(Bytecode(span, Bytecode::Op::kCall, invocation_data));

  // Increment the index.
  bytecodes.push_back(
      Bytecode(span, Bytecode::Op::kLoad, Bytecode::SlotIndex(1)));
  bytecodes.push_back(
      Bytecode(span, Bytecode::Op::kLiteral, InterpValue::MakeU32(1)));
  bytecodes.push_back(Bytecode(span, Bytecode::Op::kUAdd));
  bytecodes.push_back(
      Bytecode(span, Bytecode::Op::kStore, Bytecode::SlotIndex(1)));

  // Is index < input_size?
  bytecodes.push_back(
      Bytecode(span, Bytecode::Op::kLoad, Bytecode::SlotIndex(1)));
  bytecodes.push_back(
      Bytecode(span, Bytecode::Op::kLiteral,
               InterpValue::MakeU32(static_cast<uint32_t>(elements->size()))));
  bytecodes.push_back(Bytecode(span, Bytecode::Op::kLt));

  // If true, jump to top-of-loop, else create the result array.
  bytecodes.push_back(
      Bytecode(span, Bytecode::Op::kJumpRelIf,
               Bytecode::JumpTarget(top_of_loop_index - bytecodes.size())));
  bytecodes.push_back(Bytecode(span, Bytecode::Op::kCreateArray,
                               Bytecode::NumElements(elements->size())));

  // Now take the collected bytecodes and cram them into a BytecodeFunction,
  // then start executing it.
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<BytecodeFunction> bf,
                       BytecodeFunction::Create(
                           /*owner=*/nullptr, /*source_fn=*/nullptr,
                           frames_.back().type_info(), std::move(bytecodes)));
  BytecodeFunction* bf_ptr = bf.get();
  frames_.push_back(Frame(bf_ptr, {inputs}, bf_ptr->type_info(),
                          invocation_data.caller_bindings(),
                          /*initial_args=*/{}, std::move(bf)));
  return absl::OkStatus();
}

std::string BytecodeInterpreter::FormatChannelNameForTracing(
    const Bytecode::ChannelData& channel) {
  if (proc_id().has_value()) {
    // Include a string describing the instantiation path to this channel
    // instance. For example:
    //   my_top->my_proc#1->your_proc#2::the_channel
    std::string result;
    for (auto [proc, instance] : proc_id()->proc_instance_stack) {
      if (result.empty()) {
        result += proc->identifier();
      } else {
        absl::StrAppendFormat(&result, "->%s#%d", proc->identifier(), instance);
      }
    }
    return absl::StrCat(result, "::", channel.channel_name());
  }
  return std::string(channel.channel_name());
}

}  // namespace xls::dslx
