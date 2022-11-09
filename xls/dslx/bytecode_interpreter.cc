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

#include "xls/dslx/bytecode_interpreter.h"

#include <variant>

#include "absl/status/status.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/ast.h"
#include "xls/dslx/bytecode_emitter.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/interp_value_helpers.h"
#include "xls/ir/bits_ops.h"

namespace xls::dslx {

Frame::Frame(BytecodeFunction* bf, std::vector<InterpValue> args,
             const TypeInfo* type_info,
             const std::optional<SymbolicBindings>& bindings,
             std::vector<InterpValue> initial_args,
             std::unique_ptr<BytecodeFunction> bf_holder)
    : pc_(0),
      slots_(std::move(args)),
      bf_(bf),
      type_info_(type_info),
      bindings_(bindings),
      initial_args_(initial_args),
      bf_holder_(std::move(bf_holder)) {}

void Frame::StoreSlot(Bytecode::SlotIndex slot, InterpValue value) {
  // Slots are usually encountered in order of use (and assignment), except for
  // those declared inside conditional branches, which may never be seen,
  // so we may have to add more than one slot at a time in such cases.
  while (slots_.size() <= slot.value()) {
    slots_.push_back(InterpValue::MakeToken());
  }

  slots_.at(slot.value()) = value;
}

/* static */ absl::StatusOr<InterpValue> BytecodeInterpreter::Interpret(
    ImportData* import_data, BytecodeFunction* bf,
    const std::vector<InterpValue>& args, PostFnEvalHook post_fn_eval_hook) {
  XLS_ASSIGN_OR_RETURN(auto interpreter, BytecodeInterpreter::CreateUnique(
                                             import_data, bf, args));
  XLS_RETURN_IF_ERROR(interpreter->Run(post_fn_eval_hook));
  return interpreter->stack_.back();
}

BytecodeInterpreter::BytecodeInterpreter(ImportData* import_data,
                                         BytecodeFunction* bf)
    : import_data_(import_data) {}

absl::Status BytecodeInterpreter::InitFrame(
    BytecodeFunction* bf, const std::vector<InterpValue>& args,
    const TypeInfo* type_info) {
  XLS_RET_CHECK(frames_.empty());

  // In "mission mode" we expect type_info to be non-null in the frame, but for
  // bytecode-level testing we may not have an AST.
  if (type_info == nullptr && bf->owner() != nullptr) {
    type_info = import_data_->GetRootTypeInfo(bf->owner()).value();
  }
  frames_.push_back(
      Frame(bf, args, type_info, absl::nullopt, /*initial_args=*/{}));
  return absl::OkStatus();
}

/* static */ absl::StatusOr<std::unique_ptr<BytecodeInterpreter>>
BytecodeInterpreter::CreateUnique(ImportData* import_data, BytecodeFunction* bf,
                                  const std::vector<InterpValue>& args) {
  auto interp = absl::WrapUnique(new BytecodeInterpreter(import_data, bf));
  XLS_RETURN_IF_ERROR(interp->InitFrame(bf, args, bf->type_info()));
  return interp;
}

absl::Status BytecodeInterpreter::Run(PostFnEvalHook post_fn_eval_hook) {
  while (!frames_.empty()) {
    Frame* frame = &frames_.back();
    while (frame->pc() < frame->bf()->bytecodes().size()) {
      const std::vector<Bytecode>& bytecodes = frame->bf()->bytecodes();
      const Bytecode& bytecode = bytecodes.at(frame->pc());
      XLS_VLOG(2) << std::hex << "PC: " << frame->pc() << " : "
                  << bytecode.ToString();
      XLS_VLOG(3) << " - TOS pre: "
                  << (stack_.empty() ? "-empty-" : stack_.back().ToString());
      int64_t old_pc = frame->pc();
      XLS_RETURN_IF_ERROR(EvalNextInstruction());
      XLS_VLOG(3) << " - TOS post: "
                  << (stack_.empty() ? "-empty-" : stack_.back().ToString());

      if (bytecode.op() == Bytecode::Op::kCall) {
        frame = &frames_.back();
      } else if (frame->pc() != old_pc + 1) {
        XLS_RET_CHECK(bytecodes.at(frame->pc()).op() == Bytecode::Op::kJumpDest)
            << "Jumping from PC " << old_pc << " to PC: " << frame->pc()
            << " bytecode: " << bytecodes.at(frame->pc()).ToString()
            << " not a jump_dest or old bytecode: " << bytecode.ToString()
            << " was not a call op.";
      }
    }

    // Run the post-fn eval hook, if our function has a return value and if we
    // actually have a hook.
    const Function* source_fn = frame->bf()->source_fn();
    if (source_fn != nullptr) {
      std::optional<ConcreteType*> fn_return =
          frame->type_info()->GetItem(source_fn);
      if (fn_return.has_value()) {
        bool fn_returns_value = *fn_return.value() != *ConcreteType::MakeUnit();
        if (post_fn_eval_hook != nullptr && fn_returns_value) {
          SymbolicBindings holder;
          const SymbolicBindings* bindings = &holder;
          if (frame->bindings().has_value()) {
            bindings = &frame->bindings().value();
          }
          XLS_RETURN_IF_ERROR(post_fn_eval_hook(
              source_fn, frame->initial_args(), bindings, stack_.back()));
        }
      }
    }

    // We've reached the end of a function. Time to load the next frame up!
    frames_.pop_back();
  }

  return absl::OkStatus();
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
  XLS_VLOG(10) << "Running bytecode: " << bytecode.ToString()
               << " depth before: " << stack_.size();
  switch (bytecode.op()) {
    case Bytecode::Op::kAdd: {
      XLS_RETURN_IF_ERROR(EvalAdd(bytecode));
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
    case Bytecode::Op::kMul: {
      XLS_RETURN_IF_ERROR(EvalMul(bytecode));
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
      break;
    }
    case Bytecode::Op::kStore: {
      XLS_RETURN_IF_ERROR(EvalStore(bytecode));
      break;
    }
    case Bytecode::Op::kSub: {
      XLS_RETURN_IF_ERROR(EvalSub(bytecode));
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

/* static */ absl::StatusOr<InterpValue> BytecodeInterpreter::Pop(
    std::vector<InterpValue>& stack) {
  if (stack.empty()) {
    return absl::InternalError("Tried to pop off an empty stack.");
  }
  InterpValue value = std::move(stack.back());
  stack.pop_back();
  return value;
}

absl::Status BytecodeInterpreter::EvalUnop(
    const std::function<absl::StatusOr<InterpValue>(const InterpValue& arg)>&
        op) {
  XLS_RET_CHECK_GE(stack_.size(), 1);
  XLS_ASSIGN_OR_RETURN(InterpValue arg, Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue result, op(arg));
  stack_.push_back(std::move(result));
  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::EvalBinop(
    const std::function<absl::StatusOr<InterpValue>(
        const InterpValue& lhs, const InterpValue& rhs)>& op) {
  XLS_RET_CHECK_GE(stack_.size(), 2);
  XLS_ASSIGN_OR_RETURN(InterpValue rhs, Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue lhs, Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue result, op(lhs, rhs));
  stack_.push_back(std::move(result));
  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::EvalAdd(const Bytecode& bytecode) {
  return EvalBinop([](const InterpValue& lhs, const InterpValue& rhs) {
    return lhs.Add(rhs);
  });
}

absl::Status BytecodeInterpreter::EvalAnd(const Bytecode& bytecode) {
  return EvalBinop([](const InterpValue& lhs, const InterpValue& rhs) {
    return lhs.BitwiseAnd(rhs);
  });
}

absl::StatusOr<BytecodeFunction*> BytecodeInterpreter::GetBytecodeFn(
    Function* f, const Invocation* invocation,
    const std::optional<SymbolicBindings>& caller_bindings) {
  const Frame& frame = frames_.back();
  const TypeInfo* type_info = frame.type_info();

  BytecodeCacheInterface* cache = import_data_->bytecode_cache();
  if (cache == nullptr) {
    return absl::InvalidArgumentError("Bytecode cache is NULL.");
  }

  if (f->IsParametric() || f->tag() == Function::Tag::kProcInit) {
    XLS_RET_CHECK(caller_bindings.has_value());
    std::optional<TypeInfo*> maybe_type_info =
        type_info->GetInvocationTypeInfo(invocation,
                                            caller_bindings.value());
    if (!maybe_type_info.has_value()) {
      return absl::InternalError(absl::StrCat(
          "Could not find type info for invocation ", invocation->ToString(),
          " : ", invocation->span().ToString()));
    }
    type_info = maybe_type_info.value();
  } else if (f->owner() != type_info->module()) {
    // If the new function is in a different module and it's NOT parametric,
    // then we need the root TypeInfo for the new module.
    XLS_ASSIGN_OR_RETURN(type_info, import_data_->GetRootTypeInfo(f->owner()));
  }

  return cache->GetOrCreateBytecodeFunction(f, type_info, caller_bindings);
}

absl::Status BytecodeInterpreter::EvalCall(const Bytecode& bytecode) {
  XLS_VLOG(3) << "BytecodeInterpreter::EvalCall: " << bytecode.ToString();
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

  XLS_ASSIGN_OR_RETURN(
      BytecodeFunction * bf,
      GetBytecodeFn(user_fn_data.function, data.invocation, data.bindings));

  // Store the _return_ PC.
  frames_.back().IncrementPc();

  int64_t num_args = user_fn_data.function->params().size();
  std::vector<InterpValue> args(num_args, InterpValue::MakeToken());
  for (int i = 0; i < num_args; i++) {
    XLS_ASSIGN_OR_RETURN(InterpValue arg, Pop());
    args[num_args - i - 1] = arg;
  }

  std::vector<InterpValue> args_copy = args;
  frames_.push_back(Frame(bf, std::move(args), bf->type_info(), data.bindings,
                          std::move(args_copy)));

  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::EvalCast(const Bytecode& bytecode) {
  if (!bytecode.data().has_value() ||
      !std::holds_alternative<std::unique_ptr<ConcreteType>>(
          bytecode.data().value())) {
    return absl::InternalError("Cast op requires ConcreteType data.");
  }

  XLS_ASSIGN_OR_RETURN(InterpValue from, Pop());

  if (!bytecode.data().has_value()) {
    return absl::InternalError("Cast op is missing its data element!");
  }

  ConcreteType* to =
      std::get<std::unique_ptr<ConcreteType>>(bytecode.data().value()).get();
  if (from.IsArray()) {
    // From array to bits.
    BitsType* to_bits = dynamic_cast<BitsType*>(to);
    if (to_bits == nullptr) {
      return absl::InvalidArgumentError(
          "Array types can only be cast to bits.");
    }
    XLS_ASSIGN_OR_RETURN(InterpValue converted, from.Flatten());
    stack_.push_back(converted);
    return absl::OkStatus();
  }

  if (from.IsEnum()) {
    // From enum to bits.
    BitsType* to_bits = dynamic_cast<BitsType*>(to);
    if (to_bits == nullptr) {
      return absl::InvalidArgumentError("Enum types can only be cast to bits.");
    }

    stack_.push_back(
        InterpValue::MakeBits(from.IsSigned(), from.GetBitsOrDie()));
    return absl::OkStatus();
  }

  if (!from.IsBits()) {
    return absl::InvalidArgumentError(
        "Only casts from arrays, enums, and bits are supported.");
  }

  int64_t from_bit_count = from.GetBits().value().bit_count();
  // From bits to array.
  if (ArrayType* to_array = dynamic_cast<ArrayType*>(to); to_array != nullptr) {
    XLS_ASSIGN_OR_RETURN(int64_t to_bit_count,
                         to_array->GetTotalBitCount().value().GetAsInt64());
    if (from_bit_count != to_bit_count) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Cast to array had mismatching bit counts: from %d to %d.",
          from_bit_count, to_bit_count));
    }
    XLS_ASSIGN_OR_RETURN(InterpValue casted, CastBitsToArray(from, *to_array));
    stack_.push_back(casted);
    return absl::OkStatus();
  }

  // From bits to enum.
  if (EnumType* to_enum = dynamic_cast<EnumType*>(to); to_enum != nullptr) {
    XLS_ASSIGN_OR_RETURN(InterpValue converted, CastBitsToEnum(from, *to_enum));
    stack_.push_back(converted);
    return absl::OkStatus();
  }

  BitsType* to_bits = dynamic_cast<BitsType*>(to);
  if (to_bits == nullptr) {
    return absl::InvalidArgumentError(
        "Bits can only be cast to arrays, enums, or other bits types.");
  }

  XLS_ASSIGN_OR_RETURN(int64_t to_bit_count,
                       to_bits->GetTotalBitCount().value().GetAsInt64());

  Bits result_bits;
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
  InterpValue result = InterpValue::MakeBits(to_bits->is_signed(), result_bits);

  stack_.push_back(result);

  return absl::OkStatus();
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
  stack_.push_back(array);
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

  stack_.push_back(InterpValue::MakeTuple(elements));
  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::EvalDiv(const Bytecode& bytecode) {
  return EvalBinop([](const InterpValue& lhs, const InterpValue& rhs) {
    return lhs.FloorDiv(rhs);
  });
}

absl::Status BytecodeInterpreter::EvalDup(const Bytecode& bytecode) {
  XLS_RET_CHECK(!stack_.empty());
  stack_.push_back(stack_.back());
  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::EvalEq(const Bytecode& bytecode) {
  return EvalBinop([](const InterpValue& lhs, const InterpValue& rhs) {
    return InterpValue::MakeBool(lhs.Eq(rhs));
  });
}

absl::Status BytecodeInterpreter::EvalExpandTuple(const Bytecode& bytecode) {
  InterpValue tuple = stack_.back();
  stack_.pop_back();
  if (!tuple.IsTuple()) {
    return FailureErrorStatus(
        bytecode.source_span(),
        absl::StrCat("Stack top for ExpandTuple was not a tuple, was: ",
                     TagToString(tuple.tag())));
  }

  // Note that we destructure the tuple in "reverse" order, with the first
  // element on top of the stack.
  XLS_ASSIGN_OR_RETURN(int64_t tuple_size, tuple.GetLength());
  for (int64_t i = tuple_size - 1; i >= 0; i--) {
    XLS_ASSIGN_OR_RETURN(InterpValue element,
                         tuple.Index(InterpValue::MakeUBits(64, i)));
    stack_.push_back(element);
  }

  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::EvalFail(const Bytecode& bytecode) {
  XLS_ASSIGN_OR_RETURN(const Bytecode::TraceData* trace_data,
                       bytecode.trace_data());
  XLS_ASSIGN_OR_RETURN(std::string message,
                       TraceDataToString(*trace_data, stack_));
  return FailureErrorStatus(bytecode.source_span(), message);
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

absl::Status BytecodeInterpreter::EvalIndex(const Bytecode& bytecode) {
  XLS_ASSIGN_OR_RETURN(InterpValue index, Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue basis, Pop());

  if (!basis.IsArray() && !basis.IsTuple()) {
    return absl::InvalidArgumentError(
        "Can only index on array or tuple values.");
  }

  XLS_ASSIGN_OR_RETURN(InterpValue result, basis.Index(index),
                       _ << " while processing " << bytecode.ToString());
  stack_.push_back(result);
  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::EvalInvert(const Bytecode& bytecode) {
  XLS_ASSIGN_OR_RETURN(InterpValue operand, Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue result, operand.BitwiseNegate());
  stack_.push_back(result);
  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::EvalLe(const Bytecode& bytecode) {
  return EvalBinop([](const InterpValue& lhs, const InterpValue& rhs) {
    return lhs.Le(rhs);
  });
}

absl::Status BytecodeInterpreter::EvalLiteral(const Bytecode& bytecode) {
  XLS_ASSIGN_OR_RETURN(InterpValue value, bytecode.value_data());
  stack_.push_back(value);
  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::EvalLoad(const Bytecode& bytecode) {
  XLS_ASSIGN_OR_RETURN(Bytecode::SlotIndex slot, bytecode.slot_index());
  if (frames_.back().slots().size() <= slot.value()) {
    return absl::InternalError(absl::StrFormat(
        "Attempted to access local data in slot %d, which is out of range.",
        slot.value()));
  }
  stack_.push_back(frames_.back().slots().at(slot.value()));
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
  stack_.push_back(result);
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
  stack_.push_back(result);
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
  if (item.kind() == Kind::kInterpValue) {
    XLS_ASSIGN_OR_RETURN(InterpValue arm_value, item.interp_value());
    return arm_value.Eq(value);
  }

  if (item.kind() == Kind::kLoad) {
    XLS_ASSIGN_OR_RETURN(Bytecode::SlotIndex slot_index, item.slot_index());
    if (frame->slots().size() <= slot_index.value()) {
      return absl::InternalError(
          absl::StrCat("MatchArm load item index was OOB: ", slot_index.value(),
                       " vs. ", frame->slots().size(), "."));
    }
    InterpValue arm_value = frame->slots().at(slot_index.value());
    return arm_value.Eq(value);
  }

  if (item.kind() == Kind::kStore) {
    XLS_ASSIGN_OR_RETURN(Bytecode::SlotIndex slot_index, item.slot_index());
    frame->StoreSlot(slot_index, value);
    return true;
  }

  if (item.kind() == Kind::kWildcard) {
    return true;
  }

  // Otherwise, we're a tuple. Recurse.
  XLS_ASSIGN_OR_RETURN(auto item_elements, item.tuple_elements());
  XLS_ASSIGN_OR_RETURN(auto* value_elements, value.GetValues());
  if (item_elements.size() != value_elements->size()) {
    return absl::InternalError(
        absl::StrCat("Match arm item had a different number of elements "
                     "than the corresponding InterpValue: ",
                     item.ToString(), " vs. ", value.ToString()));
  }

  for (int i = 0; i < item_elements.size(); i++) {
    XLS_ASSIGN_OR_RETURN(
        bool equal, MatchArmEqualsInterpValue(&frames_.back(), item_elements[i],
                                              value_elements->at(i)));
    if (!equal) {
      return false;
    }
  }

  return true;
}

absl::Status BytecodeInterpreter::EvalMatchArm(const Bytecode& bytecode) {
  // Puts true on the stack if the items are equal and false otherwise.
  XLS_ASSIGN_OR_RETURN(const Bytecode::MatchArmItem* item,
                       bytecode.match_arm_item());
  XLS_ASSIGN_OR_RETURN(InterpValue matchee, Pop());
  XLS_ASSIGN_OR_RETURN(
      bool equal, MatchArmEqualsInterpValue(&frames_.back(), *item, matchee));
  stack_.push_back(InterpValue::MakeBool(equal));
  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::EvalMul(const Bytecode& bytecode) {
  return EvalBinop([](const InterpValue& lhs, const InterpValue& rhs) {
    return lhs.Mul(rhs);
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
  stack_.push_back(result);
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
  return RangeInternal();
}

absl::Status BytecodeInterpreter::EvalRecvNonBlocking(
    const Bytecode& bytecode) {
  // TODO(rspringer): 2022-03-10 Thread safety!
  XLS_ASSIGN_OR_RETURN(InterpValue condition, Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue channel_value, Pop());
  XLS_ASSIGN_OR_RETURN(auto channel, channel_value.GetChannel());
  XLS_ASSIGN_OR_RETURN(InterpValue token, Pop());

  if (condition.IsTrue()) {
    if (channel->empty()) {
      XLS_RET_CHECK(bytecode.has_data());
      const Bytecode::Data& data = bytecode.data().value();

      XLS_RET_CHECK(
          std::holds_alternative<std::unique_ptr<ConcreteType>>(data));
      const std::unique_ptr<ConcreteType>& payload_type =
          std::get<std::unique_ptr<ConcreteType>>(data);

      XLS_ASSIGN_OR_RETURN(InterpValue zero,
                           CreateZeroValueFromType(*payload_type));
      stack_.push_back(
          InterpValue::MakeTuple({token, zero, InterpValue::MakeBool(false)}));
    } else {
      stack_.push_back(InterpValue::MakeTuple(
          {token, channel->front(), InterpValue::MakeBool(true)}));
      channel->pop_front();
    }
  } else {
    XLS_RET_CHECK(bytecode.has_data());
    const Bytecode::Data& data = bytecode.data().value();

    XLS_RET_CHECK(std::holds_alternative<std::unique_ptr<ConcreteType>>(data));
    const std::unique_ptr<ConcreteType>& payload_type =
        std::get<std::unique_ptr<ConcreteType>>(data);

    XLS_ASSIGN_OR_RETURN(InterpValue zero,
                         CreateZeroValueFromType(*payload_type));

    XLS_ASSIGN_OR_RETURN(InterpValue token, Pop());
    stack_.push_back(
        InterpValue::MakeTuple({token, zero, InterpValue::MakeBool(false)}));
  }

  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::EvalRecv(const Bytecode& bytecode) {
  // TODO(rspringer): 2022-03-10 Thread safety!
  XLS_ASSIGN_OR_RETURN(InterpValue condition, Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue channel_value, Pop());
  XLS_ASSIGN_OR_RETURN(auto channel, channel_value.GetChannel());

  if (condition.IsTrue()) {
    if (channel->empty()) {
      // Restore the stack!
      stack_.push_back(channel_value);
      stack_.push_back(condition);
      return absl::UnavailableError("Channel is empty.");
    }

    XLS_ASSIGN_OR_RETURN(InterpValue token, Pop());

    stack_.push_back(InterpValue::MakeTuple({token, channel->front()}));
    channel->pop_front();
  } else {
    XLS_RET_CHECK(bytecode.has_data());
    const Bytecode::Data& data = bytecode.data().value();

    XLS_RET_CHECK(std::holds_alternative<std::unique_ptr<ConcreteType>>(data));
    const std::unique_ptr<ConcreteType>& payload_type =
        std::get<std::unique_ptr<ConcreteType>>(data);

    XLS_ASSIGN_OR_RETURN(InterpValue zero,
                         CreateZeroValueFromType(*payload_type));

    XLS_ASSIGN_OR_RETURN(InterpValue token, Pop());
    stack_.push_back(InterpValue::MakeTuple({token, zero}));
  }

  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::EvalSend(const Bytecode& bytecode) {
  XLS_ASSIGN_OR_RETURN(InterpValue condition, Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue payload, Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue channel_value, Pop());
  XLS_ASSIGN_OR_RETURN(auto channel, channel_value.GetChannel());
  XLS_ASSIGN_OR_RETURN(InterpValue token, Pop());
  if (condition.IsTrue()) {
    channel->push_back(payload);
  }
  stack_.push_back(token);
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
  XLS_ASSIGN_OR_RETURN(InterpValue limit, Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue start, Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue basis, Pop());
  XLS_ASSIGN_OR_RETURN(int64_t basis_bit_count, basis.GetBitCount());
  XLS_ASSIGN_OR_RETURN(int64_t start_bit_count, start.GetBitCount());

  InterpValue zero = InterpValue::MakeSBits(start_bit_count, 0);
  InterpValue basis_length =
      InterpValue::MakeSBits(start_bit_count, basis_bit_count);

  XLS_ASSIGN_OR_RETURN(InterpValue start_lt_zero, start.Lt(zero));
  if (start_lt_zero.IsTrue()) {
    // Remember, start is negative if we're here.
    XLS_ASSIGN_OR_RETURN(start, basis_length.Add(start));
    // If start is _still_ less than zero, then we clamp to zero.
    XLS_ASSIGN_OR_RETURN(start_lt_zero, start.Lt(zero));
    if (start_lt_zero.IsTrue()) {
      start = zero;
    }
  }

  XLS_ASSIGN_OR_RETURN(InterpValue limit_lt_zero, limit.Lt(zero));
  if (limit_lt_zero.IsTrue()) {
    // Ditto.
    XLS_ASSIGN_OR_RETURN(limit, basis_length.Add(limit));
    XLS_ASSIGN_OR_RETURN(limit_lt_zero, limit.Lt(zero));
    if (limit_lt_zero.IsTrue()) {
      limit = zero;
    }
  }

  // If limit extends past the basis, then we truncate limit.
  XLS_ASSIGN_OR_RETURN(InterpValue limit_ge_basis_length,
                       limit.Ge(basis_length));
  if (limit_ge_basis_length.IsTrue()) {
    limit =
        InterpValue::MakeSBits(start_bit_count, basis.GetBitCount().value());
  }
  XLS_ASSIGN_OR_RETURN(InterpValue length, limit.Sub(start));

  // At this point, both start and length must be nonnegative, so we force them
  // to UBits, since Slice expects that.
  XLS_ASSIGN_OR_RETURN(int64_t start_value, start.GetBitValueInt64());
  XLS_ASSIGN_OR_RETURN(int64_t length_value, length.GetBitValueInt64());
  XLS_RET_CHECK_GE(start_value, 0);
  XLS_RET_CHECK_GE(length_value, 0);
  start = InterpValue::MakeBits(/*is_signed=*/false, start.GetBitsOrDie());
  length = InterpValue::MakeBits(/*is_signed=*/false, length.GetBitsOrDie());
  XLS_ASSIGN_OR_RETURN(InterpValue result, basis.Slice(start, length));
  stack_.push_back(result);
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
  XLS_VLOG(2) << "jump_rel_if value: " << top.ToString();
  if (top.IsTrue()) {
    XLS_ASSIGN_OR_RETURN(Bytecode::JumpTarget target, bytecode.jump_target());
    return pc + target.value();
  }
  return std::nullopt;
}

absl::Status BytecodeInterpreter::EvalSub(const Bytecode& bytecode) {
  return EvalBinop([](const InterpValue& lhs, const InterpValue& rhs) {
    return lhs.Sub(rhs);
  });
}

absl::Status BytecodeInterpreter::EvalSwap(const Bytecode& bytecode) {
  XLS_RET_CHECK_GE(stack_.size(), 2);
  XLS_ASSIGN_OR_RETURN(InterpValue tos0, Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue tos1, Pop());
  stack_.push_back(tos0);
  stack_.push_back(tos1);
  return absl::OkStatus();
}

/* static */ absl::StatusOr<std::string> BytecodeInterpreter::TraceDataToString(
    const Bytecode::TraceData& trace_data, std::vector<InterpValue>& stack) {
  std::deque<std::string> pieces;
  for (int i = trace_data.size() - 1; i >= 0; i--) {
    std::variant<std::string, FormatPreference> trace_element =
        trace_data.at(i);
    if (std::holds_alternative<std::string>(trace_element)) {
      pieces.push_front(std::get<std::string>(trace_element));
    } else {
      XLS_RET_CHECK(!stack.empty());
      XLS_ASSIGN_OR_RETURN(InterpValue value, Pop(stack));
      pieces.push_front(value.ToString(
          /*humanize=*/true, std::get<FormatPreference>(trace_element)));
    }
  }

  return absl::StrJoin(pieces, "");
}

absl::Status BytecodeInterpreter::EvalTrace(const Bytecode& bytecode) {
  XLS_ASSIGN_OR_RETURN(const Bytecode::TraceData* trace_data,
                       bytecode.trace_data());
  XLS_ASSIGN_OR_RETURN(std::string message,
                       TraceDataToString(*trace_data, stack_));
  // Note: trace is specified to log to the INFO log.
  XLS_LOG(INFO) << message;
  stack_.push_back(InterpValue::MakeToken());
  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::EvalWidthSlice(const Bytecode& bytecode) {
  XLS_ASSIGN_OR_RETURN(const ConcreteType* type, bytecode.type_data());
  const BitsType* bits_type = dynamic_cast<const BitsType*>(type);
  XLS_RET_CHECK_NE(bits_type, nullptr);
  XLS_ASSIGN_OR_RETURN(int64_t width_value, bits_type->size().GetAsInt64());

  InterpValue oob_value(InterpValue::MakeUBits(width_value, /*value=*/0));
  XLS_ASSIGN_OR_RETURN(InterpValue start, Pop());
  if (!start.FitsInUint64()) {
    XLS_RETURN_IF_ERROR(Pop().status());
    stack_.push_back(oob_value);
    return absl::OkStatus();
  }
  XLS_ASSIGN_OR_RETURN(uint64_t start_index, start.GetBitValueUint64());

  XLS_ASSIGN_OR_RETURN(InterpValue basis, Pop());
  XLS_ASSIGN_OR_RETURN(Bits basis_bits, basis.GetBits());
  XLS_ASSIGN_OR_RETURN(int64_t basis_width, basis.GetBitCount());
  InterpValue width = InterpValue::MakeUBits(64, width_value);

  if (start_index >= basis_width) {
    stack_.push_back(oob_value);
    return absl::OkStatus();
  }

  if (start_index + width_value > basis_width) {
    basis_bits = bits_ops::ZeroExtend(basis_bits, start_index + width_value);
  }

  Bits result_bits = basis_bits.Slice(start_index, width_value);
  InterpValueTag tag =
      bits_type->is_signed() ? InterpValueTag::kSBits : InterpValueTag::kUBits;
  XLS_ASSIGN_OR_RETURN(InterpValue result,
                       InterpValue::MakeBits(tag, result_bits));
  stack_.push_back(result);
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
    case Builtin::kAddWithCarry:
      return RunBuiltinAddWithCarry(bytecode);
    case Builtin::kAndReduce:
      return RunBuiltinAndReduce(bytecode);
    case Builtin::kAssertEq:
      return RunBuiltinAssertEq(bytecode);
    case Builtin::kAssertLt:
      return RunBuiltinAssertLt(bytecode);
    case Builtin::kBitSlice:
      return RunBuiltinBitSlice(bytecode);
    case Builtin::kBitSliceUpdate:
      return RunBuiltinBitSliceUpdate(bytecode);
    case Builtin::kClz:
      return RunBuiltinClz(bytecode);
    case Builtin::kCover:
      stack_.push_back(InterpValue::MakeToken());
      return absl::OkStatus();
    case Builtin::kCtz:
      return RunBuiltinCtz(bytecode);
    case Builtin::kEnumerate:
      return RunBuiltinEnumerate(bytecode);
    case Builtin::kFail: {
      XLS_ASSIGN_OR_RETURN(InterpValue value, Pop());
      return FailureErrorStatus(bytecode.source_span(), value.ToString());
    }
    case Builtin::kGate:
      return RunBuiltinGate(bytecode);
    case Builtin::kMap:
      return RunBuiltinMap(bytecode);
    case Builtin::kOneHot:
      return RunBuiltinOneHot(bytecode);
    case Builtin::kOneHotSel:
      return RunBuiltinOneHotSel(bytecode);
    case Builtin::kPriorityhSel:
      return RunBuiltinPrioritySel(bytecode);
    case Builtin::kOrReduce:
      return RunBuiltinOrReduce(bytecode);
    case Builtin::kRange:
      return RunBuiltinRange(bytecode);
    case Builtin::kRev:
      return RunBuiltinRev(bytecode);
    case Builtin::kSignex:
      return RunBuiltinSignex(bytecode);
    case Builtin::kSlice:
      return RunBuiltinSlice(bytecode);
    case Builtin::kSMulp:
      return RunBuiltinSMulp(bytecode);
    case Builtin::kTrace:
      return absl::InternalError(
          "`trace!` builtins should be converted into kTrace opcodes.");
    case Builtin::kUMulp:
      return RunBuiltinUMulp(bytecode);
    case Builtin::kUpdate:
      return RunBuiltinUpdate(bytecode);
    case Builtin::kXorReduce:
      return RunBuiltinXorReduce(bytecode);
    default:
      return absl::UnimplementedError(
          absl::StrFormat("Builtin function \"%s\" not yet implemented.",
                          BuiltinToString(builtin)));
  }
}

absl::Status BytecodeInterpreter::RunBuiltinAddWithCarry(
    const Bytecode& bytecode) {
  XLS_VLOG(3) << "Executing builtin AddWithCarry.";
  XLS_RET_CHECK_GE(stack_.size(), 2);
  XLS_ASSIGN_OR_RETURN(InterpValue lhs, Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue rhs, Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue result, lhs.AddWithCarry(rhs));
  stack_.push_back(result);
  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::RunBuiltinAndReduce(
    const Bytecode& bytecode) {
  XLS_VLOG(3) << "Executing builtin AndReduce.";
  XLS_RET_CHECK(!stack_.empty());
  XLS_ASSIGN_OR_RETURN(InterpValue value, Pop());
  XLS_ASSIGN_OR_RETURN(Bits bits, value.GetBits());
  bits = bits_ops::AndReduce(bits);
  stack_.push_back(InterpValue::MakeBool(bits.IsOne()));
  return absl::OkStatus();
}

absl::StatusOr<std::string> PrettyPrintValue(const InterpValue& value,
                                             const ConcreteType* type,
                                             int indent = 0) {
  std::string indent_str(indent * 4, ' ');
  if (const auto* array_type = dynamic_cast<const ArrayType*>(type);
      array_type != nullptr) {
    const ConcreteType* element_type = &array_type->element_type();
    std::vector<std::string> elements;
    for (const auto& element_value : value.GetValuesOrDie()) {
      XLS_ASSIGN_OR_RETURN(
          std::string element,
          PrettyPrintValue(element_value, element_type, indent + 1));
      elements.push_back(element);
    }

    std::string element_type_name = element_type->ToString();
    std::string value_prefix = "";
    std::string separator = ", ";
    std::string value_suffix = "";
    if (const auto* struct_type = dynamic_cast<const StructType*>(element_type);
        struct_type != nullptr) {
      element_type_name = struct_type->nominal_type().identifier();
      std::string next_indent((indent + 1) * 4, ' ');
      value_prefix = absl::StrCat("\n", next_indent);
      separator = absl::StrCat(",", value_prefix);
      value_suffix = absl::StrCat("\n", indent_str);
    }

    return absl::StrFormat("%s[%d]:[%s%s%s]", element_type_name,
                           value.GetValuesOrDie().size(), value_prefix,
                           absl::StrJoin(elements, separator), value_suffix);
  }

  if (const auto* struct_type = dynamic_cast<const StructType*>(type);
      struct_type != nullptr) {
    std::vector<std::string> members;
    members.reserve(struct_type->size());
    for (int i = 0; i < struct_type->size(); i++) {
      std::string sub_indent_str((indent + 1) * 4, ' ');
      const ConcreteType& member_type = struct_type->GetMemberType(i);
      XLS_ASSIGN_OR_RETURN(std::string member_value,
                           PrettyPrintValue(value.GetValuesOrDie()[i],
                                            &member_type, indent + 1));
      members.push_back(absl::StrFormat("%s%s: %s", sub_indent_str,
                                        struct_type->GetMemberName(i),
                                        member_value));
    }

    return absl::StrFormat("%s {\n%s\n%s}",
                           struct_type->nominal_type().identifier(),
                           absl::StrJoin(members, "\n"), indent_str);
  }
  return value.ToString();
}

absl::Status BytecodeInterpreter::RunBuiltinAssertEq(const Bytecode& bytecode) {
  XLS_VLOG(3) << "Executing builtin AssertEq.";
  XLS_RET_CHECK_GE(stack_.size(), 2);
  // Get copies of the args for error reporting, but don't Pop(), as that's done
  // in EvalEq().
  InterpValue lhs = stack_[stack_.size() - 2];
  InterpValue rhs = stack_[stack_.size() - 1];

  XLS_RETURN_IF_ERROR(EvalEq(bytecode));
  if (stack_.back().IsFalse()) {
    const TypeInfo* type_info = frames_.back().type_info();
    XLS_ASSIGN_OR_RETURN(Bytecode::InvocationData invocation_data,
                         bytecode.invocation_data());
    XLS_ASSIGN_OR_RETURN(
        ConcreteType * lhs_type,
        type_info->GetItemOrError(invocation_data.invocation->args()[0]));
    XLS_ASSIGN_OR_RETURN(
        ConcreteType * rhs_type,
        type_info->GetItemOrError(invocation_data.invocation->args()[1]));
    XLS_ASSIGN_OR_RETURN(std::string pretty_lhs,
                         PrettyPrintValue(lhs, lhs_type));
    XLS_ASSIGN_OR_RETURN(std::string pretty_rhs,
                         PrettyPrintValue(rhs, rhs_type));
    std::string message = absl::StrFormat(
        "\n  lhs: %s\n  rhs: %s\n  were not equal", pretty_lhs, pretty_rhs);
    if (lhs.IsArray() && rhs.IsArray()) {
      XLS_ASSIGN_OR_RETURN(
          std::optional<int64_t> i,
          FindFirstDifferingIndex(lhs.GetValuesOrDie(), rhs.GetValuesOrDie()));
      XLS_RET_CHECK(i.has_value());
      const auto& lhs_values = lhs.GetValuesOrDie();
      const auto& rhs_values = rhs.GetValuesOrDie();
      message += absl::StrFormat("; first differing index: %d :: %s vs %s", *i,
                                 lhs_values[*i].ToHumanString(),
                                 rhs_values[*i].ToHumanString());
    }
    return FailureErrorStatus(bytecode.source_span(), message);
  }

  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::RunBuiltinAssertLt(const Bytecode& bytecode) {
  XLS_VLOG(3) << "Executing builtin AssertLt.";
  XLS_RET_CHECK_GE(stack_.size(), 2);
  // Get copies of the args for error reporting, but don't Pop(), as that's done
  // in EvalLt().
  InterpValue lhs = stack_[stack_.size() - 2];
  InterpValue rhs = stack_[stack_.size() - 1];

  XLS_RETURN_IF_ERROR(EvalLt(bytecode));
  if (stack_.back().IsFalse()) {
    const TypeInfo* type_info = frames_.back().type_info();
    XLS_ASSIGN_OR_RETURN(Bytecode::InvocationData invocation_data,
                         bytecode.invocation_data());
    XLS_ASSIGN_OR_RETURN(
        ConcreteType * lhs_type,
        type_info->GetItemOrError(invocation_data.invocation->args()[0]));
    XLS_ASSIGN_OR_RETURN(
        ConcreteType * rhs_type,
        type_info->GetItemOrError(invocation_data.invocation->args()[1]));
    XLS_ASSIGN_OR_RETURN(std::string pretty_lhs,
                         PrettyPrintValue(lhs, lhs_type));
    XLS_ASSIGN_OR_RETURN(std::string pretty_rhs,
                         PrettyPrintValue(rhs, rhs_type));
    std::string message = absl::StrFormat(
        "\n  lhs: %s\n was not less than rhs: %s", pretty_lhs, pretty_rhs);
    if (lhs.IsArray() && rhs.IsArray()) {
      const auto& lhs_values = lhs.GetValuesOrDie();
      const auto& rhs_values = rhs.GetValuesOrDie();
      int first_idx = -1;
      for (int i = 0; i < lhs_values.size(); i++) {
        if (rhs_values[i] >= lhs_values[i]) {
          first_idx = i;
          break;
        }
      }
      XLS_RET_CHECK_NE(first_idx, -1);
      message +=
          absl::StrFormat("; first differing index: %d :: %s vs %s", first_idx,
                          lhs_values[first_idx].ToHumanString(),
                          rhs_values[first_idx].ToHumanString());
    }
    return FailureErrorStatus(bytecode.source_span(), message);
  }

  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::RunBuiltinBitSlice(const Bytecode& bytecode) {
  XLS_VLOG(3) << "Executing builtin BitSlice.";
  return RunTernaryBuiltin(
      [](const InterpValue& subject, const InterpValue& start,
         const InterpValue& width) -> absl::StatusOr<InterpValue> {
        XLS_ASSIGN_OR_RETURN(Bits subject_bits, subject.GetBits());
        XLS_ASSIGN_OR_RETURN(int64_t start_index, start.GetBitValueInt64());
        if (start_index >= subject_bits.bit_count()) {
          start_index = subject_bits.bit_count();
        }
        XLS_ASSIGN_OR_RETURN(int64_t bit_count, width.GetBitCount());
        return InterpValue::MakeBits(
            /*is_signed=*/false, subject_bits.Slice(start_index, bit_count));
      });

  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::RunBuiltinBitSliceUpdate(
    const Bytecode& bytecode) {
  XLS_VLOG(3) << "Executing builtin BitSliceUpdate.";

  return RunTernaryBuiltin(
      [](const InterpValue& subject, const InterpValue& start,
         const InterpValue& update_value) -> absl::StatusOr<InterpValue> {
        XLS_ASSIGN_OR_RETURN(Bits subject_bits, subject.GetBits());
        XLS_ASSIGN_OR_RETURN(Bits start_bits, start.GetBits());
        XLS_ASSIGN_OR_RETURN(Bits update_value_bits, update_value.GetBits());

        if (bits_ops::UGreaterThanOrEqual(start_bits,
                                          subject_bits.bit_count())) {
          // Update is entirely out of bounds, so no bits of the subject are
          // updated.
          return InterpValue::MakeBits(InterpValueTag::kUBits, subject_bits);
        }

        XLS_ASSIGN_OR_RETURN(int64_t start_index, start_bits.ToUint64());
        return InterpValue::MakeBits(
            InterpValueTag::kUBits,
            bits_ops::BitSliceUpdate(subject_bits, start_index,
                                     update_value_bits));
      });
}

absl::Status BytecodeInterpreter::RunBuiltinClz(const Bytecode& bytecode) {
  XLS_VLOG(3) << "Executing builtin BitSliceUpdate.";
  XLS_RET_CHECK(!stack_.empty());

  XLS_ASSIGN_OR_RETURN(InterpValue input, Pop());
  XLS_ASSIGN_OR_RETURN(Bits bits, input.GetBits());
  stack_.push_back(
      InterpValue::MakeUBits(bits.bit_count(), bits.CountLeadingZeros()));

  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::RunBuiltinCtz(const Bytecode& bytecode) {
  XLS_VLOG(3) << "Executing builtin BitSliceUpdate.";
  XLS_RET_CHECK(!stack_.empty());

  XLS_ASSIGN_OR_RETURN(InterpValue input, Pop());
  XLS_ASSIGN_OR_RETURN(Bits bits, input.GetBits());
  stack_.push_back(
      InterpValue::MakeUBits(bits.bit_count(), bits.CountTrailingZeros()));

  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::RunBuiltinEnumerate(
    const Bytecode& bytecode) {
  XLS_RET_CHECK(!stack_.empty());
  XLS_ASSIGN_OR_RETURN(InterpValue input, Pop());
  XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* values,
                       input.GetValues());

  std::vector<InterpValue> elements;
  elements.reserve(values->size());
  for (int i = 0; i < values->size(); i++) {
    elements.push_back(
        InterpValue::MakeTuple({InterpValue::MakeU32(i), values->at(i)}));
  }
  XLS_ASSIGN_OR_RETURN(InterpValue result, InterpValue::MakeArray(elements));
  stack_.push_back(result);
  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::RunBuiltinGate(const Bytecode& bytecode) {
  return RunBinaryBuiltin(
      [](const InterpValue& pass_value,
         const InterpValue& value) -> absl::StatusOr<InterpValue> {
        if (pass_value.IsTrue()) {
          return value;
        }

        return CreateZeroValue(value);
      });
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
  Span span = invocation_data.invocation->span();

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
  int top_of_loop_index = bytecodes.size();
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
  bytecodes.push_back(Bytecode(span, Bytecode::Op::kAdd));
  bytecodes.push_back(
      Bytecode(span, Bytecode::Op::kStore, Bytecode::SlotIndex(1)));

  // Is index < input_size?
  bytecodes.push_back(
      Bytecode(span, Bytecode::Op::kLoad, Bytecode::SlotIndex(1)));
  bytecodes.push_back(
      Bytecode(span, Bytecode::Op::kLiteral,
               InterpValue::MakeU32(elements->size())));
  bytecodes.push_back(Bytecode(span, Bytecode::Op::kLt));

  // If true, jump to top-of-loop, else create the result array.
  bytecodes.push_back(
      Bytecode(span, Bytecode::Op::kJumpRelIf,
               Bytecode::JumpTarget(top_of_loop_index - bytecodes.size())));
  bytecodes.push_back(
      Bytecode(span, Bytecode::Op::kCreateArray,
               Bytecode::NumElements(elements->size())));

  // Now take the collected bytecodes and cram them into a BytecodeFunction,
  // then start executing it.
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<BytecodeFunction> bf,
                       BytecodeFunction::Create(
                           /*owner=*/nullptr, /*source_fn=*/nullptr,
                           frames_.back().type_info(), std::move(bytecodes)));
  BytecodeFunction* bf_ptr = bf.get();
  frames_.push_back(Frame(bf_ptr, {inputs}, bf_ptr->type_info(),
                          invocation_data.bindings,
                          /*initial_args=*/{}, std::move(bf)));
  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::RunBuiltinOneHot(const Bytecode& bytecode) {
  return RunBinaryBuiltin(
      [](const InterpValue& input,
         const InterpValue& lsb_is_prio) -> absl::StatusOr<InterpValue> {
        return input.OneHot(lsb_is_prio.IsTrue());
      });
}

absl::Status BytecodeInterpreter::RunBuiltinOneHotSel(
    const Bytecode& bytecode) {
  return RunBinaryBuiltin(
      [](const InterpValue& selector,
         const InterpValue& cases_array) -> absl::StatusOr<InterpValue> {
        XLS_ASSIGN_OR_RETURN(Bits selector_bits, selector.GetBits());
        XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* cases,
                             cases_array.GetValues());
        if (cases->empty()) {
          return absl::InternalError(
              "At least one case must be specified for one_hot_sel.");
        }
        XLS_ASSIGN_OR_RETURN(int64_t result_bit_count,
                             cases->at(0).GetBitCount());
        Bits result(result_bit_count);
        for (int i = 0; i < cases->size(); i++) {
          if (!selector_bits.Get(i)) {
            continue;
          }

          XLS_ASSIGN_OR_RETURN(Bits case_bits, cases->at(i).GetBits());
          result = bits_ops::Or(result, case_bits);
        }

        return InterpValue::MakeBits(cases->at(0).tag(), result);
      });
}

absl::Status BytecodeInterpreter::RunBuiltinPrioritySel(
    const Bytecode& bytecode) {
  return RunBinaryBuiltin(
      [](const InterpValue& selector,
         const InterpValue& cases_array) -> absl::StatusOr<InterpValue> {
        XLS_ASSIGN_OR_RETURN(Bits selector_bits, selector.GetBits());
        XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* cases,
                             cases_array.GetValues());
        if (cases->empty()) {
          return absl::InternalError(
              "At least one case must be specified for priority_sel.");
        }
        XLS_ASSIGN_OR_RETURN(int64_t result_bit_count,
                             cases->at(0).GetBitCount());
        for (int i = 0; i < cases->size(); i++) {
          if (selector_bits.Get(i)) {
            XLS_ASSIGN_OR_RETURN(Bits case_bits, cases->at(i).GetBits());
            return InterpValue::MakeBits(cases->at(0).tag(), case_bits);
          }
        }

        Bits empty_result(result_bit_count);
        return InterpValue::MakeBits(cases->at(0).tag(), empty_result);
      });
}

absl::Status BytecodeInterpreter::RunBuiltinOrReduce(const Bytecode& bytecode) {
  XLS_VLOG(3) << "Executing builtin OrReduce.";
  XLS_RET_CHECK(!stack_.empty());
  XLS_ASSIGN_OR_RETURN(InterpValue value, Pop());
  XLS_ASSIGN_OR_RETURN(Bits bits, value.GetBits());
  bits = bits_ops::OrReduce(bits);
  stack_.push_back(InterpValue::MakeBool(bits.IsOne()));
  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::RunBuiltinRange(const Bytecode& bytecode) {
  return RangeInternal();
}

absl::Status BytecodeInterpreter::RunBuiltinRev(const Bytecode& bytecode) {
  XLS_RET_CHECK(!stack_.empty());
  XLS_ASSIGN_OR_RETURN(InterpValue value, Pop());
  if (!value.IsBits() || value.IsSigned()) {
    return absl::InvalidArgumentError(
        "Argument to `rev` builtin must be an unsigned bits-typed value.");
  }
  XLS_ASSIGN_OR_RETURN(Bits bits, value.GetBits());
  stack_.push_back(
      InterpValue::MakeBits(/*is_signed=*/false, bits_ops::Reverse(bits)));
  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::RunBuiltinSignex(const Bytecode& bytecode) {
  return RunBinaryBuiltin(
      [](const InterpValue& value,
         const InterpValue& type_value) -> absl::StatusOr<InterpValue> {
        XLS_ASSIGN_OR_RETURN(int64_t old_bit_count, value.GetBitCount());
        XLS_ASSIGN_OR_RETURN(int64_t new_bit_count, type_value.GetBitCount());
        if (old_bit_count > new_bit_count) {
          return absl::InternalError(absl::StrCat(
              "Old bit count must be less than or equal to the new: ",
              old_bit_count, " vs. ", new_bit_count, "."));
        }
        XLS_ASSIGN_OR_RETURN(Bits bits, value.GetBits());

        return InterpValue::MakeBits(value.IsSigned(),
                                     bits_ops::SignExtend(bits, new_bit_count));
      });
}

absl::Status BytecodeInterpreter::RunBuiltinSlice(const Bytecode& bytecode) {
  return RunTernaryBuiltin([](const InterpValue& basis,
                              const InterpValue& start,
                              const InterpValue& type_value) {
    return basis.Slice(start, type_value);
  });
}

absl::Status BytecodeInterpreter::RunBuiltinSMulp(const Bytecode& bytecode) {
  return RunBinaryBuiltin(
      [](const InterpValue& lhs,
         const InterpValue& rhs) -> absl::StatusOr<InterpValue> {
        XLS_ASSIGN_OR_RETURN(int64_t lhs_bitwidth, lhs.GetBitCount());
        XLS_ASSIGN_OR_RETURN(int64_t rhs_bitwidth, lhs.GetBitCount());
        XLS_CHECK_EQ(lhs_bitwidth, rhs_bitwidth);
        int64_t product_bitwidth = lhs_bitwidth;
        std::vector<InterpValue> outputs;
        outputs.push_back(InterpValue::MakeSBits(product_bitwidth, 0));
        XLS_ASSIGN_OR_RETURN(InterpValue product, lhs.Mul(rhs));
        outputs.push_back(product);
        return InterpValue::MakeTuple(outputs);
      });
}

absl::Status BytecodeInterpreter::RunBuiltinUMulp(const Bytecode& bytecode) {
  return RunBinaryBuiltin(
      [](const InterpValue& lhs,
         const InterpValue& rhs) -> absl::StatusOr<InterpValue> {
        XLS_ASSIGN_OR_RETURN(int64_t lhs_bitwidth, lhs.GetBitCount());
        XLS_ASSIGN_OR_RETURN(int64_t rhs_bitwidth, lhs.GetBitCount());
        XLS_CHECK_EQ(lhs_bitwidth, rhs_bitwidth);
        int64_t product_bitwidth = lhs_bitwidth;
        std::vector<InterpValue> outputs;
        outputs.push_back(InterpValue::MakeUBits(product_bitwidth, 0));
        XLS_ASSIGN_OR_RETURN(InterpValue product, lhs.Mul(rhs));
        outputs.push_back(product);
        return InterpValue::MakeTuple(outputs);
      });
}

absl::Status BytecodeInterpreter::RunBuiltinUpdate(const Bytecode& bytecode) {
  return RunTernaryBuiltin([](const InterpValue& array,
                              const InterpValue& index,
                              const InterpValue& new_value) {
    return array.Update(index, new_value);
  });
}

absl::Status BytecodeInterpreter::RunBuiltinXorReduce(
    const Bytecode& bytecode) {
  XLS_VLOG(3) << "Executing builtin XorReduce.";
  XLS_RET_CHECK(!stack_.empty());
  XLS_ASSIGN_OR_RETURN(InterpValue value, Pop());
  XLS_ASSIGN_OR_RETURN(Bits bits, value.GetBits());
  bits = bits_ops::XorReduce(bits);
  stack_.push_back(InterpValue::MakeBool(bits.IsOne()));
  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::RunBinaryBuiltin(
    std::function<absl::StatusOr<InterpValue>(const InterpValue& a,
                                              const InterpValue& b)>
        fn) {
  XLS_RET_CHECK_GE(stack_.size(), 2);
  XLS_ASSIGN_OR_RETURN(InterpValue b, Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue a, Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue result, fn(a, b));
  stack_.push_back(result);
  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::RunTernaryBuiltin(
    std::function<absl::StatusOr<InterpValue>(
        const InterpValue& a, const InterpValue& b, const InterpValue& c)>
        fn) {
  XLS_RET_CHECK_GE(stack_.size(), 3);
  XLS_ASSIGN_OR_RETURN(InterpValue c, Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue b, Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue a, Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue result, fn(a, b, c));
  stack_.push_back(result);
  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::RangeInternal() {
  return RunBinaryBuiltin(
      [](const InterpValue& start,
         const InterpValue& end) -> absl::StatusOr<InterpValue> {
        XLS_RET_CHECK(start.IsBits());
        XLS_RET_CHECK(end.IsBits());
        XLS_ASSIGN_OR_RETURN(InterpValue start_ge_end, start.Ge(end));
        if (start_ge_end.IsTrue()) {
          return InterpValue::MakeArray({});
        }

        std::vector<InterpValue> elements;
        InterpValue cur = start;
        XLS_ASSIGN_OR_RETURN(InterpValue done, cur.Ge(end));
        XLS_ASSIGN_OR_RETURN(int64_t cur_bits, cur.GetBitCount());
        InterpValue one(cur.IsSigned() ? InterpValue::MakeSBits(cur_bits, 1)
                                       : InterpValue::MakeUBits(cur_bits, 1));
        while (done.IsFalse()) {
          elements.push_back(cur);
          XLS_ASSIGN_OR_RETURN(cur, cur.Add(one));
          XLS_ASSIGN_OR_RETURN(done, cur.Ge(end));
        }
        return InterpValue::MakeArray(elements);
      });
}

ProcConfigBytecodeInterpreter::ProcConfigBytecodeInterpreter(
    ImportData* import_data, BytecodeFunction* bf,
    std::vector<ProcInstance>* proc_instances)
    : BytecodeInterpreter(import_data, bf), proc_instances_(proc_instances) {}

absl::Status ProcConfigBytecodeInterpreter::InitializeProcNetwork(
    ImportData* import_data, TypeInfo* type_info, Proc* root_proc,
    InterpValue terminator, std::vector<ProcInstance>* proc_instances) {
  return EvalSpawn(import_data, type_info, absl::nullopt, absl::nullopt,
                   root_proc,
                   /*config_args=*/{terminator}, proc_instances);
}

absl::Status ProcConfigBytecodeInterpreter::EvalSpawn(
    const Bytecode& bytecode) {
  Frame& frame = frames().back();
  XLS_ASSIGN_OR_RETURN(const Bytecode::SpawnData* spawn_data,
                       bytecode.spawn_data());
  return EvalSpawn(import_data(), frame.type_info(),
                   spawn_data->caller_bindings, spawn_data->spawn,
                   spawn_data->proc, spawn_data->config_args, proc_instances_);
}

/* static */ absl::Status ProcConfigBytecodeInterpreter::EvalSpawn(
    ImportData* import_data, const TypeInfo* type_info,
    const std::optional<SymbolicBindings>& caller_bindings,
    std::optional<const Spawn*> maybe_spawn, Proc* proc,
    const std::vector<InterpValue>& config_args,
    std::vector<ProcInstance>* proc_instances) {
  const TypeInfo* parent_ti = type_info;
  auto get_parametric_type_info =
      [type_info](const Spawn* spawn, const Invocation* invoc,
                  const std::optional<SymbolicBindings>& caller_bindings)
      -> absl::StatusOr<TypeInfo*> {
    std::optional<TypeInfo*> maybe_type_info = type_info->GetInvocationTypeInfo(
        invoc, caller_bindings.has_value() ? caller_bindings.value()
                                           : SymbolicBindings());
    if (!maybe_type_info.has_value()) {
      return absl::InternalError(
          absl::StrCat("Could not find type info for invocation ",
                       spawn->ToString(), " : ", spawn->span().ToString()));
    }
    return maybe_type_info.value();
  };

  // We need to get a new TI if there's a spawn, i.e., this isn't a top-level
  // proc instantiation, to avoid constexpr values from colliding between
  // different proc instantiations.
  if (maybe_spawn.has_value()) {
    // We're guaranteed that these have values if the proc is parametric (the
    // root proc can't be parametric).
    XLS_ASSIGN_OR_RETURN(type_info,
                         get_parametric_type_info(maybe_spawn.value(),
                                                  maybe_spawn.value()->config(),
                                                  caller_bindings));
  }

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<BytecodeFunction> config_bf,
                       BytecodeEmitter::Emit(import_data, type_info,
                                             proc->config(), caller_bindings));

  ProcConfigBytecodeInterpreter cbi(import_data, config_bf.get(),
                                    proc_instances);
  XLS_RETURN_IF_ERROR(cbi.InitFrame(config_bf.get(), config_args, type_info));
  XLS_RETURN_IF_ERROR(cbi.Run());
  XLS_RET_CHECK_EQ(cbi.stack().size(), 1);
  InterpValue constants_tuple = cbi.stack().back();
  XLS_RET_CHECK(constants_tuple.IsTuple());
  XLS_ASSIGN_OR_RETURN(const std::vector<InterpValue>* constants,
                       constants_tuple.GetValues());

  // The Proc's "next" state includes all proc members (i.e., constants) as well
  // as the implicit token and the _actual_ state args themselves.
  std::vector<InterpValue> full_next_args = *constants;
  full_next_args.push_back(InterpValue::MakeToken());
  InterpValue initial_state(InterpValue::MakeToken());
  if (maybe_spawn.has_value()) {
    XLS_ASSIGN_OR_RETURN(
        initial_state,
        parent_ti->GetConstExpr(maybe_spawn.value()->next()->args()[0]));
  } else {
    // If this is the top-level proc, then we can get its initial state from the
    // ModuleMember typechecking, since A) top-level procs can't be
    // parameterized and B) typechecking will eagerly constexpr evaluate init
    // functions.
    XLS_ASSIGN_OR_RETURN(initial_state,
                         parent_ti->GetConstExpr(proc->init()->body()));
  }

  full_next_args.insert(full_next_args.end(), initial_state);

  std::vector<NameDef*> member_defs;
  member_defs.reserve(proc->members().size());
  for (const Param* param : proc->members()) {
    member_defs.push_back(param->name_def());
  }

  if (maybe_spawn.has_value()) {
    XLS_ASSIGN_OR_RETURN(
        type_info,
        get_parametric_type_info(maybe_spawn.value(),
                                 maybe_spawn.value()->next(), caller_bindings));
  }

  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<BytecodeFunction> next_bf,
      BytecodeEmitter::EmitProcNext(import_data, type_info, proc->next(),
                                    caller_bindings, member_defs));
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<BytecodeInterpreter> next_interpreter,
      CreateUnique(import_data, next_bf.get(), full_next_args));
  proc_instances->push_back(ProcInstance{proc, std::move(next_interpreter),
                                         std::move(next_bf), full_next_args});
  return absl::OkStatus();
}

absl::Status ProcInstance::Run() {
  absl::Status result_status = interpreter_->Run();

  if (result_status.ok()) {
    InterpValue result_value = interpreter_->stack_.back();
    // If we're starting from next fn top, then set [non-member] args for the
    // next go-around.
    // Don't forget to add the [implicit] token!
    // If we get an empty tuple and the proc has no recurrent state, then don't
    // add it.
    if (next_args_.size() == proc_->members().size() + 2) {
      next_args_[proc_->members().size() + 1] = result_value;
    } else {
      XLS_QCHECK(result_value.IsTuple() &&
                 result_value.GetLength().value() == 0);
    }

    XLS_RETURN_IF_ERROR(interpreter_->InitFrame(next_fn_.get(), next_args_,
                                                /*type_info=*/nullptr));
    return absl::OkStatus();
  }

  if (result_status.code() == absl::StatusCode::kUnavailable) {
    // Empty recv channel. Just return Ok and we'll try again next time.
    return absl::OkStatus();
  }

  return result_status;
}

}  // namespace xls::dslx
