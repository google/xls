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

#include "absl/status/status.h"
#include "xls/common/status/ret_check.h"
#include "xls/dslx/ast.h"
#include "xls/dslx/builtins.h"
#include "xls/dslx/interp_value.h"

namespace xls::dslx {

absl::Status BytecodeInterpreter::Run() {
  int64_t pc = 0;
  while (pc < bytecode_.size()) {
    XLS_VLOG(2) << std::hex << "PC: " << pc << " : "
                << bytecode_.at(pc).ToString();
    const Bytecode& bytecode = bytecode_.at(pc);
    XLS_ASSIGN_OR_RETURN(int64_t new_pc, EvalInstruction(pc, bytecode));
    if (new_pc != pc + 1) {
      XLS_RET_CHECK(bytecode_.at(new_pc).op() == Bytecode::Op::kJumpDest)
          << "Jumping from PC " << pc << " to PC: " << new_pc
          << " bytecode: " << bytecode_.at(new_pc).ToString()
          << " not a jump_dest";
    }
    pc = new_pc;
  }
  return absl::OkStatus();
}

absl::StatusOr<int64_t> BytecodeInterpreter::EvalInstruction(
    int64_t pc, const Bytecode& bytecode) {
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
    case Bytecode::Op::kEq: {
      XLS_RETURN_IF_ERROR(EvalEq(bytecode));
      break;
    }
    case Bytecode::Op::kExpandTuple: {
      XLS_RETURN_IF_ERROR(EvalExpandTuple(bytecode));
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
    case Bytecode::Op::kJumpRel:
      return pc + bytecode.integer_data().value();
    case Bytecode::Op::kJumpRelIf: {
      XLS_ASSIGN_OR_RETURN(
          std::optional<int64_t> new_pc, EvalJumpRelIf(pc, bytecode));
      if (new_pc.has_value()) {
        return new_pc.value();
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
    case Bytecode::Op::kShll: {
      XLS_RETURN_IF_ERROR(EvalShll(bytecode));
      break;
    }
    case Bytecode::Op::kShrl: {
      XLS_RETURN_IF_ERROR(EvalShrl(bytecode));
      break;
    }
    case Bytecode::Op::kSlice: {
      XLS_RETURN_IF_ERROR(EvalSlice(bytecode));
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
    case Bytecode::Op::kWidthSlice: {
      XLS_RETURN_IF_ERROR(EvalWidthSlice(bytecode));
      break;
    }
    case Bytecode::Op::kXor: {
      XLS_RETURN_IF_ERROR(EvalXor(bytecode));
      break;
    }
  }
  return pc + 1;
}

absl::StatusOr<InterpValue> BytecodeInterpreter::Pop() {
  if (stack_.empty()) {
    return absl::InternalError("Tried to pop off an empty stack.");
  }
  InterpValue value = std::move(stack_.back());
  stack_.pop_back();
  return value;
}

absl::Status BytecodeInterpreter::EvalBinop(
    const std::function<absl::StatusOr<InterpValue>(const InterpValue& lhs,
                                                    const InterpValue& rhs)>
        op) {
  XLS_RET_CHECK_GE(stack_.size(), 2);
  XLS_ASSIGN_OR_RETURN(InterpValue rhs, Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue lhs, Pop());
  XLS_ASSIGN_OR_RETURN(InterpValue result, op(lhs, rhs));
  stack_.push_back(result);
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

absl::Status BytecodeInterpreter::EvalCall(const Bytecode& bytecode) {
  XLS_ASSIGN_OR_RETURN(InterpValue value, bytecode.value_data());
  if (value.IsBuiltinFunction()) {
    return RunBuiltinFn(bytecode, absl::get<Builtin>(value.GetFunctionOrDie()));
  }

  return absl::UnimplementedError(
      "User-defined functions are not yet supported.");
}

absl::Status BytecodeInterpreter::EvalConcat(const Bytecode& bytecode) {
  return EvalBinop([](const InterpValue& lhs, const InterpValue& rhs) {
    return lhs.Concat(rhs);
  });
}

absl::Status BytecodeInterpreter::EvalCreateArray(const Bytecode& bytecode) {
  XLS_ASSIGN_OR_RETURN(int64_t array_size, bytecode.integer_data());
  XLS_RET_CHECK_GE(stack_.size(), array_size);

  std::vector<InterpValue> elements;
  elements.reserve(array_size);
  for (int64_t i = 0; i < array_size; i++) {
    XLS_ASSIGN_OR_RETURN(InterpValue value, Pop());
    elements.push_back(value);
  }

  std::reverse(elements.begin(), elements.end());
  XLS_ASSIGN_OR_RETURN(InterpValue array, InterpValue::MakeArray(elements));
  stack_.push_back(array);
  return absl::OkStatus();
}

absl::Status BytecodeInterpreter::EvalCreateTuple(const Bytecode& bytecode) {
  XLS_ASSIGN_OR_RETURN(int64_t tuple_size, bytecode.integer_data());
  XLS_RET_CHECK_GE(stack_.size(), tuple_size);

  std::vector<InterpValue> elements;
  elements.reserve(tuple_size);
  for (int64_t i = 0; i < tuple_size; i++) {
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

absl::Status BytecodeInterpreter::EvalEq(const Bytecode& bytecode) {
  return EvalBinop([](const InterpValue& lhs, const InterpValue& rhs) {
    return InterpValue::MakeBool(lhs.Eq(rhs));
  });
}

absl::Status BytecodeInterpreter::EvalExpandTuple(const Bytecode& bytecode) {
  InterpValue tuple = stack_.back();
  stack_.pop_back();
  if (!tuple.IsTuple()) {
    return absl::InvalidArgumentError(
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

  XLS_ASSIGN_OR_RETURN(InterpValue result, basis.Index(index));
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
  XLS_ASSIGN_OR_RETURN(int64_t slot, bytecode.integer_data());
  if (slots_->size() <= slot) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Attempted to access local data in slot %d, which out of range.",
        slot));
  }
  stack_.push_back(slots_->at(slot));
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

absl::Status BytecodeInterpreter::EvalShll(const Bytecode& bytecode) {
  return EvalBinop([](const InterpValue& lhs, const InterpValue& rhs) {
    return lhs.Shl(rhs);
  });
}

absl::Status BytecodeInterpreter::EvalShrl(const Bytecode& bytecode) {
  return EvalBinop([](const InterpValue& lhs, const InterpValue& rhs) {
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
      InterpValue::MakeSBits(basis_bit_count, basis_bit_count);

  XLS_ASSIGN_OR_RETURN(InterpValue start_lt_zero, start.Lt(zero));
  if (start_lt_zero.IsTrue()) {
    // Remember, start is negative if we're here.
    XLS_ASSIGN_OR_RETURN(start, basis_length.Add(start));
  }

  XLS_ASSIGN_OR_RETURN(InterpValue limit_lt_zero, limit.Lt(zero));
  if (limit_lt_zero.IsTrue()) {
    // Ditto.
    XLS_ASSIGN_OR_RETURN(limit, basis_length.Add(limit));
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
  XLS_ASSIGN_OR_RETURN(int64_t slot, bytecode.integer_data());
  if (stack_.empty()) {
    return absl::InvalidArgumentError(
        "Attempted to store value from empty stack.");
  }

  if (slots_->size() <= slot) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Attempted to access local data in slot %d, which out of range.",
        slot));
  }

  slots_->at(slot) = stack_.back();
  stack_.pop_back();
  return absl::OkStatus();
}

absl::StatusOr<std::optional<int64_t>> BytecodeInterpreter::EvalJumpRelIf(
    int64_t pc, const Bytecode& bytecode) {
  XLS_ASSIGN_OR_RETURN(InterpValue top, Pop());
  XLS_VLOG(2) << "jump_rel_if value: " << top.ToString();
  if (top.IsTrue()) {
    return pc + bytecode.integer_data().value();
  }
  return std::nullopt;
}

absl::Status BytecodeInterpreter::EvalSub(const Bytecode& bytecode) {
  return EvalBinop([](const InterpValue& lhs, const InterpValue& rhs) {
    return lhs.Sub(rhs);
  });
}

absl::Status BytecodeInterpreter::EvalWidthSlice(const Bytecode& bytecode) {
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
    case Builtin::kAssertEq:
      return RunBuiltinAssertEq(bytecode);
    default:
      return absl::UnimplementedError(
          absl::StrFormat("Builtin function \"%s\" not yet implemented.",
                          BuiltinToString(builtin)));
  }
}

absl::Status BytecodeInterpreter::RunBuiltinAssertEq(const Bytecode& bytecode) {
  XLS_VLOG(3) << "Executing builtin AssertEq.";
  XLS_RET_CHECK_GE(stack_.size(), 2);
  InterpValue lhs = stack_[stack_.size() - 2];
  InterpValue rhs = stack_[stack_.size() - 1];

  XLS_RETURN_IF_ERROR(EvalEq(bytecode));
  if (stack_.back().IsFalse()) {
    std::string message =
        absl::StrFormat("\n  lhs: %s\n  rhs: %s\n  were not equal",
                        lhs.ToHumanString(), rhs.ToHumanString());
    return FailureErrorStatus(bytecode.source_span(), message);
  }

  return absl::OkStatus();
}

}  // namespace xls::dslx
