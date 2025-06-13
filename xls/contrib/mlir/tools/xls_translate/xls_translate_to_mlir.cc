// Copyright 2024 The XLS Authors
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

#include "xls/contrib/mlir/tools/xls_translate/xls_translate_to_mlir.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/include/llvm/Support/MemoryBuffer.h"
#include "llvm/include/llvm/Support/SourceMgr.h"
#include "llvm/include/llvm/Support/raw_ostream.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"
#include "mlir/include/mlir/IR/Location.h"
#include "mlir/include/mlir/IR/MLIRContext.h"
#include "mlir/include/mlir/IR/OwningOpRef.h"
#include "mlir/include/mlir/IR/TypeRange.h"
#include "mlir/include/mlir/IR/Types.h"
#include "mlir/include/mlir/IR/Value.h"
#include "mlir/include/mlir/IR/ValueRange.h"
#include "mlir/include/mlir/Support/LLVM.h"
#include "xls/contrib/mlir/IR/xls_ops.h"
#include "xls/contrib/mlir/util/conversion_utils.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/fileno.h"
#include "xls/ir/format_strings.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/lsb_or_msb.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/public/ir_parser.h"

namespace mlir::xls {

//===----------------------------------------------------------------------===//
// Translation State
//===----------------------------------------------------------------------===//

class TranslationState {
 public:
  TranslationState() = default;

  absl::Status setFileName(::xls::Fileno num, StringAttr name) {
    if (!fileno_map_.insert({num, name}).second) {
      return absl::InternalError(
          absl::StrCat("Duplicate file number ", num.value()));
    }
    return absl::OkStatus();
  }

  absl::StatusOr<StringAttr> getFileName(::xls::Fileno no) {
    if (auto name = fileno_map_.find(no); name == fileno_map_.end()) {
      return absl::InternalError(
          absl::StrCat("Unknown file number  ", no.value()));
    } else {
      return name->second;
    }
  }

  absl::Status setFunction(const std::string& name, FlatSymbolRefAttr fn) {
    if (!function_map_.insert({name, fn}).second) {
      return absl::InternalError(
          absl::StrCat("Duplicate function name ", name));
    }
    return absl::OkStatus();
  }

  absl::StatusOr<FlatSymbolRefAttr> getFunction(const std::string& name) {
    if (auto fn = function_map_.find(name); fn == function_map_.end()) {
      return absl::InternalError(absl::StrCat("Unknown function name  ", name));
    } else {
      return fn->second;
    }
  }

  absl::Status setChannel(const std::string& name, FlatSymbolRefAttr chn) {
    if (!channel_map_.insert({name, chn}).second) {
      return absl::InternalError(absl::StrCat("Duplicate channel name ", name));
    }
    return absl::OkStatus();
  }

  absl::StatusOr<SymbolRefAttr> getChannel(const std::string& name) {
    if (auto chn = channel_map_.find(name); chn == channel_map_.end()) {
      return absl::InternalError(absl::StrCat("Unknown channel name  ", name));
    } else {
      return chn->second;
    }
  }

  absl::Status setMlirValue(int64_t id, Value op) {
    if (!ssa_map_.insert({id, op}).second) {
      return absl::InternalError(
          absl::StrCat("Duplicate assignment to value id ", id));
    }
    return absl::OkStatus();
  }

  absl::StatusOr<Value> getMlirValue(int64_t id) {
    if (auto val = ssa_map_.find(id); val == ssa_map_.end()) {
      return absl::InternalError(absl::StrCat("Unknown value id ", id));
    } else {
      return val->second;
    }
  }

  void newFunctionBaseContext(bool is_proc) {
    ssa_map_ = {};
    is_proc_ = is_proc;
    state_element_map = {};
  }

  void setStateElement(const std::string& name, Value elem) {
    assert(is_proc_ && "Only proc functions contain state elements");
    state_element_map[name] = elem;
  }

  absl::StatusOr<Value> getStateElement(const std::string id) {
    assert(is_proc_ && "Only proc functions contain state elements");
    if (auto elem = state_element_map.find(id);
        elem == state_element_map.end()) {
      return absl::InternalError(absl::StrCat("Unknown state element ", id));
    } else {
      return elem->second;
    }
  }

 private:
  // --- package/module state ----
  // Maps name -> MLIR ref for all functions in current
  // package/module.
  absl::flat_hash_map<std::string, FlatSymbolRefAttr> function_map_;
  // Maps name -> MLIR ref for all channels in current
  // package/module.
  absl::flat_hash_map<std::string, FlatSymbolRefAttr> channel_map_;
  // Maps XLS source file ID -> pathsourcefile_map_.
  absl::flat_hash_map<::xls::Fileno, StringAttr> fileno_map_;

  // --- function base (func/proc) state ----
  // Track if current function base is a proc or plain function. At the moment
  // this is for assertions/diagnostics only.
  [[maybe_unused]] bool is_proc_;

  // Maps id -> MLIR value for all SSA values inside the current
  // function base (function or proc body).
  absl::flat_hash_map<int64_t, Value> ssa_map_;

  // Maps name -> MLIR value for all state elements of this proc.
  absl::flat_hash_map<std::string, Value> state_element_map;
};

//===----------------------------------------------------------------------===//
// Type Translation
//===----------------------------------------------------------------------===//

Type translateType(::xls::Type* xls_type, OpBuilder& builder) {
  MLIRContext* ctx = builder.getContext();
  switch (xls_type->kind()) {
    case ::xls::TypeKind::kTuple: {
      SmallVector<Type> types;
      for (auto* type : xls_type->AsTupleOrDie()->element_types()) {
        types.push_back(translateType(type, builder));
      }
      return TupleType::get(ctx, types);
    }
    case ::xls::TypeKind::kBits: {
      return builder.getIntegerType(xls_type->AsBitsOrDie()->bit_count());
    }
    case ::xls::TypeKind::kArray: {
      return ArrayType::get(
          xls_type->AsArrayOrDie()->size(),
          translateType(xls_type->AsArrayOrDie()->element_type(), builder));
    }
    case ::xls::TypeKind::kToken: {
      return TokenType::get(ctx);
    }
  }
}

//===----------------------------------------------------------------------===//
// Literal Translation
//===----------------------------------------------------------------------===//

absl::StatusOr<Operation*> translateBitsLiteral(const ::xls::Bits& b,
                                                Location loc,
                                                OpBuilder& builder,
                                                TranslationState& state) {
  APInt converted_value = bitsToAPInt(b);
  auto type = builder.getIntegerType(b.bit_count());
  return builder
      .create<ConstantScalarOp>(loc, type,
                                builder.getIntegerAttr(type, converted_value))
      .getOperation();
}

absl::StatusOr<Operation*> translateLiteral(const ::xls::Value& b,
                                            OpBuilder& builder,
                                            TranslationState& state) {
  MLIRContext* ctx = builder.getContext();
  switch (b.kind()) {
    case ::xls::ValueKind::kBits:
      return translateBitsLiteral(b.bits(), builder.getUnknownLoc(), builder,
                                  state);

    case ::xls::ValueKind::kToken: {
      return builder.create<xls::AfterAllOp>(builder.getUnknownLoc(),
                                             ValueRange{});
    }

    case ::xls::ValueKind::kTuple:
    case ::xls::ValueKind::kArray: {
      SmallVector<Value> members;
      SmallVector<Type> types;

      for (auto& xls_member : b.elements()) {
        auto member = translateLiteral(xls_member, builder, state);
        if (!member.ok()) {
          return member.status();
        }
        members.push_back(member.value()->getResult(0));
        types.push_back(member.value()->getResult(0).getType());
      }

      if (b.IsArray()) {
        if (members.empty()) {
          return absl::InternalError("Empty arrays are not supported.");
        }
        Type result_type = ArrayType::get(types.size(), types[0]);
        return builder.create<xls::ArrayOp>(builder.getUnknownLoc(),
                                            result_type, ValueRange(members));
      } else {
        auto result_type = TupleType::get(ctx, types);
        return builder.create<xls::TupleOp>(builder.getUnknownLoc(),
                                            result_type, ValueRange(members));
      }

      break;
    }

    case ::xls::ValueKind::kInvalid:
    default: {
      return absl::InternalError(absl::StrCat(
          "Cannot construct literal. Unknown kind: ", b.ToHumanString()));
    }
  }
}

//===----------------------------------------------------------------------===//
// Source Location Translation
//===----------------------------------------------------------------------===//

absl::StatusOr<Location> translateLoc(const ::xls::SourceInfo& xls_loc,
                                      Builder& builder,
                                      TranslationState& state) {
  SmallVector<Location> locs;
  for (auto loc : xls_loc.locations) {
    auto filename = state.getFileName(loc.fileno());
    if (!filename.ok()) {
      return filename.status();
    }
    locs.push_back(FileLineColLoc::get(*filename, loc.fileno().value(),
                                       loc.colno().value()));
  }

  if (locs.empty()) {
    return builder.getUnknownLoc();
  } else if (locs.size() == 1) {
    return locs[0];
  } else {
    return builder.getFusedLoc(locs);
  }
}

//===----------------------------------------------------------------------===//
// Operation Translation
//===----------------------------------------------------------------------===//

absl::StatusOr<Operation*> translateOp(::xls::ArithOp& node, OpBuilder& builder,
                                       TranslationState& state) {
  auto opr_lhs =
      state.getMlirValue(node.operands()[::xls::ArithOp::kLhsOperand]->id());
  if (!opr_lhs.ok()) {
    return opr_lhs.status();
  }

  auto opr_rhs =
      state.getMlirValue(node.operands()[::xls::ArithOp::kRhsOperand]->id());
  if (!opr_rhs.ok()) {
    return opr_rhs.status();
  }

  auto result_type = translateType(node.GetType(), builder);
  auto loc = translateLoc(node.loc(), builder, state);
  if (!loc.ok()) {
    return loc.status();
  }

  switch (node.op()) {
    case ::xls::Op::kUMul:
      return builder.create<xls::UmulOp>(*loc, result_type, *opr_lhs, *opr_rhs);
    case ::xls::Op::kSMul:
      return builder.create<xls::SmulOp>(*loc, result_type, *opr_lhs, *opr_rhs);
    default:
      return absl::InternalError(absl::StrCat(
          "Expected ArithOp operation, not ", ::xls::OpToString(node.op())));
  }
}

absl::StatusOr<Operation*> translateOp(::xls::PartialProductOp& node,
                                       OpBuilder& builder,
                                       TranslationState& state) {
  auto opr_lhs = state.getMlirValue(
      node.operands()[::xls::PartialProductOp::kLhsOperand]->id());
  if (!opr_lhs.ok()) {
    return opr_lhs.status();
  }

  auto opr_rhs = state.getMlirValue(
      node.operands()[::xls::PartialProductOp::kRhsOperand]->id());
  if (!opr_rhs.ok()) {
    return opr_rhs.status();
  }

  auto result_type =
      translateType(node.GetType()->AsTupleOrDie()->element_type(0), builder);
  auto loc = translateLoc(node.loc(), builder, state);
  if (!loc.ok()) {
    return loc.status();
  }

  Operation* op = nullptr;
  switch (node.op()) {
    case ::xls::Op::kSMulp:
      op = builder.create<xls::SmulpOp>(*loc, result_type, *opr_lhs, *opr_rhs);
      break;
    case ::xls::Op::kUMulp:
      op = builder.create<xls::UmulpOp>(*loc, result_type, *opr_lhs, *opr_rhs);
      break;
    default:
      return absl::InternalError(
          absl::StrCat("Expected PartialProductOp operation, not ",
                       ::xls::OpToString(node.op())));
  }

  SmallVector<Value> result_tuple_elems;
  result_tuple_elems.push_back(op->getResult(0));
  result_tuple_elems.push_back(op->getResult(1));
  ValueRange result_tuple(result_tuple_elems);
  return builder.create<xls::TupleOp>(*loc, result_tuple);
}

absl::StatusOr<Operation*> translateOp(::xls::CompareOp& node,
                                       OpBuilder& builder,
                                       TranslationState& state) {
  auto opr_lhs =
      state.getMlirValue(node.operands()[::xls::CompareOp::kLhsOperand]->id());
  if (!opr_lhs.ok()) {
    return opr_lhs.status();
  }

  auto opr_rhs =
      state.getMlirValue(node.operands()[::xls::CompareOp::kRhsOperand]->id());
  if (!opr_rhs.ok()) {
    return opr_rhs.status();
  }

  auto result_type = translateType(node.GetType(), builder);
  auto loc = translateLoc(node.loc(), builder, state);
  if (!loc.ok()) {
    return loc.status();
  }

  switch (node.op()) {
    case ::xls::Op::kEq:
      return builder.create<xls::EqOp>(*loc, result_type, *opr_lhs, *opr_rhs);
    case ::xls::Op::kNe:
      return builder.create<xls::NeOp>(*loc, result_type, *opr_lhs, *opr_rhs);
    case ::xls::Op::kSLe:
      return builder.create<xls::SleOp>(*loc, result_type, *opr_lhs, *opr_rhs);
    case ::xls::Op::kSGe:
      return builder.create<xls::SgeOp>(*loc, result_type, *opr_lhs, *opr_rhs);
    case ::xls::Op::kSLt:
      return builder.create<xls::SltOp>(*loc, result_type, *opr_lhs, *opr_rhs);
    case ::xls::Op::kSGt:
      return builder.create<xls::SgtOp>(*loc, result_type, *opr_lhs, *opr_rhs);
    case ::xls::Op::kULe:
      return builder.create<xls::UleOp>(*loc, result_type, *opr_lhs, *opr_rhs);
    case ::xls::Op::kUGe:
      return builder.create<xls::UgeOp>(*loc, result_type, *opr_lhs, *opr_rhs);
    case ::xls::Op::kULt:
      return builder.create<xls::UltOp>(*loc, result_type, *opr_lhs, *opr_rhs);
    case ::xls::Op::kUGt:
      return builder.create<xls::UgtOp>(*loc, result_type, *opr_lhs, *opr_rhs);
    default:
      return absl::InternalError(absl::StrCat(
          "Expected CompareOp operation, not ", ::xls::OpToString(node.op())));
  }
}

absl::StatusOr<Operation*> translateOp(::xls::BinOp& node, OpBuilder& builder,
                                       TranslationState& state) {
  auto opr_lhs =
      state.getMlirValue(node.operands()[::xls::BinOp::kLhsOperand]->id());
  if (!opr_lhs.ok()) {
    return opr_lhs.status();
  }
  auto opr_rhs =
      state.getMlirValue(node.operands()[::xls::BinOp::kRhsOperand]->id());
  if (!opr_rhs.ok()) {
    return opr_rhs.status();
  }

  auto result_type = translateType(node.GetType(), builder);
  auto loc = translateLoc(node.loc(), builder, state);
  if (!loc.ok()) {
    return loc.status();
  }

  switch (node.op()) {
    case ::xls::Op::kAdd:
      return builder.create<xls::AddOp>(*loc, result_type, *opr_lhs, *opr_rhs);
    case ::xls::Op::kSDiv:
      return builder.create<xls::SdivOp>(*loc, result_type, *opr_lhs, *opr_rhs);
    case ::xls::Op::kSMod:
      return builder.create<xls::SmodOp>(*loc, result_type, *opr_lhs, *opr_rhs);
    case ::xls::Op::kShll:
      return builder.create<xls::ShllOp>(*loc, result_type, *opr_lhs, *opr_rhs);
    case ::xls::Op::kShrl:
      return builder.create<xls::ShrlOp>(*loc, result_type, *opr_lhs, *opr_rhs);
    case ::xls::Op::kShra:
      return builder.create<xls::ShraOp>(*loc, result_type, *opr_lhs, *opr_rhs);
    case ::xls::Op::kSub:
      return builder.create<xls::SubOp>(*loc, result_type, *opr_lhs, *opr_rhs);
    case ::xls::Op::kUDiv:
      return builder.create<xls::UdivOp>(*loc, result_type, *opr_lhs, *opr_rhs);
    case ::xls::Op::kUMod:
      return builder.create<xls::UmodOp>(*loc, result_type, *opr_lhs, *opr_rhs);
    default:
      return absl::InternalError(absl::StrCat("Expected BinOp operation, not ",
                                              ::xls::OpToString(node.op())));
  }
}

absl::StatusOr<Operation*> translateOp(::xls::UnOp& node, OpBuilder& builder,
                                       TranslationState& state) {
  auto operand =
      state.getMlirValue(node.operands()[::xls::UnOp::kArgOperand]->id());
  if (!operand.ok()) {
    return operand.status();
  }

  auto loc = translateLoc(node.loc(), builder, state);
  if (!loc.ok()) {
    return loc.status();
  }

  switch (node.op()) {
    case ::xls::Op::kIdentity:
      return builder.create<xls::IdentityOp>(*loc, *operand);
    case ::xls::Op::kNeg:
      return builder.create<xls::NegOp>(*loc, *operand);
    case ::xls::Op::kNot:
      return builder.create<xls::NotOp>(*loc, *operand);
    case ::xls::Op::kReverse:
      return builder.create<xls::ReverseOp>(*loc, *operand);
    default:
      return absl::InternalError(absl::StrCat("Expected UnOp operation, not ",
                                              ::xls::OpToString(node.op())));
  }
}

absl::StatusOr<Operation*> translateOp(::xls::ExtendOp& node,
                                       OpBuilder& builder,
                                       TranslationState& state) {
  auto operand =
      state.getMlirValue(node.operands()[::xls::ExtendOp::kArgOperand]->id());
  if (!operand.ok()) {
    return operand.status();
  }

  auto result_type = translateType(node.GetType(), builder);
  auto loc = translateLoc(node.loc(), builder, state);
  if (!loc.ok()) {
    return loc.status();
  }

  switch (node.op()) {
    case ::xls::Op::kZeroExt:
      return builder.create<xls::ZeroExtOp>(*loc, result_type, *operand);
    case ::xls::Op::kSignExt:
      return builder.create<xls::SignExtOp>(*loc, result_type, *operand);
    default:
      return absl::InternalError(absl::StrCat(
          "Expected ExtendOp operation, not ", ::xls::OpToString(node.op())));
  }
}

absl::StatusOr<Operation*> translateOp(::xls::TupleIndex& node,
                                       OpBuilder& builder,
                                       TranslationState& state) {
  auto operand =
      state.getMlirValue(node.operands()[::xls::TupleIndex::kArgOperand]->id());
  if (!operand.ok()) {
    return operand.status();
  }

  auto result_type = translateType(node.GetType(), builder);
  auto loc = translateLoc(node.loc(), builder, state);
  if (!loc.ok()) {
    return loc.status();
  }
  auto index = builder.getI64IntegerAttr(node.index());

  return builder.create<xls::TupleIndexOp>(*loc, result_type, *operand, index);
}

absl::StatusOr<Operation*> translateOp(::xls::Array& node, OpBuilder& builder,
                                       TranslationState& state) {
  SmallVector<Value> operands_vec;
  for (auto* xls_operand : node.operands()) {
    auto operand = state.getMlirValue(xls_operand->id());
    if (!operand.ok()) {
      return operand.status();
    }
    operands_vec.push_back(*operand);
  }
  ValueRange operands(operands_vec);

  auto result_type = translateType(node.GetType(), builder);
  auto loc = translateLoc(node.loc(), builder, state);
  if (!loc.ok()) {
    return loc.status();
  }

  return builder.create<xls::ArrayOp>(*loc, result_type, operands);
}

absl::StatusOr<Operation*> translateOp(::xls::ArrayIndex& node,
                                       OpBuilder& builder,
                                       TranslationState& state) {
  auto xls_arg =
      state.getMlirValue(node.operands()[::xls::ArrayIndex::kArgOperand]->id());
  if (!xls_arg.ok()) {
    return xls_arg.status();
  }

  if (node.indices().length() != 1) {
    return absl::InternalError(
        "MLIR currently only supports ArrayIndex with a single index!");
  }
  auto xls_index = state.getMlirValue(
      node.operands()[::xls::ArrayIndex::kIndexOperandStart]->id());
  if (!xls_arg.ok()) {
    return xls_arg.status();
  }

  auto result_type = translateType(node.GetType(), builder);
  auto loc = translateLoc(node.loc(), builder, state);
  if (!loc.ok()) {
    return loc.status();
  }

  return builder.create<xls::ArrayIndexOp>(*loc, result_type, *xls_arg,
                                           *xls_index);
}

absl::StatusOr<Operation*> translateOp(::xls::ArrayConcat& node,
                                       OpBuilder& builder,
                                       TranslationState& state) {
  SmallVector<Value> operands_vec;
  for (auto* xls_operand : node.operands()) {
    auto operand = state.getMlirValue(xls_operand->id());
    if (!operand.ok()) {
      return operand.status();
    }
    operands_vec.push_back(*operand);
  }
  ValueRange operands(operands_vec);

  auto result_type = translateType(node.GetType(), builder);
  auto loc = translateLoc(node.loc(), builder, state);
  if (!loc.ok()) {
    return loc.status();
  }

  return builder.create<xls::ArrayConcatOp>(*loc, result_type, operands);
}

absl::StatusOr<Operation*> translateOp(::xls::ArraySlice& node,
                                       OpBuilder& builder,
                                       TranslationState& state) {
  auto array = state.getMlirValue(
      node.operands()[::xls::ArraySlice::kArrayOperand]->id());
  if (!array.ok()) {
    return array.status();
  }
  auto start = state.getMlirValue(
      node.operands()[::xls::ArraySlice::kStartOperand]->id());
  if (!start.ok()) {
    return start.status();
  }

  auto result_type = translateType(node.GetType(), builder);
  auto loc = translateLoc(node.loc(), builder, state);
  if (!loc.ok()) {
    return loc.status();
  }

  return builder.create<xls::ArraySliceOp>(*loc, result_type, *array, *start,
                                           node.width());
}

absl::StatusOr<Operation*> translateOp(::xls::ArrayUpdate& node,
                                       OpBuilder& builder,
                                       TranslationState& state) {
  auto array_to_update = state.getMlirValue(node.array_to_update()->id());
  if (!array_to_update.ok()) {
    return array_to_update.status();
  }

  if (node.indices().length() != 1) {
    return absl::InternalError(
        "MLIR currently only supports ArrayUpdate with a single index!");
  }
  auto index = state.getMlirValue(node.indices()[0]->id());
  if (!index.ok()) {
    return index.status();
  }

  auto value = state.getMlirValue(node.update_value()->id());
  if (!value.ok()) {
    return value.status();
  }

  auto loc = translateLoc(node.loc(), builder, state);
  if (!loc.ok()) {
    return loc.status();
  }

  return builder.create<xls::ArrayUpdateOp>(*loc, *array_to_update, *value,
                                            *index);
}

absl::StatusOr<Operation*> translateOp(::xls::BitSlice& node,
                                       OpBuilder& builder,
                                       TranslationState& state) {
  auto arg =
      state.getMlirValue(node.operands()[::xls::BitSlice::kArgOperand]->id());
  if (!arg.ok()) {
    return arg.status();
  }

  auto result_type = builder.getIntegerType(node.width());
  auto loc = translateLoc(node.loc(), builder, state);
  if (!loc.ok()) {
    return loc.status();
  }

  return builder.create<xls::BitSliceOp>(*loc, result_type, *arg, node.start(),
                                         node.width());
}

absl::StatusOr<Operation*> translateOp(::xls::BitSliceUpdate& node,
                                       OpBuilder& builder,
                                       TranslationState& state) {
  auto to_update = state.getMlirValue(node.to_update()->id());
  if (!to_update.ok()) {
    return to_update.status();
  }

  auto start = state.getMlirValue(node.start()->id());
  if (!start.ok()) {
    return start.status();
  }

  auto update_value = state.getMlirValue(node.update_value()->id());
  if (!update_value.ok()) {
    return update_value.status();
  }

  auto result_type =
      builder.getIntegerType(node.to_update()->GetType()->GetFlatBitCount());
  auto loc = translateLoc(node.loc(), builder, state);
  if (!loc.ok()) {
    return loc.status();
  }

  return builder.create<xls::BitSliceUpdateOp>(*loc, result_type, *to_update,
                                               *start, *update_value);
}

absl::StatusOr<Operation*> translateOp(::xls::DynamicBitSlice& node,
                                       OpBuilder& builder,
                                       TranslationState& state) {
  auto arg = state.getMlirValue(
      node.operands()[::xls::DynamicBitSlice::kArgOperand]->id());
  if (!arg.ok()) {
    return arg.status();
  }

  auto start = state.getMlirValue(node.start()->id());
  if (!start.ok()) {
    return start.status();
  }

  auto result_type = builder.getIntegerType(node.width());
  auto loc = translateLoc(node.loc(), builder, state);
  if (!loc.ok()) {
    return loc.status();
  }

  return builder.create<xls::DynamicBitSliceOp>(
      *loc, result_type, *arg, *start, builder.getI64IntegerAttr(node.width()));
}

absl::StatusOr<Operation*> translateOp(::xls::Concat& node, OpBuilder& builder,
                                       TranslationState& state) {
  SmallVector<Value> operands_vec;
  uint64_t result_width = 0;
  for (auto* xls_operand : node.operands()) {
    auto operand = state.getMlirValue(xls_operand->id());
    if (!operand.ok()) {
      return operand.status();
    }
    operands_vec.push_back(*operand);
    result_width += xls_operand->GetType()->GetFlatBitCount();
  }
  ValueRange operands(operands_vec);

  auto result_type = builder.getIntegerType(result_width);
  auto loc = translateLoc(node.loc(), builder, state);
  if (!loc.ok()) {
    return loc.status();
  }

  return builder.create<xls::ConcatOp>(*loc, result_type, operands);
}

absl::StatusOr<Operation*> translateOp(::xls::Tuple& node, OpBuilder& builder,
                                       TranslationState& state) {
  SmallVector<Value> operands_vec;
  for (auto* xls_operand : node.operands()) {
    auto operand = state.getMlirValue(xls_operand->id());
    if (!operand.ok()) {
      return operand.status();
    }
    operands_vec.push_back(*operand);
  }
  ValueRange operands(operands_vec);

  auto loc = translateLoc(node.loc(), builder, state);
  if (!loc.ok()) {
    return loc.status();
  }

  return builder.create<xls::TupleOp>(*loc, operands);
}

absl::StatusOr<Operation*> translateOp(::xls::Literal& node, OpBuilder& builder,
                                       TranslationState& state) {
  auto loc = translateLoc(node.loc(), builder, state);
  if (!loc.ok()) {
    return loc.status();
  }

  switch (node.GetType()->kind()) {
    case ::xls::TypeKind::kBits:
      return translateBitsLiteral(node.value().bits(), *loc, builder, state);
    default: {
      auto result_type = translateType(node.GetType(), builder);

      // Create literal with region:
      auto literal_op = builder.create<xls::LiteralOp>(*loc, result_type);
      builder.setInsertionPointToStart(&literal_op.getRegion().emplaceBlock());

      // Construct value using ops:
      auto final_op = translateLiteral(node.value(), builder, state);
      if (!final_op.ok()) {
        return final_op.status();
      }

      // Add yield:
      builder.create<YieldOp>(builder.getUnknownLoc(),
                              final_op.value()->getResults());

      return literal_op;
    }
  }
}

absl::StatusOr<Operation*> translateOp(::xls::NaryOp& node, OpBuilder& builder,
                                       TranslationState& state) {
  SmallVector<Value> operands_vec;
  for (auto* xls_operand : node.operands()) {
    auto operand = state.getMlirValue(xls_operand->id());
    if (!operand.ok()) {
      return operand.status();
    }
    operands_vec.push_back(*operand);
  }
  ValueRange operands(operands_vec);

  auto loc = translateLoc(node.loc(), builder, state);
  if (!loc.ok()) {
    return loc.status();
  }

  switch (node.op()) {
    case ::xls::Op::kAnd:
      return builder.create<xls::AndOp>(*loc, operands);
    case ::xls::Op::kNand:
      return builder.create<xls::NandOp>(*loc, operands);
    case ::xls::Op::kOr:
      return builder.create<xls::OrOp>(*loc, operands);
    case ::xls::Op::kNor:
      return builder.create<xls::NorOp>(*loc, operands);
    case ::xls::Op::kXor:
      return builder.create<xls::XorOp>(*loc, operands);
    default:
      return absl::InternalError(absl::StrCat("Expected BinOp operation, not ",
                                              ::xls::OpToString(node.op())));
  }
}

absl::StatusOr<Operation*> translateOp(::xls::BitwiseReductionOp& node,
                                       OpBuilder& builder,
                                       TranslationState& state) {
  auto operand = state.getMlirValue(
      node.operands()[::xls::BitwiseReductionOp::kOperandOperand]->id());
  if (!operand.ok()) {
    return operand.status();
  }

  auto result_type = builder.getIntegerType(1);
  auto loc = translateLoc(node.loc(), builder, state);
  if (!loc.ok()) {
    return loc.status();
  }

  switch (node.op()) {
    case ::xls::Op::kAndReduce:
      return builder.create<xls::AndReductionOp>(*loc, result_type, *operand);
    case ::xls::Op::kOrReduce:
      return builder.create<xls::OrReductionOp>(*loc, result_type, *operand);
    case ::xls::Op::kXorReduce:
      return builder.create<xls::XorReductionOp>(*loc, result_type, *operand);
    default:
      return absl::InternalError(
          absl::StrCat("Expected BitwiseReductionOp operation, not ",
                       ::xls::OpToString(node.op())));
  }
}

absl::StatusOr<Operation*> translateOp(::xls::Encode& node, OpBuilder& builder,
                                       TranslationState& state) {
  auto arg =
      state.getMlirValue(node.operands()[::xls::Encode::kArgOperand]->id());
  if (!arg.ok()) {
    return arg.status();
  }

  auto result_type = builder.getIntegerType(node.GetType()->GetFlatBitCount());
  auto loc = translateLoc(node.loc(), builder, state);
  if (!loc.ok()) {
    return loc.status();
  }

  return builder.create<xls::EncodeOp>(*loc, result_type, *arg);
}

absl::StatusOr<Operation*> translateOp(::xls::Decode& node, OpBuilder& builder,
                                       TranslationState& state) {
  auto arg =
      state.getMlirValue(node.operands()[::xls::Decode::kArgOperand]->id());
  if (!arg.ok()) {
    return arg.status();
  }

  auto width = builder.getI64IntegerAttr(node.GetType()->GetFlatBitCount());

  auto result_type = builder.getIntegerType(node.GetType()->GetFlatBitCount());
  auto loc = translateLoc(node.loc(), builder, state);
  if (!loc.ok()) {
    return loc.status();
  }

  return builder.create<xls::DecodeOp>(*loc, result_type, *arg, width);
}

absl::StatusOr<Operation*> translateOp(::xls::OneHot& node, OpBuilder& builder,
                                       TranslationState& state) {
  auto arg =
      state.getMlirValue(node.operands()[::xls::OneHot::kInputOperand]->id());
  if (!arg.ok()) {
    return arg.status();
  }

  auto lsb_prio = builder.getBoolAttr(node.priority() == ::xls::LsbOrMsb::kLsb);

  auto result_type = builder.getIntegerType(node.GetType()->GetFlatBitCount());
  auto loc = translateLoc(node.loc(), builder, state);
  if (!loc.ok()) {
    return loc.status();
  }

  return builder.create<xls::OneHotOp>(*loc, result_type, *arg, lsb_prio);
}

absl::StatusOr<Operation*> translateOp(::xls::OneHotSelect& node,
                                       OpBuilder& builder,
                                       TranslationState& state) {
  auto selector = state.getMlirValue(
      node.operands()[::xls::OneHotSelect::kSelectorOperand]->id());
  if (!selector.ok()) {
    return selector.status();
  }

  SmallVector<Value> cases_vec;
  for (auto* xls_case : node.cases()) {
    auto operand = state.getMlirValue(xls_case->id());
    if (!operand.ok()) {
      return operand.status();
    }
    cases_vec.push_back(*operand);
  }
  ValueRange cases(cases_vec);

  auto result_type = builder.getIntegerType(node.GetType()->GetFlatBitCount());
  auto loc = translateLoc(node.loc(), builder, state);
  if (!loc.ok()) {
    return loc.status();
  }

  return builder.create<xls::OneHotSelOp>(*loc, result_type, *selector, cases);
}

absl::StatusOr<Operation*> translateOp(::xls::Invoke& node, OpBuilder& builder,
                                       TranslationState& state) {
  auto fn = state.getFunction(node.to_apply()->name());
  if (!fn.ok()) {
    return fn.status();
  }

  SmallVector<Value> operands_vec;
  for (auto* xls_operand : node.operands()) {
    auto operand = state.getMlirValue(xls_operand->id());
    if (!operand.ok()) {
      return operand.status();
    }
    operands_vec.push_back(*operand);
  }
  ValueRange operands(operands_vec);

  auto result_type = translateType(node.GetType(), builder);
  auto loc = translateLoc(node.loc(), builder, state);
  if (!loc.ok()) {
    return loc.status();
  }

  return builder.create<func::CallOp>(*loc, *fn, result_type, operands);
}

absl::StatusOr<Operation*> translateOp(::xls::Map& node, OpBuilder& builder,
                                       TranslationState& state) {
  auto fn = state.getFunction(node.to_apply()->name());
  if (!fn.ok()) {
    return fn.status();
  }

  auto array =
      state.getMlirValue(node.operands()[::xls::Map::kArgOperand]->id());
  if (!array.ok()) {
    return array.status();
  }

  auto result_type = translateType(node.GetType(), builder);
  auto loc = translateLoc(node.loc(), builder, state);
  if (!loc.ok()) {
    return loc.status();
  }

  return builder.create<xls::MapOp>(*loc, result_type, *array, *fn);
}

absl::StatusOr<Operation*> translateOp(::xls::Select& node, OpBuilder& builder,
                                       TranslationState& state) {
  auto selector = state.getMlirValue(node.selector()->id());
  if (!selector.ok()) {
    return selector.status();
  }

  SmallVector<Value> cases_vec;
  for (auto* xls_case : node.cases()) {
    auto operand = state.getMlirValue(xls_case->id());
    if (!operand.ok()) {
      return operand.status();
    }
    cases_vec.push_back(*operand);
  }
  ValueRange cases(cases_vec);

  Value default_value;

  if (auto xls_default_val = node.default_value();
      xls_default_val.has_value()) {
    auto maybe_default_val =
        state.getMlirValue(node.default_value().value()->id());
    if (!maybe_default_val.ok()) {
      return maybe_default_val.status();
    }
    default_value = maybe_default_val.value();
  }

  auto result_type = translateType(node.GetType(), builder);
  auto loc = translateLoc(node.loc(), builder, state);
  if (!loc.ok()) {
    return loc.status();
  }

  return builder.create<xls::SelOp>(*loc, result_type, *selector, default_value,
                                    cases);
}

absl::StatusOr<Operation*> translateOp(::xls::PrioritySelect& node,
                                       OpBuilder& builder,
                                       TranslationState& state) {
  auto selector = state.getMlirValue(
      node.operands()[::xls::PrioritySelect::kSelectorOperand]->id());
  if (!selector.ok()) {
    return selector.status();
  }

  SmallVector<Value> cases_vec;
  for (auto* xls_case : node.cases()) {
    auto operand = state.getMlirValue(xls_case->id());
    if (!operand.ok()) {
      return operand.status();
    }
    cases_vec.push_back(*operand);
  }
  ValueRange cases(cases_vec);

  auto default_value = state.getMlirValue(node.default_value()->id());
  if (!default_value.ok()) {
    return default_value.status();
  }

  auto result_type = translateType(node.GetType(), builder);
  auto loc = translateLoc(node.loc(), builder, state);
  if (!loc.ok()) {
    return loc.status();
  }

  return builder.create<xls::PrioritySelOp>(*loc, result_type, *selector, cases,
                                            *default_value);
}

absl::StatusOr<Operation*> translateOp(::xls::AfterAll& node,
                                       OpBuilder& builder,
                                       TranslationState& state) {
  SmallVector<Value> operands_vec;
  for (auto* xls_operand : node.operands()) {
    auto operand = state.getMlirValue(xls_operand->id());
    if (!operand.ok()) {
      return operand.status();
    }
    operands_vec.push_back(*operand);
  }
  ValueRange operands(operands_vec);

  auto loc = translateLoc(node.loc(), builder, state);
  if (!loc.ok()) {
    return loc.status();
  }

  return builder.create<xls::AfterAllOp>(*loc, operands);
}

absl::StatusOr<Operation*> translateOp(::xls::ChannelNode& node,
                                       OpBuilder& builder,
                                       TranslationState& state) {
  MLIRContext* ctx = builder.getContext();
  auto chn = state.getChannel(node.channel_name());
  if (!chn.ok()) {
    return chn.status();
  }

  auto inp_token = state.getMlirValue(node.token()->id());
  if (!inp_token.ok()) {
    return inp_token.status();
  }

  Value predicate;
  if (auto xls_predicate = node.predicate(); xls_predicate.has_value()) {
    auto maybe_predicate = state.getMlirValue(xls_predicate.value()->id());
    if (!maybe_predicate.ok()) {
      return maybe_predicate.status();
    }
    predicate = *maybe_predicate;
  }

  auto token_type = TokenType::get(ctx);
  auto data_type = translateType(node.GetPayloadType(), builder);
  auto valid_type = builder.getIntegerType(1);

  auto loc = translateLoc(node.loc(), builder, state);
  if (!loc.ok()) {
    return loc.status();
  }

  if (node.Is<::xls::Receive>()) {
    auto recv = node.As<::xls::Receive>();

    SmallVector<Value> result_elems;

    if (recv->is_blocking()) {
      auto receive_op = builder.create<xls::BlockingReceiveOp>(
          *loc, token_type, data_type, *inp_token, predicate, *chn);

      result_elems.push_back(receive_op.getTknOut());
      result_elems.push_back(receive_op.getResult());

    } else {
      auto receive_op = builder.create<xls::NonblockingReceiveOp>(
          *loc, token_type, data_type, valid_type, *inp_token, predicate, *chn);

      result_elems.push_back(receive_op.getTknOut());
      result_elems.push_back(receive_op.getResult());
      result_elems.push_back(receive_op.getValid());
    }

    // In XLS, receive returns a tuple. In MLIR, receive returns separate return
    // values. We add a TupleOp that fuses the elements together, because
    // the operation using the result expects a tuple.
    ValueRange operands(result_elems);
    return builder.create<xls::TupleOp>(*loc, operands);

  } else {
    auto send = node.As<::xls::Send>();

    auto data = state.getMlirValue(send->data()->id());
    if (!data.ok()) {
      return data.status();
    }

    return builder.create<xls::SendOp>(*loc, token_type, *inp_token, *data,
                                       predicate, *chn);
  }
}

absl::StatusOr<Operation*> translateOp(::xls::Gate& node, OpBuilder& builder,
                                       TranslationState& state) {
  auto data = state.getMlirValue(node.data()->id());
  if (!data.ok()) {
    return data.status();
  }

  auto condition = state.getMlirValue(node.condition()->id());
  if (!condition.ok()) {
    return condition.status();
  }

  auto loc = translateLoc(node.loc(), builder, state);
  if (!loc.ok()) {
    return loc.status();
  }

  return builder.create<xls::GateOp>(*loc, *condition, *data);
}

absl::StatusOr<Operation*> translateOp(::xls::CountedFor& node,
                                       OpBuilder& builder,
                                       TranslationState& state) {
  auto initial_value = state.getMlirValue(node.initial_value()->id());
  if (!initial_value.ok()) {
    return initial_value.status();
  }

  auto trip_count = builder.getI64IntegerAttr(node.trip_count());

  auto stride = builder.getI64IntegerAttr(node.stride());

  SmallVector<Value> invar_args_vec;
  for (auto* xls_operand : node.invariant_args()) {
    auto invar_arg = state.getMlirValue(xls_operand->id());
    if (!invar_arg.ok()) {
      return invar_arg.status();
    }
    invar_args_vec.push_back(*invar_arg);
  }
  ValueRange invar_args(invar_args_vec);

  auto body = state.getFunction(node.body()->name());
  if (!body.ok()) {
    return body.status();
  }

  auto result_type = translateType(node.GetType(), builder);
  auto loc = translateLoc(node.loc(), builder, state);
  if (!loc.ok()) {
    return loc.status();
  }

  return builder.create<xls::CountedForOp>(
      *loc, result_type, *initial_value, invar_args, trip_count, *body, stride);
}

absl::StatusOr<Operation*> translateOp(::xls::Trace& node, OpBuilder& builder,
                                       TranslationState& state) {
  auto token = state.getMlirValue(node.token()->id());
  if (!token.ok()) {
    return token.status();
  }

  auto predicate = state.getMlirValue(node.condition()->id());
  if (!predicate.ok()) {
    return predicate.status();
  }

  SmallVector<Value> args_vec;
  for (auto* xls_operand : node.args()) {
    auto arg = state.getMlirValue(xls_operand->id());
    if (!arg.ok()) {
      return arg.status();
    }
    args_vec.push_back(*arg);
  }
  ValueRange args(args_vec);

  auto result_type = translateType(node.GetType(), builder);
  auto loc = translateLoc(node.loc(), builder, state);
  if (!loc.ok()) {
    return loc.status();
  }

  return builder.create<xls::TraceOp>(
      *loc, result_type, *token, *predicate, args,
      builder.getStringAttr(::xls::StepsToXlsFormatString(node.format())),
      builder.getI64IntegerAttr(node.verbosity()));
}

absl::StatusOr<Operation*> translateAnyOp(::xls::Node& xls_node,
                                          OpBuilder& builder,
                                          TranslationState& state) {
  absl::StatusOr<Operation*> op;

  if (auto* xls_op = dynamic_cast<::xls::Literal*>(&xls_node)) {
    op = translateOp(*xls_op, builder, state);
  } else if (auto* xls_op = dynamic_cast<::xls::BinOp*>(&xls_node)) {
    op = translateOp(*xls_op, builder, state);
  } else if (auto* xls_op = dynamic_cast<::xls::ArithOp*>(&xls_node)) {
    op = translateOp(*xls_op, builder, state);
  } else if (auto* xls_op = dynamic_cast<::xls::PartialProductOp*>(&xls_node)) {
    op = translateOp(*xls_op, builder, state);
  } else if (auto* xls_op = dynamic_cast<::xls::UnOp*>(&xls_node)) {
    op = translateOp(*xls_op, builder, state);
  } else if (auto* xls_op = dynamic_cast<::xls::CompareOp*>(&xls_node)) {
    op = translateOp(*xls_op, builder, state);
  } else if (auto* xls_op = dynamic_cast<::xls::NaryOp*>(&xls_node)) {
    op = translateOp(*xls_op, builder, state);
  } else if (auto* xls_op =
                 dynamic_cast<::xls::BitwiseReductionOp*>(&xls_node)) {
    op = translateOp(*xls_op, builder, state);
  } else if (auto* xls_op = dynamic_cast<::xls::ExtendOp*>(&xls_node)) {
    op = translateOp(*xls_op, builder, state);
  } else if (auto* xls_op = dynamic_cast<::xls::Tuple*>(&xls_node)) {
    op = translateOp(*xls_op, builder, state);
  } else if (auto* xls_op = dynamic_cast<::xls::TupleIndex*>(&xls_node)) {
    op = translateOp(*xls_op, builder, state);
  } else if (auto* xls_op = dynamic_cast<::xls::Array*>(&xls_node)) {
    op = translateOp(*xls_op, builder, state);
  } else if (auto* xls_op = dynamic_cast<::xls::ArrayIndex*>(&xls_node)) {
    op = translateOp(*xls_op, builder, state);
  } else if (auto* xls_op = dynamic_cast<::xls::ArrayConcat*>(&xls_node)) {
    op = translateOp(*xls_op, builder, state);
  } else if (auto* xls_op = dynamic_cast<::xls::ArraySlice*>(&xls_node)) {
    op = translateOp(*xls_op, builder, state);
  } else if (auto* xls_op = dynamic_cast<::xls::ArrayUpdate*>(&xls_node)) {
    op = translateOp(*xls_op, builder, state);
  } else if (auto* xls_op = dynamic_cast<::xls::BitSlice*>(&xls_node)) {
    op = translateOp(*xls_op, builder, state);
  } else if (auto* xls_op = dynamic_cast<::xls::BitSliceUpdate*>(&xls_node)) {
    op = translateOp(*xls_op, builder, state);
  } else if (auto* xls_op = dynamic_cast<::xls::DynamicBitSlice*>(&xls_node)) {
    op = translateOp(*xls_op, builder, state);
  } else if (auto* xls_op = dynamic_cast<::xls::Concat*>(&xls_node)) {
    op = translateOp(*xls_op, builder, state);
  } else if (auto* xls_op = dynamic_cast<::xls::Encode*>(&xls_node)) {
    op = translateOp(*xls_op, builder, state);
  } else if (auto* xls_op = dynamic_cast<::xls::Decode*>(&xls_node)) {
    op = translateOp(*xls_op, builder, state);
  } else if (auto* xls_op = dynamic_cast<::xls::OneHot*>(&xls_node)) {
    op = translateOp(*xls_op, builder, state);
  } else if (auto* xls_op = dynamic_cast<::xls::OneHotSelect*>(&xls_node)) {
    op = translateOp(*xls_op, builder, state);
  } else if (auto* xls_op = dynamic_cast<::xls::Invoke*>(&xls_node)) {
    op = translateOp(*xls_op, builder, state);
  } else if (auto* xls_op = dynamic_cast<::xls::Select*>(&xls_node)) {
    op = translateOp(*xls_op, builder, state);
  } else if (auto* xls_op = dynamic_cast<::xls::PrioritySelect*>(&xls_node)) {
    op = translateOp(*xls_op, builder, state);
  } else if (auto* xls_op = dynamic_cast<::xls::Map*>(&xls_node)) {
    op = translateOp(*xls_op, builder, state);
  } else if (auto* xls_op = dynamic_cast<::xls::AfterAll*>(&xls_node)) {
    op = translateOp(*xls_op, builder, state);
  } else if (auto* xls_op = dynamic_cast<::xls::ChannelNode*>(&xls_node)) {
    op = translateOp(*xls_op, builder, state);
  } else if (auto* xls_op = dynamic_cast<::xls::Gate*>(&xls_node)) {
    op = translateOp(*xls_op, builder, state);
  } else if (auto* xls_op = dynamic_cast<::xls::CountedFor*>(&xls_node)) {
    op = translateOp(*xls_op, builder, state);
  } else if (auto* xls_op = dynamic_cast<::xls::Trace*>(&xls_node)) {
    op = translateOp(*xls_op, builder, state);
  } else if (dynamic_cast<::xls::Param*>(&xls_node)) {
    return absl::InternalError(
        "Param not handeled during function translation.");
  } else if (dynamic_cast<::xls::StateRead*>(&xls_node)) {
    return absl::InternalError(
        "StateRead not handeled during proc translation.");
  } else if (dynamic_cast<::xls::Next*>(&xls_node)) {
    return absl::InternalError("Next not handeled during proc translation.");
  } else if (dynamic_cast<::xls::Cover*>(&xls_node) ||
             dynamic_cast<::xls::Assert*>(&xls_node) ||
             dynamic_cast<::xls::MinDelay*>(&xls_node) ||
             dynamic_cast<::xls::RegisterRead*>(&xls_node) ||
             dynamic_cast<::xls::RegisterWrite*>(&xls_node)) {
    return absl::InternalError(absl::StrCat(
        "Unsupported operation: ", ::xls::OpToString(xls_node.op()),
        " - Not yet available in XLS MLIR!"));
  } else {
    return absl::InternalError(absl::StrCat("Unsupported operation: ",
                                            ::xls::OpToString(xls_node.op())));
  }

  if (!op.ok()) {
    return op.status();
  }

  if (auto err = state.setMlirValue(xls_node.id(), (*op)->getResult(0));
      !err.ok()) {
    return err;
  }

  return *op;
}

//===----------------------------------------------------------------------===//
// Function Translation
//===----------------------------------------------------------------------===//

absl::StatusOr<Operation*> translateFunction(::xls::Function& xls_func,
                                             OpBuilder& builder,
                                             TranslationState& state) {
  MLIRContext* ctx = builder.getContext();
  // Argument types + names:
  SmallVector<Type> mlir_arg_types;
  SmallVector<Location> mlir_arg_locs;
  for (auto* arg : xls_func.params()) {
    mlir_arg_types.push_back(translateType(arg->GetType(), builder));
    mlir_arg_locs.push_back(NameLoc::get(StringAttr::get(ctx, arg->name())));
  }

  // Return type:
  auto return_type = translateType(xls_func.GetType()->return_type(), builder);

  // Create Function:
  auto funcType =
      FunctionType::get(ctx, TypeRange(mlir_arg_types), {return_type});
  auto func =
      func::FuncOp::create(builder.getUnknownLoc(), xls_func.name(), funcType);

  builder.insert(func);

  // Add function to package context:
  if (auto err = state.setFunction(xls_func.name(),
                                   SymbolRefAttr::get(func.getNameAttr()));
      !err.ok()) {
    return err;
  }

  // Create function body:
  auto* body = func.addEntryBlock();

  // New function base scope
  state.newFunctionBaseContext(/*is_proc=*/false);

  // Add parameters to function context + attach parameter locs
  for (uint64_t arg_idx = 0; arg_idx < xls_func.params().length(); arg_idx++) {
    auto xls_param = xls_func.params()[arg_idx];
    auto mlir_arg = body->getArgument(arg_idx);
    if (auto err = state.setMlirValue(xls_param->id(), mlir_arg); !err.ok()) {
      return err;
    }
    mlir_arg.setLoc(mlir_arg_locs[arg_idx]);
  }

  // Function body:
  Value return_value;
  for (auto* n : xls_func.nodes()) {
    builder.setInsertionPointToEnd(body);
    if (n->Is<::xls::Param>()) {
      // Params have already been converted and added to func context.
      if (xls_func.return_value() == n) {
        auto maybe_return_value =
            state.getMlirValue(n->As<::xls::Param>()->id());
        if (!maybe_return_value.ok()) {
          return maybe_return_value.status();
        }
        return_value = *maybe_return_value;
      }

      continue;
    }

    auto op = translateAnyOp(*n, builder, state);
    if (!op.ok()) {
      return op;
    }

    if (xls_func.return_value() == n) {
      assert(op.value()->getResults().size() == 1 &&
             "Returning op with multiple result.");
      return_value = op.value()->getResult(0);
    }
  }

  if (return_value == Value()) {
    assert(return_value != Value() && "Function return op not translated.");
  }

  // Function body terminator (return):
  builder.setInsertionPointToEnd(body);
  builder.create<func::ReturnOp>(builder.getUnknownLoc(),
                                 ValueRange(return_value));

  return func;
}

//===----------------------------------------------------------------------===//
// Proc Translation
//===----------------------------------------------------------------------===//

absl::StatusOr<Operation*> translateProc(::xls::Proc& xls_proc,
                                         OpBuilder& builder,
                                         TranslationState& state) {
  MLIRContext* ctx = builder.getContext();
  // New function base scope
  state.newFunctionBaseContext(/*is_proc=*/true);

  // Create Eproc:
  EprocOp eproc =
      builder.create<EprocOp>(builder.getUnknownLoc(),
                              /*name=*/builder.getStringAttr(xls_proc.name()),
                              /*discardable=*/false);

  auto* body = &eproc.getRegion().emplaceBlock();

  // State types and initial value:
  for (int64_t i = 0; i < xls_proc.StateElements().size(); i++) {
    auto xls_elem = xls_proc.StateElements()[i];
    auto elem_type = translateType(xls_elem->type(), builder);
    // TODO(schilkp): Initial state!
    body->addArgument(elem_type, builder.getUnknownLoc());
    auto mlir_elem = body->getArgument(i);
    state.setStateElement(xls_elem->name(), mlir_elem);
    mlir_elem.setLoc(NameLoc::get(StringAttr::get(ctx, xls_elem->name())));
  }

  // For each state element, track the SSA value that defines its next value.
  absl::flat_hash_map<std::string, Value> state_elem_next_value{};

  // For each state element, track all `next_value` nodes that contribute
  // to it. They will be jointly converted in one NextValueOp per state element.
  absl::flat_hash_map<std::string, std::vector<::xls::Next*>> next_value_ops{};

  // Proc next:
  for (auto n : xls_proc.nodes()) {
    builder.setInsertionPointToEnd(body);

    // StateRead nodes give state elements an SSA identifier:
    if (auto state_read = dynamic_cast<::xls::StateRead*>(n)) {
      auto state_value =
          state.getStateElement(state_read->state_element()->name());
      if (!state_value.ok()) {
        return state_value.status();
      }
      auto err = state.setMlirValue(state_read->id(), *state_value);
      if (!err.ok()) {
        return err;
      }
      continue;  // StateRead get directly added to the ssa context and don't
                 // otherwise map to a MLIR operation.
    }

    if (n->Is<::xls::Next>()) {
      auto xls_next = n->As<::xls::Next>();
      auto state_elem_name = xls_next->state_read()
                                 ->As<::xls::StateRead>()
                                 ->state_element()
                                 ->name();
      next_value_ops[state_elem_name].push_back(xls_next);
      continue;
    }
    auto op = translateAnyOp(*n, builder, state);
    if (!op.ok()) {
      return op;
    }
  }

  // Generate NextValueOps:
  for (auto* state_elem : xls_proc.StateElements()) {
    // All Next nodes that contribute to this state element:
    std::vector<::xls::Next*> elem_next_value_ops =
        next_value_ops[state_elem->name()];

    if (elem_next_value_ops.empty()) {
      return absl::InternalError(absl::StrCat(
          "No `next_value` nodes for state element ", state_elem->name()));
    }

    if (elem_next_value_ops.size() == 1 &&
        !elem_next_value_ops[0]->predicate().has_value()) {
      // State element defined by single Next node without predicate. No MLIR
      // NextValueOp needed - directly feed YieldOp from value:
      auto redundant_next_node = elem_next_value_ops[0];
      auto next_value = state.getMlirValue(redundant_next_node->value()->id());
      if (!next_value.ok()) {
        return next_value.status();
      }
      state_elem_next_value[state_elem->name()] = *next_value;
    } else {
      // Generate new NextValueOp:
      SmallVector<Value> values_vec{};
      SmallVector<Value> predicates_vec{};

      for (auto* next_value_op : elem_next_value_ops) {
        if (!next_value_op->predicate().has_value()) {
          return absl::InternalError(absl::StrCat(
              "`next_value` nodes for state element ", state_elem->name(),
              " lacks predicate but there are multiple `next_value` nodes."));
        }

        auto val = state.getMlirValue(next_value_op->value()->id());
        if (!val.ok()) {
          return val.status();
        }
        values_vec.push_back(*val);

        auto predicate =
            state.getMlirValue(next_value_op->predicate().value()->id());
        if (!predicate.ok()) {
          return val.status();
        }
        predicates_vec.push_back(*predicate);
      }

      auto elem_type = translateType(state_elem->type(), builder);

      builder.setInsertionPointToEnd(body);
      auto next = builder.create<NextValueOp>(
          builder.getUnknownLoc(), elem_type, ValueRange(predicates_vec),
          ValueRange(values_vec));

      state_elem_next_value[state_elem->name()] = next;
    }
  }

  // Generate terminating yield op:
  SmallVector<Value> next_value_elems{};
  for (auto* state_elem : xls_proc.StateElements()) {
    next_value_elems.push_back(state_elem_next_value[state_elem->name()]);
  }
  builder.setInsertionPointToEnd(body);
  builder.create<YieldOp>(builder.getUnknownLoc(),
                          ValueRange(next_value_elems));

  return eproc;
}

//===----------------------------------------------------------------------===//
// Channel Translation
//===----------------------------------------------------------------------===//

FlopKind convertFlopKind(::xls::FlopKind kind) {
  switch (kind) {
    case ::xls::FlopKind::kNone:
      return FlopKind::kNone;
    case ::xls::FlopKind::kFlop:
      return FlopKind::kFlop;
    case ::xls::FlopKind::kSkid:
      return FlopKind::kSkid;
    case ::xls::FlopKind::kZeroLatency:
      return FlopKind::kZeroLatency;
  }
}

absl::Status translateChannel(::xls::Channel& xls_chn, OpBuilder& builder,
                              TranslationState& state) {
  MLIRContext* ctx = builder.getContext();
  auto chn = builder.create<xls::ChanOp>(
      builder.getUnknownLoc(),
      /*name=*/builder.getStringAttr(xls_chn.name()),
      /*type=*/TypeAttr::get(translateType(xls_chn.type(), builder)),
      /*fifo_config=*/nullptr,
      /*input_flop_kind=*/nullptr,
      /*output_flop_kind=*/nullptr,
      /*send_supported=*/builder.getBoolAttr(xls_chn.CanSend()),
      /*recv_supported=*/builder.getBoolAttr(xls_chn.CanReceive()));

  if (auto xls_streaming_chn =
          dynamic_cast<::xls::StreamingChannel*>(&xls_chn)) {
    auto xls_ch_config = xls_streaming_chn->channel_config();
    if (auto xls_fifo_config = xls_ch_config.fifo_config()) {
      chn.setFifoConfigAttr(FifoConfig::get(
          ctx, xls_fifo_config->depth(), xls_fifo_config->bypass(),
          xls_fifo_config->register_push_outputs(),
          xls_fifo_config->register_pop_outputs()));
    }

    if (auto xls_input_flop = xls_ch_config.input_flop_kind()) {
      chn.setInputFlopKind(convertFlopKind(*xls_input_flop));
    }

    if (auto xls_output_flop = xls_ch_config.output_flop_kind()) {
      chn.setOutputFlopKind(convertFlopKind(*xls_output_flop));
    }
  }

  return state.setChannel(std::string(xls_chn.name()),
                          SymbolRefAttr::get(chn.getNameAttr()));
}

//===----------------------------------------------------------------------===//
// Package Translation
//===----------------------------------------------------------------------===//

absl::Status translatePackage(::xls::Package& xls_pkg, OpBuilder& builder,
                              ModuleOp module) {
  TranslationState state;

  // Translate file numbers:
  auto xls_fileno_map = xls_pkg.fileno_to_name();
  for (auto& [fileno, name] : xls_fileno_map) {
    auto name_attr = builder.getStringAttr(name);
    if (auto err = state.setFileName(fileno, name_attr); !err.ok()) {
      return err;
    }
  }

  // Translate all channels:
  for (auto* c : xls_pkg.channels()) {
    builder.setInsertionPointToEnd(module.getBody());
    if (auto err = translateChannel(*c, builder, state); !err.ok()) {
      return err;
    }
  }

  // Translate all functions:
  for (const auto& f : xls_pkg.functions()) {
    builder.setInsertionPointToEnd(module.getBody());
    auto func = translateFunction(*f, builder, state);
    if (!func.ok()) {
      return func.status();
    }
  }

  // Translate all procs:
  for (const auto& p : xls_pkg.procs()) {
    builder.setInsertionPointToEnd(module.getBody());
    auto proc = translateProc(*p, builder, state);
    if (!proc.ok()) {
      return proc.status();
    }
  }

  return absl::OkStatus();
}

OwningOpRef<Operation*> XlsToMlirXlsTranslate(llvm::SourceMgr& mgr,
                                              MLIRContext* ctx) {
  OpBuilder builder(ctx);

  // Load XLS dialect we will be emitting:
  ctx->loadDialect<XlsDialect>();

  // New top module to hold generated MLIR:
  const llvm::MemoryBuffer* buf = mgr.getMemoryBuffer(mgr.getMainFileID());
  auto loc = FileLineColLoc::get(
      StringAttr::get(ctx, buf->getBufferIdentifier()), /*line=*/0,
      /*column=*/0);
  OwningOpRef<ModuleOp> module(ModuleOp::create(loc));

  // Parse XLS IR:
  absl::StatusOr<std::unique_ptr<::xls::Package>> package =
      ::xls::ParsePackage(buf->getBuffer().str(), std::nullopt);
  if (!package.ok()) {
    llvm::errs() << "Failed to parse: " << package.status().message() << "\n";
    return {};
  }

  // Translate package from XLS IR to MLIR:
  if (auto err = translatePackage(**package, builder, module.get());
      !err.ok()) {
    llvm::errs() << err.message() << "\n";
    return {};
  }

  return module;
}

}  // namespace mlir::xls
