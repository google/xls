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
#include "xls/dslx/bytecode_emitter.h"

#include "xls/dslx/ast.h"

namespace xls::dslx {

BytecodeEmitter::BytecodeEmitter(
    ImportData* import_data, TypeInfo* type_info,
    absl::flat_hash_map<const NameDef*, int64_t>* namedef_to_slot)
    : import_data_(import_data),
      type_info_(type_info),
      namedef_to_slot_(namedef_to_slot) {}

BytecodeEmitter::~BytecodeEmitter() = default;

absl::StatusOr<std::vector<Bytecode>> BytecodeEmitter::Emit(Function* f) {
  status_ = absl::OkStatus();
  bytecode_.clear();
  f->body()->AcceptExpr(this);
  if (!status_.ok()) {
    return status_;
  }

  return bytecode_;
}

void BytecodeEmitter::HandleBinop(Binop* node) {
  if (!status_.ok()) {
    return;
  }

  node->lhs()->AcceptExpr(this);
  node->rhs()->AcceptExpr(this);
  switch (node->binop_kind()) {
    case BinopKind::kAdd:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kAdd));
      return;
    default:
      status_ = absl::UnimplementedError("foo");
  }
}

void BytecodeEmitter::HandleInvocation(Invocation* node) {
  if (!status_.ok()) {
    return;
  }

  for (auto* arg : node->args()) {
    arg->AcceptExpr(this);
  }

  InterpValue callee_value(InterpValue::MakeUnit());
  if (NameRef* name_ref = dynamic_cast<NameRef*>(node->callee());
      name_ref != nullptr) {
    absl::StatusOr<InterpValue> status_or_callee =
        import_data_->GetOrCreateTopLevelBindings(node->owner())
            .ResolveValue(name_ref);
    if (!status_or_callee.ok()) {
      status_ = absl::InternalError(
          absl::StrCat("Unable to get value for invocation callee: ",
                       node->callee()->ToString()));
      return;
    }
    bytecode_.push_back(
        Bytecode(node->span(), Bytecode::Op::kCall, status_or_callee.value()));
    return;
  }

  // Else it's a ColonRef.
  status_ =
      absl::UnimplementedError("ColonRef invocations not yet implemented.");
}

void BytecodeEmitter::HandleLet(Let* node) {
  node->rhs()->AcceptExpr(this);
  if (!status_.ok()) {
    return;
  }

  // TODO(rspringer): Handle destructuring let.
  NameDef* name_def = node->name_def_tree()->GetNameDefs()[0];
  if (!namedef_to_slot_->contains(name_def)) {
    namedef_to_slot_->insert({name_def, namedef_to_slot_->size()});
  }
  int64_t slot = namedef_to_slot_->at(name_def);

  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kStore, slot));
  node->body()->AcceptExpr(this);
}

void BytecodeEmitter::HandleNameRef(NameRef* node) {
  if (absl::holds_alternative<BuiltinNameDef*>(node->name_def())) {
    status_ =
        absl::UnimplementedError("NameRefs to builtins are not yet supported.");
  }
  int64_t slot = namedef_to_slot_->at(absl::get<NameDef*>(node->name_def()));
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kLoad, slot));
}

void BytecodeEmitter::HandleNumber(Number* node) {
  absl::optional<ConcreteType*> type_or = type_info_->GetItem(node);
  BitsType* bits_type = dynamic_cast<BitsType*>(type_or.value());
  if (bits_type == nullptr) {
    status_ = absl::InternalError(
        "Error in type deduction; number did not have \"bits\" type.");
    return;
  }

  ConcreteTypeDim dim = bits_type->size();
  absl::StatusOr<int64_t> dim_val = dim.GetAsInt64();
  if (!dim_val.ok()) {
    status_ = absl::InternalError("Unable to get bits type size as integer.");
    return;
  }

  absl::StatusOr<Bits> bits = node->GetBits(dim_val.value());
  if (!bits.ok()) {
    status_ = absl::InternalError(absl::StrCat(
        "Unable to convert number \"", node->ToString(), "\" to Bits."));
    return;
  }

  bytecode_.push_back(
      Bytecode(node->span(), Bytecode::Op::kLiteral,
               InterpValue::MakeBits(bits_type->is_signed(), bits.value())));
}

}  // namespace xls::dslx
