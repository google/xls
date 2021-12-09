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
namespace {

std::string OpToString(Bytecode::Op op) {
  switch (op) {
    case Bytecode::Op::kAdd:
      return "add";
    case Bytecode::Op::kCall:
      return "call";
    case Bytecode::Op::kCreateTuple:
      return "create_tuple";
    case Bytecode::Op::kExpandTuple:
      return "expand_tuple";
    case Bytecode::Op::kEq:
      return "eq";
    case Bytecode::Op::kLoad:
      return "load";
    case Bytecode::Op::kLiteral:
      return "literal";
    case Bytecode::Op::kStore:
      return "store";
  }
  return absl::StrCat("<invalid: ", static_cast<int>(op), ">");
}

}  // namespace

std::string Bytecode::ToString() const {
  std::string op_string = OpToString(op_);

  if (data_.has_value()) {
    std::string data_string;
    if (absl::holds_alternative<int64_t>(data_.value())) {
      int64_t data = absl::get<int64_t>(data_.value());
      data_string = absl::StrCat("d", data);
    } else {
      data_string = absl::get<InterpValue>(data_.value()).ToHumanString();
    }
    return absl::StrFormat("%s <%s> @ %s", op_string, data_string,
                           source_span_.ToString());
  }
  return absl::StrFormat("%s @ %s", op_string, source_span_.ToString());
}

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

void BytecodeEmitter::DestructureLet(NameDefTree* tree) {
  if (tree->is_leaf()) {
    NameDef* name_def = absl::get<NameDef*>(tree->leaf());
    if (!namedef_to_slot_->contains(name_def)) {
      namedef_to_slot_->insert({name_def, namedef_to_slot_->size()});
    }
    int64_t slot = namedef_to_slot_->at(name_def);
    bytecode_.push_back(Bytecode(tree->span(), Bytecode::Op::kStore, slot));
  } else {
    bytecode_.push_back(Bytecode(tree->span(), Bytecode::Op::kExpandTuple));
    for (const auto& node : tree->nodes()) {
      DestructureLet(node);
    }
  }
}

void BytecodeEmitter::HandleLet(Let* node) {
  if (!status_.ok()) {
    return;
  }

  node->rhs()->AcceptExpr(this);
  if (!status_.ok()) {
    return;
  }

  DestructureLet(node->name_def_tree());

  node->body()->AcceptExpr(this);
}

void BytecodeEmitter::HandleNameRef(NameRef* node) {
  if (!status_.ok()) {
    return;
  }

  if (absl::holds_alternative<BuiltinNameDef*>(node->name_def())) {
    status_ =
        absl::UnimplementedError("NameRefs to builtins are not yet supported.");
  }
  int64_t slot = namedef_to_slot_->at(absl::get<NameDef*>(node->name_def()));
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kLoad, slot));
}

void BytecodeEmitter::HandleNumber(Number* node) {
  if (!status_.ok()) {
    return;
  }

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

void BytecodeEmitter::HandleXlsTuple(XlsTuple* node) {
  if (!status_.ok()) {
    return;
  }

  for (auto* member : node->members()) {
    member->AcceptExpr(this);
  }

  // Pop the N elements and push the result as a single value.
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kCreateTuple,
                               node->members().size()));
}
}  // namespace xls::dslx
