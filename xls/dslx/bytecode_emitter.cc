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

#include "absl/status/status.h"
#include "absl/strings/str_split.h"
#include "absl/types/variant.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/ast.h"

namespace xls::dslx {

BytecodeEmitter::BytecodeEmitter(ImportData* import_data, TypeInfo* type_info)
    : import_data_(import_data), type_info_(type_info) {}

BytecodeEmitter::~BytecodeEmitter() = default;

absl::StatusOr<BytecodeFunction> BytecodeEmitter::Emit(Function* f) {
  status_ = absl::OkStatus();
  bytecode_.clear();

  // Params fill the first N slots. The bytecode interpreter is responsible for
  // placing params correctly.
  for (const auto* param : f->params()) {
    namedef_to_slot_[param->name_def()] = namedef_to_slot_.size();
  }
  f->body()->AcceptExpr(this);

  if (!status_.ok()) {
    return status_;
  }

  return BytecodeFunction::Create(f, std::move(bytecode_));
}

void BytecodeEmitter::HandleArray(Array* node) {
  if (!status_.ok()) {
    return;
  }

  int num_members = node->members().size();
  for (auto* member : node->members()) {
    member->AcceptExpr(this);
  }

  // If we've got an ellipsis, then repeat the last element until we reach the
  // full array size.
  // must have; do check
  // Eval dim to int; must be constexpr in our type info
  if (node->has_ellipsis()) {
    ArrayTypeAnnotation* array_type =
        dynamic_cast<ArrayTypeAnnotation*>(node->type_annotation());
    absl::optional<InterpValue> array_dim =
        type_info_->GetConstExpr(array_type->dim());
    if (!array_dim.has_value()) {
      status_ = absl::InternalError(
          absl::StrFormat("Array %s did not have constant dimension: %s",
                          node->ToString(), array_type->dim()->ToString()));
      return;
    }

    absl::StatusOr<int64_t> dim_value = array_dim.value().GetBitValueInt64();
    if (!dim_value.ok()) {
      status_ = dim_value.status();
      return;
    }
    num_members = dim_value.value();
    int64_t remaining_members = num_members - node->members().size();
    for (int i = 0; i < remaining_members; i++) {
      node->members().back()->AcceptExpr(this);
    }
  }

  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kCreateArray,
                               Bytecode::NumElements(num_members)));
}

void BytecodeEmitter::HandleAttr(Attr* node) {
  if (!status_.ok()) {
    return;
  }

  // Will place a struct instance on the stack.
  node->lhs()->AcceptExpr(this);

  // Now we need the index of the attr NameRef in the struct def.
  absl::StatusOr<StructType*> struct_type_or =
      type_info_->GetItemAs<StructType>(node->lhs());
  if (!struct_type_or.ok()) {
    status_ = struct_type_or.status();
    return;
  }

  absl::StatusOr<int64_t> member_index_or =
      struct_type_or.value()->GetMemberIndex(node->attr()->identifier());
  if (!member_index_or.ok()) {
    status_ = member_index_or.status();
    return;
  }

  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kLiteral,
                               InterpValue::MakeS64(member_index_or.value())));
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kIndex));
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
    case BinopKind::kAnd:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kAnd));
      return;
    case BinopKind::kConcat:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kConcat));
      return;
    case BinopKind::kDiv:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kDiv));
      return;
    case BinopKind::kEq:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kEq));
      return;
    case BinopKind::kGe:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kGe));
      return;
    case BinopKind::kGt:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kGt));
      return;
    case BinopKind::kLe:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kLe));
      return;
    case BinopKind::kLt:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kLt));
      return;
    case BinopKind::kMul:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kMul));
      return;
    case BinopKind::kNe:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kNe));
      return;
    case BinopKind::kOr:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kOr));
      return;
    case BinopKind::kShl:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kShll));
      return;
    case BinopKind::kShr:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kShrl));
      return;
    case BinopKind::kSub:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kSub));
      return;
    case BinopKind::kXor:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kXor));
      return;
    default:
      status_ = absl::UnimplementedError(
          absl::StrCat("Unimplemented binary operator: ",
                       BinopKindToString(node->binop_kind())));
  }
}

absl::Status BytecodeEmitter::CastArrayToBits(Span span, ArrayType* from_array,
                                              BitsType* to_bits) {
  if (from_array->element_type().GetAllDims().size() != 1) {
    return absl::InternalError(
        "Only casts to/from one-dimensional arrays are supported.");
  }

  XLS_ASSIGN_OR_RETURN(ConcreteTypeDim bit_count_dim,
                       from_array->GetTotalBitCount());
  XLS_ASSIGN_OR_RETURN(int64_t array_bit_count, bit_count_dim.GetAsInt64());

  XLS_ASSIGN_OR_RETURN(bit_count_dim, to_bits->GetTotalBitCount());
  XLS_ASSIGN_OR_RETURN(int64_t bits_bit_count, bit_count_dim.GetAsInt64());

  if (array_bit_count != bits_bit_count) {
    return absl::InternalError(
        absl::StrFormat("Array-to-bits cast bit counts must match. "
                        "Saw %d vs %d.",
                        array_bit_count, bits_bit_count));
  }

  bytecode_.push_back(
      Bytecode(span, Bytecode::Op::kCast, to_bits->CloneToUnique()));
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::CastBitsToArray(Span span, BitsType* from_bits,
                                              ArrayType* to_array) {
  if (to_array->element_type().GetAllDims().size() != 1) {
    return absl::InternalError(
        "Only casts to/from one-dimensional arrays are supported.");
  }

  XLS_ASSIGN_OR_RETURN(ConcreteTypeDim bit_count_dim,
                       from_bits->GetTotalBitCount());
  XLS_ASSIGN_OR_RETURN(int64_t bits_bit_count, bit_count_dim.GetAsInt64());

  XLS_ASSIGN_OR_RETURN(bit_count_dim, to_array->GetTotalBitCount());
  XLS_ASSIGN_OR_RETURN(int64_t array_bit_count, bit_count_dim.GetAsInt64());

  if (array_bit_count != bits_bit_count) {
    return absl::InternalError(
        absl::StrFormat("Bits-to-array cast bit counts must match. "
                        "Saw %d vs %d.",
                        bits_bit_count, array_bit_count));
  }

  bytecode_.push_back(
      Bytecode(span, Bytecode::Op::kCast, to_array->CloneToUnique()));
  return absl::OkStatus();
}

void BytecodeEmitter::HandleCast(Cast* node) {
  if (!status_.ok()) {
    return;
  }

  node->expr()->AcceptExpr(this);

  absl::optional<ConcreteType*> maybe_from = type_info_->GetItem(node->expr());
  if (!maybe_from.has_value()) {
    status_ = absl::InternalError(
        absl::StrCat("Could not find type for cast \"from\" arg: ",
                     node->expr()->ToString()));
    return;
  }
  ConcreteType* from = maybe_from.value();

  absl::optional<ConcreteType*> maybe_to = type_info_->GetItem(node);
  if (!maybe_to.has_value()) {
    status_ = absl::InternalError(
        absl::StrCat("Could not find concrete type for cast \"to\" type: ",
                     node->type_annotation()->ToString()));
    return;
  }
  ConcreteType* to = maybe_to.value();

  if (ArrayType* from_array = dynamic_cast<ArrayType*>(from);
      from_array != nullptr) {
    BitsType* to_bits = dynamic_cast<BitsType*>(to);
    if (to_bits == nullptr) {
      status_ = absl::InternalError(absl::StrCat(
          "The only valid array cast is to bits: ", node->ToString()));
    }

    status_ = CastArrayToBits(node->span(), from_array, to_bits);
    return;
  }

  if (EnumType* from_enum = dynamic_cast<EnumType*>(from);
      from_enum != nullptr) {
    BitsType* to_bits = dynamic_cast<BitsType*>(to);
    if (to_bits == nullptr) {
      status_ = absl::InternalError(absl::StrCat(
          "The only valid enum cast is to bits: ", node->ToString()));
    }

    bytecode_.push_back(
        Bytecode(node->span(), Bytecode::Op::kCast, to_bits->CloneToUnique()));
    return;
  }

  BitsType* from_bits = dynamic_cast<BitsType*>(from);
  if (from_bits == nullptr) {
    status_ = absl::InternalError(
        "Only casts from arrays, enums, or bits are allowed.");
    return;
  }

  if (ArrayType* to_array = dynamic_cast<ArrayType*>(to); to_array != nullptr) {
    status_ = CastBitsToArray(node->span(), from_bits, to_array);
    return;
  }

  if (EnumType* to_enum = dynamic_cast<EnumType*>(to); to_enum != nullptr) {
    bytecode_.push_back(
        Bytecode(node->span(), Bytecode::Op::kCast, to_enum->CloneToUnique()));
    return;
  }

  BitsType* to_bits = dynamic_cast<BitsType*>(to);
  if (to_bits == nullptr) {
    status_ = absl::InternalError(
        "Only casts to arrays, enums, or bits are allowed.");
    return;
  }

  bytecode_.push_back(
      Bytecode(node->span(), Bytecode::Op::kCast, to_bits->CloneToUnique()));
}

absl::Status BytecodeEmitter::HandleColonRefToImportedConstant(
    ColonRef* colon_ref, Import* import, absl::string_view constant_name) {
  absl::optional<const ImportedInfo*> imported_info =
      type_info_->GetImported(import);
  if (!imported_info.has_value()) {
    return absl::InternalError(
        absl::StrCat("Could not find import: ", import->ToString()));
  }

  Module* imported_module = imported_info.value()->module;
  if (imported_module->GetFunction(constant_name).has_value()) {
    return absl::UnimplementedError(
        "ColonRefs to functions are not yet implemented.");
  }

  XLS_ASSIGN_OR_RETURN(ConstantDef * constant_def,
                       imported_module->GetConstantDef(constant_name));
  absl::optional<InterpValue> interp_value =
      imported_info.value()->type_info->GetConstExpr(constant_def->value());
  if (!interp_value.has_value()) {
    return absl::InternalError(
        absl::StrCat("Unable to find constant value for constant def ",
                     constant_def->ToString()));
  }
  bytecode_.push_back(Bytecode(colon_ref->span(), Bytecode::Op::kLiteral,
                               interp_value.value()));

  return absl::OkStatus();
}

absl::StatusOr<Bytecode> HandleColonRefToEnum(ColonRef* colon_ref,
                                              EnumDef* enum_def,
                                              TypeInfo* type_info) {
  absl::string_view attr = colon_ref->attr();
  XLS_ASSIGN_OR_RETURN(Expr * value_expr, enum_def->GetValue(attr));

  absl::optional<InterpValue> value_or = type_info->GetConstExpr(value_expr);
  if (!value_or.has_value()) {
    return absl::InternalError(absl::StrFormat(
        "Could not find value for enum \"%s\" attribute \"%s\".",
        enum_def->identifier(), attr));
  }

  return Bytecode(colon_ref->span(), Bytecode::Op::kLiteral, value_or.value());
}

absl::Status BytecodeEmitter::HandleColonRefToImportedEnum(
    ColonRef* colon_ref) {
  ColonRef* second_colon_ref = absl::get<ColonRef*>(colon_ref->subject());
  // Get the import module.
  NameRef* import_name = absl::get<NameRef*>(second_colon_ref->subject());
  Module* module = type_info_->module();
  absl::optional<ModuleMember*> maybe_member =
      module->FindMemberWithName(import_name->identifier());
  if (!maybe_member.has_value()) {
    return absl::InternalError(
        absl::StrCat("Unable to find module member with name \"",
                     import_name->identifier(), "\""));
  }

  if (!absl::holds_alternative<Import*>(*maybe_member.value())) {
    return absl::InternalError(absl::StrCat(
        "Expected ColonRef LHS to be Import: ", colon_ref->ToString()));
  }
  Import* import = absl::get<Import*>(*maybe_member.value());
  absl::optional<const ImportedInfo*> imported =
      type_info_->GetImported(import);
  if (!imported.has_value()) {
    return absl::NotFoundError(
        absl::StrCat("Could not find import: ", import->ToString()));
  }

  XLS_ASSIGN_OR_RETURN(
      TypeDefinition td,
      imported.value()->module->GetTypeDefinition(second_colon_ref->attr()));
  if (!absl::holds_alternative<EnumDef*>(td)) {
    return absl::InternalError(absl::StrCat(
        "Imported symbol ", second_colon_ref->attr(), " is not an EnumDef."));
  }

  EnumDef* enum_def = absl::get<EnumDef*>(td);
  XLS_ASSIGN_OR_RETURN(
      Bytecode bytecode,
      HandleColonRefToEnum(colon_ref, enum_def, imported.value()->type_info));
  bytecode_.push_back(std::move(bytecode));
  return absl::OkStatus();
}

void BytecodeEmitter::HandleColonRef(ColonRef* node) {
  // ColonRefs can have three forms:
  //  1. Ref to local enum: LocalEnum::Value
  //  2. Ref to imported constant or function: module::Constant
  //  3. Ref to imported enum: module::ImportedEnum::Value
  if (absl::holds_alternative<NameRef*>(node->subject())) {
    NameRef* name_ref = absl::get<NameRef*>(node->subject());
    Module* module = type_info_->module();
    absl::optional<ModuleMember*> maybe_member =
        module->FindMemberWithName(name_ref->identifier());
    if (!maybe_member.has_value()) {
      status_ = absl::InternalError(
          absl::StrCat("Unable to find module member with name \"",
                       name_ref->identifier(), "\""));
      return;
    }

    if (absl::holds_alternative<Import*>(*maybe_member.value())) {
      // Case 2.
      absl::Status status = HandleColonRefToImportedConstant(
          node, absl::get<Import*>(*maybe_member.value()), node->attr());
      if (!status.ok()) {
        status_ = status;
      }
    } else {
      // Case 1.
      EnumDef* enum_def = absl::get<EnumDef*>(*maybe_member.value());
      absl::StatusOr<Bytecode> bytecode_or =
          HandleColonRefToEnum(node, enum_def, type_info_);
      if (!bytecode_or.ok()) {
        status_ = bytecode_or.status();
      } else {
        bytecode_.push_back(std::move(bytecode_or.value()));
      }
    }
    return;
  }

  // Case 3: Ref to imported enum.
  status_ = HandleColonRefToImportedEnum(node);
}

void BytecodeEmitter::HandleConstRef(ConstRef* node) {
  if (!status_.ok()) {
    return;
  }

  ConstantDef* constant_def = node->GetConstantDef();
  absl::optional<InterpValue> const_val = type_info_->GetConstExpr(node);

  if (!const_val.has_value()) {
    status_ = absl::InternalError(absl::StrCat(
        "Could not find value for constant ", constant_def->ToString()));
    return;
  }

  bytecode_.push_back(
      Bytecode(node->span(), Bytecode::Op::kLiteral, const_val.value()));
}

absl::StatusOr<int64_t> GetValueWidth(TypeInfo* type_info, Expr* expr) {
  absl::optional<ConcreteType*> maybe_type = type_info->GetItem(expr);
  if (!maybe_type.has_value()) {
    return absl::InternalError(
        "Could not find concrete type for slice component.");
  }
  return maybe_type.value()->GetTotalBitCount()->GetAsInt64();
}

void BytecodeEmitter::HandleIndex(Index* node) {
  node->lhs()->AcceptExpr(this);

  if (absl::holds_alternative<Slice*>(node->rhs())) {
    Slice* slice = absl::get<Slice*>(node->rhs());
    if (slice->start() == nullptr) {
      int64_t start_width;
      if (slice->limit() == nullptr) {
        // TODO(rspringer): Define a uniform `usize` to avoid specifying magic
        // numbers here. This is the default size used for untyped numbers in
        // the typechecker.
        start_width = 32;
      } else {
        absl::StatusOr<int64_t> start_width_or =
            GetValueWidth(type_info_, slice->limit());
        if (!start_width_or.ok()) {
          status_ = start_width_or.status();
          return;
        }
        start_width = start_width_or.value();
      }
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kLiteral,
                                   InterpValue::MakeSBits(start_width, 0)));
    } else {
      slice->start()->AcceptExpr(this);
    }

    if (slice->limit() == nullptr) {
      absl::optional<ConcreteType*> maybe_type =
          type_info_->GetItem(node->lhs());
      if (!maybe_type.has_value()) {
        status_ =
            absl::InternalError("Could not find concrete type for slice.");
        return;
      }
      ConcreteType* type = maybe_type.value();
      // These will never fail.
      absl::StatusOr<ConcreteTypeDim> dim_or = type->GetTotalBitCount();
      absl::StatusOr<int64_t> width_or = dim_or.value().GetAsInt64();

      int64_t limit_width;
      if (slice->start() == nullptr) {
        // TODO(rspringer): Define a uniform `usize` to avoid specifying magic
        // numbers here. This is the default size used for untyped numbers in
        // the typechecker.
        limit_width = 32;
      } else {
        absl::StatusOr<int64_t> limit_width_or =
            GetValueWidth(type_info_, slice->start());
        if (!limit_width_or.ok()) {
          status_ = limit_width_or.status();
          return;
        }
        limit_width = limit_width_or.value();
      }
      bytecode_.push_back(
          Bytecode(node->span(), Bytecode::Op::kLiteral,
                   InterpValue::MakeSBits(limit_width, width_or.value())));
    } else {
      slice->limit()->AcceptExpr(this);
    }
    bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kSlice));
    return;
  }

  if (absl::holds_alternative<WidthSlice*>(node->rhs())) {
    WidthSlice* width_slice = absl::get<WidthSlice*>(node->rhs());
    width_slice->start()->AcceptExpr(this);
    absl::optional<ConcreteType*> maybe_type =
        type_info_->GetItem(width_slice->width());
    if (!maybe_type.has_value()) {
      status_ = absl::InternalError(absl::StrCat(
          "Could not find concrete type for slice width parameter \"",
          width_slice->width()->ToString(), "\"."));
      return;
    }
    ConcreteType* type = maybe_type.value();

    // These will never fail.
    absl::StatusOr<ConcreteTypeDim> dim_or = type->GetTotalBitCount();
    absl::StatusOr<int64_t> slice_width_or = dim_or.value().GetAsInt64();
    bytecode_.push_back(Bytecode(
        width_slice->GetSpan().value(), Bytecode::Op::kLiteral,
        InterpValue::MakeSBits(/*bit_count=*/64, slice_width_or.value())));
    bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kWidthSlice));
    return;
  }

  // Otherwise, it's a regular [array or tuple] index op.
  Expr* expr = absl::get<Expr*>(node->rhs());
  expr->AcceptExpr(this);
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kIndex));
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
    if (!namedef_to_slot_.contains(name_def)) {
      namedef_to_slot_.insert({name_def, namedef_to_slot_.size()});
    }
    int64_t slot = namedef_to_slot_.at(name_def);
    bytecode_.push_back(Bytecode(tree->span(), Bytecode::Op::kStore,
                                 Bytecode::SlotIndex(slot)));
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

  NameDef* name_def = absl::get<NameDef*>(node->name_def());
  int64_t slot = namedef_to_slot_.at(name_def);
  bytecode_.push_back(
      Bytecode(node->span(), Bytecode::Op::kLoad, Bytecode::SlotIndex(slot)));
}

void BytecodeEmitter::HandleNumber(Number* node) {
  if (!status_.ok()) {
    return;
  }

  absl::optional<ConcreteType*> type_or = type_info_->GetItem(node);
  if (!type_or.has_value()) {
    status_ = absl::InternalError(
        absl::StrCat("Could not find type for number: ", node->ToString()));
    return;
  }

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

void BytecodeEmitter::HandleString(String* node) {
  if (!status_.ok()) {
    return;
  }

  // A string is just a fancy array literal.
  for (const char c : node->text()) {
    bytecode_.push_back(
        Bytecode(node->span(), Bytecode::Op::kLiteral,
                 InterpValue::MakeUBits(/*bit_count=*/8, static_cast<int>(c))));
  }

  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kCreateArray,
                               Bytecode::NumElements(node->text().size())));
}

void BytecodeEmitter::HandleStructInstance(StructInstance* node) {
  if (!status_.ok()) {
    return;
  }

  absl::StatusOr<StructType*> struct_type_or =
      type_info_->GetItemAs<StructType>(node);
  if (!struct_type_or.ok()) {
    status_ = struct_type_or.status();
    return;
  }

  const StructDef& struct_def = struct_type_or.value()->nominal_type();
  for (const std::pair<std::string, Expr*>& member :
       node->GetOrderedMembers(&struct_def)) {
    member.second->AcceptExpr(this);
  }

  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kCreateTuple,
                               Bytecode::NumElements(struct_def.size())));
}

void BytecodeEmitter::HandleSplatStructInstance(SplatStructInstance* node) {
  // For each field in the struct:
  //   If it's specified in the SplatStructInstance, then use it,
  //   otherwise extract it from the base struct.
  if (!status_.ok()) {
    return;
  }

  absl::StatusOr<StructType*> struct_type =
      type_info_->GetItemAs<StructType>(node);
  if (!struct_type.ok()) {
    status_ = struct_type.status();
    return;
  }

  absl::StatusOr<std::vector<std::string>> member_names =
      struct_type.value()->GetMemberNames();
  if (!member_names.ok()) {
    status_ = member_names.status();
    return;
  }

  absl::flat_hash_map<std::string, Expr*> new_members;
  for (const std::pair<std::string, Expr*>& new_member : node->members()) {
    new_members[new_member.first] = new_member.second;
  }

  // To extract values from the base struct, we need to visit it and extract the
  // appropriate attr (i.e., index into the representative tuple).
  // Referencing a base struct doesn't have side effects, so we can just
  // visit it as often as we need.

  // Member names is ordered.
  for (int i = 0; i < member_names->size(); i++) {
    std::string& member_name = member_names->at(i);
    if (new_members.contains(member_name)) {
      new_members.at(member_name)->AcceptExpr(this);
    } else {
      node->splatted()->AcceptExpr(this);
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kLiteral,
                                   InterpValue::MakeS64(i)));
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kIndex));
    }
  }

  const StructDef& struct_def = struct_type.value()->nominal_type();
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kCreateTuple,
                               InterpValue::MakeS64(struct_def.size())));
}

void BytecodeEmitter::HandleUnop(Unop* node) {
  if (!status_.ok()) {
    return;
  }

  node->operand()->AcceptExpr(this);
  switch (node->unop_kind()) {
    case UnopKind::kInvert:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kInvert));
      break;
    case UnopKind::kNegate:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kNegate));
      break;
  }
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
                               Bytecode::NumElements(node->members().size())));
}

void BytecodeEmitter::HandleTernary(Ternary* node) {
  if (!status_.ok()) {
    return;
  }

  // Structure is:
  //
  //  $test
  //  jump_if =>consequent
  // alternate:
  //  $alternate
  //  jump =>join
  // consequent:
  //  jump_dest
  //  $consequent
  // join:
  //  jump_dest
  node->test()->AcceptExpr(this);
  size_t jump_if_index = bytecode_.size();
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kJumpRelIf,
                               Bytecode::kPlaceholderJumpAmount));
  node->alternate()->AcceptExpr(this);
  size_t jump_index = bytecode_.size();
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kJumpRel,
                               Bytecode::kPlaceholderJumpAmount));
  size_t consequent_index = bytecode_.size();
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kJumpDest));
  node->consequent()->AcceptExpr(this);
  size_t jumpdest_index = bytecode_.size();

  // Now patch up the jumps since we now know the relative PCs we're jumping to.
  // TODO(leary): 2021-12-10 Keep track of the stack depth so we can validate it
  // at the jump destination.
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kJumpDest));
  bytecode_.at(jump_if_index).Patch(consequent_index - jump_if_index);
  bytecode_.at(jump_index).Patch(jumpdest_index - jump_index);
}

}  // namespace xls::dslx
