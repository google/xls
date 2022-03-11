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

#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/strings/str_split.h"
#include "absl/types/variant.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/dslx/ast.h"
#include "xls/dslx/ast_utils.h"
#include "xls/dslx/concrete_type.h"
#include "xls/dslx/interp_value.h"

// TODO(rspringer): 2022-03-01: Verify that, for all valid programs (or at least
// some subset that we test), interpretation terminates with only a single value
// on the stack (I believe that should be the case for any valid program).

namespace xls::dslx {

BytecodeEmitter::BytecodeEmitter(
    ImportData* import_data, const TypeInfo* type_info,
    const absl::optional<SymbolicBindings>& caller_bindings)
    : import_data_(import_data),
      type_info_(type_info),
      caller_bindings_(caller_bindings) {}

BytecodeEmitter::~BytecodeEmitter() = default;

absl::Status BytecodeEmitter::Init(const Function* f) {
  for (const auto* param : f->params()) {
    namedef_to_slot_[param->name_def()] = namedef_to_slot_.size();
  }

  return absl::OkStatus();
}

/* static */ absl::StatusOr<std::unique_ptr<BytecodeFunction>>
BytecodeEmitter::Emit(ImportData* import_data, const TypeInfo* type_info,
                      const Function* f,
                      const absl::optional<SymbolicBindings>& caller_bindings) {
  BytecodeEmitter emitter(import_data, type_info, caller_bindings);
  XLS_RETURN_IF_ERROR(emitter.Init(f));
  f->body()->AcceptExpr(&emitter);

  if (!emitter.status_.ok()) {
    return emitter.status_;
  }

  return BytecodeFunction::Create(f->owner(), type_info,
                                  std::move(emitter.bytecode_));
}

// Extracts all NameDefs "downstream" of a given AstNode. This
// is needed for Expr evaluation, so we can reserve slots for provided
// InterpValues (in namedef_to_slot_).
class NameDefCollector : public AstNodeVisitor {
 public:
  const absl::flat_hash_map<std::string, NameDef*>& name_defs() {
    return name_defs_;
  }

#define DEFAULT_HANDLER(NODE)                                      \
  absl::Status Handle##NODE(NODE* n) override {                    \
    for (AstNode * child : n->GetChildren(/*want_types=*/false)) { \
      XLS_RETURN_IF_ERROR(child->Accept(this));                    \
    }                                                              \
    return absl::OkStatus();                                       \
  }

  DEFAULT_HANDLER(Array);
  DEFAULT_HANDLER(ArrayTypeAnnotation);
  absl::Status HandleAttr(Attr* n) {
    XLS_RETURN_IF_ERROR(n->lhs()->Accept(this));
    return absl::OkStatus();
  }
  DEFAULT_HANDLER(Binop);
  DEFAULT_HANDLER(BuiltinNameDef);
  DEFAULT_HANDLER(BuiltinTypeAnnotation);
  DEFAULT_HANDLER(Cast);
  DEFAULT_HANDLER(ChannelDecl);
  DEFAULT_HANDLER(ChannelTypeAnnotation);
  DEFAULT_HANDLER(ColonRef);
  DEFAULT_HANDLER(ConstantArray);
  DEFAULT_HANDLER(ConstantDef);
  DEFAULT_HANDLER(ConstRef);
  DEFAULT_HANDLER(EnumDef);
  DEFAULT_HANDLER(For);
  DEFAULT_HANDLER(FormatMacro);
  absl::Status HandleFunction(Function* n) {
    return absl::InternalError("Encountered nested Function?");
  }
  DEFAULT_HANDLER(Index);
  DEFAULT_HANDLER(Invocation);
  DEFAULT_HANDLER(Import);
  DEFAULT_HANDLER(Join);
  DEFAULT_HANDLER(Let);
  DEFAULT_HANDLER(Match);
  DEFAULT_HANDLER(MatchArm);
  DEFAULT_HANDLER(Module);
  absl::Status HandleNameDef(NameDef* n) override {
    name_defs_[n->identifier()] = n;
    return absl::OkStatus();
  }
  DEFAULT_HANDLER(NameDefTree);
  DEFAULT_HANDLER(NameRef);
  DEFAULT_HANDLER(Number);
  DEFAULT_HANDLER(Param);
  DEFAULT_HANDLER(ParametricBinding);
  absl::Status HandleProc(Proc* n) {
    return absl::InternalError("Encountered nested Proc?");
  }
  DEFAULT_HANDLER(QuickCheck);
  DEFAULT_HANDLER(Recv);
  DEFAULT_HANDLER(RecvIf);
  DEFAULT_HANDLER(Send);
  DEFAULT_HANDLER(SendIf);
  DEFAULT_HANDLER(Slice);
  DEFAULT_HANDLER(Spawn);
  DEFAULT_HANDLER(SplatStructInstance);
  DEFAULT_HANDLER(String);
  DEFAULT_HANDLER(StructDef);
  absl::Status HandleTestFunction(TestFunction* n) {
    return absl::InternalError("Encountered nested TestFunction?");
  }
  absl::Status HandleStructInstance(StructInstance* n) {
    for (const auto& member : n->GetUnorderedMembers()) {
      XLS_RETURN_IF_ERROR(member.second->Accept(this));
    }
    return absl::OkStatus();
  }
  DEFAULT_HANDLER(Ternary);
  DEFAULT_HANDLER(TestProc);
  DEFAULT_HANDLER(TupleTypeAnnotation);
  DEFAULT_HANDLER(TypeDef);
  DEFAULT_HANDLER(TypeRef);
  DEFAULT_HANDLER(TypeRefTypeAnnotation);
  DEFAULT_HANDLER(Unop);
  DEFAULT_HANDLER(WidthSlice);
  DEFAULT_HANDLER(WildcardPattern);
  DEFAULT_HANDLER(XlsTuple);

 private:
  absl::flat_hash_map<std::string, NameDef*> name_defs_;
};

/* static */ absl::StatusOr<std::unique_ptr<BytecodeFunction>>
BytecodeEmitter::EmitExpression(
    ImportData* import_data, const TypeInfo* type_info, Expr* expr,
    const absl::flat_hash_map<std::string, InterpValue>& env,
    const absl::optional<SymbolicBindings>& caller_bindings) {
  BytecodeEmitter emitter(import_data, type_info, caller_bindings);

  NameDefCollector collector;
  XLS_RETURN_IF_ERROR(expr->Accept(&collector));

  for (const auto& [identifier, name_def] : collector.name_defs()) {
    AstNode* definer = name_def->definer();
    if (dynamic_cast<Function*>(definer) != nullptr ||
        dynamic_cast<Import*>(definer) != nullptr) {
      continue;
    }

    if (!env.contains(identifier)) {
      continue;
    }

    int64_t slot_index = emitter.namedef_to_slot_.size();
    emitter.namedef_to_slot_[name_def] = slot_index;
    emitter.Add(Bytecode::MakeLiteral(expr->span(), env.at(identifier)));
    emitter.Add(
        Bytecode::MakeStore(expr->span(), Bytecode::SlotIndex(slot_index)));
  }

  expr->AcceptExpr(&emitter);
  if (!emitter.status_.ok()) {
    return emitter.status_;
  }

  return BytecodeFunction::Create(expr->owner(), type_info,
                                  std::move(emitter.bytecode_));
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
  if (node->has_ellipsis()) {
    absl::StatusOr<ArrayType*> array_type_or =
        type_info_->GetItemAs<ArrayType>(node);
    if (!array_type_or.ok()) {
      status_ = array_type_or.status();
      return;
    }
    const ConcreteTypeDim& dim = array_type_or.value()->size();
    absl::StatusOr<int64_t> dim_value = dim.GetAsInt64();
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

  // This indexing literal needs to be unsigned since InterpValue::Index
  // requires an unsigned value.
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kLiteral,
                               InterpValue::MakeU64(member_index_or.value())));
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
    case BinopKind::kLogicalAnd:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kLogicalAnd));
      return;
    case BinopKind::kLogicalOr:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kLogicalOr));
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
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kShl));
      return;
    case BinopKind::kShr:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kShr));
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
    status_ = absl::InternalError(absl::StrCat(
        "Could not find concrete type for cast \"to\" type: ",
        node->type_annotation()->ToString(), " : ", node->span().ToString()));
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

absl::StatusOr<InterpValue> BytecodeEmitter::HandleColonRefToEnum(
    ColonRef* colon_ref, EnumDef* enum_def, const TypeInfo* type_info) {
  // TODO(rspringer): 2022-01-26 We'll need to pull the right type info during
  // ResolveTypeDefToEnum.
  absl::string_view attr = colon_ref->attr();
  XLS_ASSIGN_OR_RETURN(Expr * value_expr, enum_def->GetValue(attr));

  absl::optional<InterpValue> value_or = type_info->GetConstExpr(value_expr);
  if (!value_or.has_value()) {
    return absl::InternalError(absl::StrFormat(
        "Could not find value for enum \"%s\" attribute \"%s\".",
        enum_def->identifier(), attr));
  }

  return value_or.value();
}

absl::StatusOr<InterpValue> BytecodeEmitter::HandleColonRefToValue(
    Module* module, ColonRef* colon_ref) {
  // TODO(rspringer): We'll need subject resolution to return the appropriate
  // TypeInfo for parametrics.
  XLS_ASSIGN_OR_RETURN(TypeInfo * type_info,
                       import_data_->GetRootTypeInfo(module));

  absl::optional<ModuleMember*> member =
      module->FindMemberWithName(colon_ref->attr());
  if (!member.has_value()) {
    return absl::InternalError(
        absl::StrFormat("Could not find member %s of module %s.",
                        colon_ref->attr(), module->name()));
  }

  if (absl::holds_alternative<Function*>(*member.value())) {
    Function* f = absl::get<Function*>(*member.value());
    return InterpValue::MakeFunction(InterpValue::UserFnData{module, f});
  }

  XLS_RET_CHECK(absl::holds_alternative<ConstantDef*>(*member.value()));
  ConstantDef* constant_def = absl::get<ConstantDef*>(*member.value());
  absl::optional<InterpValue> maybe_value =
      type_info->GetConstExpr(constant_def->value());
  if (!maybe_value.has_value()) {
    return absl::InternalError(
        absl::StrCat("Could not find constexpr value for ConstantDef \"",
                     constant_def->ToString(), "\"."));
  }
  return maybe_value.value();
}

void BytecodeEmitter::HandleColonRef(ColonRef* node) {
  if (!status_.ok()) {
    return;
  }

  absl::StatusOr<InterpValue> value_or = HandleColonRefInternal(node);
  if (!value_or.ok()) {
    status_ = value_or.status();
  }

  Add(Bytecode::MakeLiteral(node->span(), value_or.value()));
}

absl::StatusOr<InterpValue> BytecodeEmitter::HandleColonRefInternal(
    ColonRef* node) {
  absl::variant<Module*, EnumDef*> resolved_subject;
  XLS_ASSIGN_OR_RETURN(resolved_subject,
                       ResolveColonRefSubject(import_data_, type_info_, node));

  if (absl::holds_alternative<EnumDef*>(resolved_subject)) {
    EnumDef* enum_def = absl::get<EnumDef*>(resolved_subject);
    const TypeInfo* type_info = type_info_;
    if (enum_def->owner() != type_info_->module()) {
      type_info = import_data_->GetRootTypeInfo(enum_def->owner()).value();
    }
    return HandleColonRefToEnum(node, enum_def, type_info);
  }

  Module* module = absl::get<Module*>(resolved_subject);
  return HandleColonRefToValue(module, node);
}

void BytecodeEmitter::HandleConstRef(ConstRef* node) { HandleNameRef(node); }

void BytecodeEmitter::HandleFor(For* node) {
  // Here's how a `for` loop is implemented, in some sort of pseudocode:
  //  - Initialize iterable & index
  //  - Initialize the accumulator
  //  - loop_top:
  //  - if index == iterable_sz: jump to end:
  //  - Create the loop carry: the tuple of (index, accumulator).
  //  - Execute the loop body.
  //  - Jump to loop_top:
  //  - end:

  // First, get the size of the iterable array.
  absl::optional<ConcreteType*> maybe_iterable_type =
      type_info_->GetItem(node->iterable());
  if (!maybe_iterable_type.has_value()) {
    status_ = absl::InternalError(
        absl::StrCat("Could not find concrete type for loop iterable: ",
                     node->iterable()->ToString()));
    return;
  }

  ArrayType* array_type = dynamic_cast<ArrayType*>(maybe_iterable_type.value());
  if (array_type == nullptr) {
    status_ = absl::InternalError(absl::StrCat(
        "Iterable was not of array type: ", node->iterable()->ToString()));
    return;
  }

  ConcreteTypeDim iterable_size_dim = array_type->size();
  absl::StatusOr<int64_t> iterable_size_or = iterable_size_dim.GetAsInt64();
  if (!iterable_size_or.ok()) {
    status_ = iterable_size_or.status();
    return;
  }
  int64_t iterable_size = iterable_size_or.value();

  // A `for` loop defines a new scope, meaning that any names defined in that
  // scope (i.e., NameDefs) aren't valid outside the loop (i.e., they shouldn't
  // be present in namedef_to_slot_.). To accomplish this, we create a
  // new namedef_to_slot_ and restrict its lifetime to the loop instructions
  // only. Once the loops scope ends, the previous map is restored.
  absl::flat_hash_map<const NameDef*, int64_t> old_namedef_to_slot =
      namedef_to_slot_;
  auto cleanup = absl::MakeCleanup([this, &old_namedef_to_slot]() {
    namedef_to_slot_ = old_namedef_to_slot;
  });

  // We need a means of referencing the loop index and accumulator in the
  // namedef_to_slot_ map, so we pretend that they're NameDefs for uniqueness.
  int iterable_slot = namedef_to_slot_.size();
  NameDef* fake_name_def = reinterpret_cast<NameDef*>(node->iterable());
  namedef_to_slot_[fake_name_def] = iterable_slot;
  node->iterable()->AcceptExpr(this);
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kStore,
                               Bytecode::SlotIndex(iterable_slot)));

  int index_slot = namedef_to_slot_.size();
  fake_name_def = reinterpret_cast<NameDef*>(node);
  namedef_to_slot_[fake_name_def] = index_slot;
  bytecode_.push_back(
      Bytecode(node->span(), Bytecode::Op::kLiteral, InterpValue::MakeU32(0)));
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kStore,
                               Bytecode::SlotIndex(index_slot)));

  // Evaluate the initial accumulator value & leave it on the stack.
  node->init()->AcceptExpr(this);

  // Jump destination for the end-of-loop jump to start.
  int top_of_loop = bytecode_.size();
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kJumpDest));

  // Loop header: Are we done iterating?
  // Reload the current index and compare against the iterable size.
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kLoad,
                               Bytecode::SlotIndex(index_slot)));
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kLiteral,
                               InterpValue::MakeU32(iterable_size)));
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kEq));

  // Cache the location of the top-of-loop jump so we can patch it up later once
  // we actually know the size of the jump.
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kJumpRelIf,
                               Bytecode::kPlaceholderJumpAmount));
  int start_jump_idx = bytecode_.size() - 1;

  // The loop-carry values are a tuple (index, accumulator), but we don't have
  // the index on the stack (so we don't have to drop it once we're out of the
  // loop). We need to plop it on there, then swap the top two values so that
  // the resulting tuple is in the right order.
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kLoad,
                               Bytecode::SlotIndex(iterable_slot)));
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kLoad,
                               Bytecode::SlotIndex(index_slot)));
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kIndex));
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kSwap));
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kCreateTuple,
                               Bytecode::NumElements(2)));
  DestructureLet(node->names());

  // Emit the loop body.
  node->body()->AcceptExpr(this);

  // End of loop: increment the loop index and jump back to the beginning.
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kLoad,
                               Bytecode::SlotIndex(index_slot)));
  bytecode_.push_back(
      Bytecode(node->span(), Bytecode::Op::kLiteral, InterpValue::MakeU32(1)));
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kAdd));
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kStore,
                               Bytecode::SlotIndex(index_slot)));
  bytecode_.push_back(
      Bytecode(node->span(), Bytecode::Op::kJumpRel,
               Bytecode::JumpTarget(top_of_loop - bytecode_.size())));

  // Finally, we're done with the loop!
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kJumpDest));

  // Now we can finally know the relative jump amount for the top-of-loop jump.
  bytecode_.at(start_jump_idx)
      .PatchJumpTarget(bytecode_.size() - start_jump_idx - 1);
}

void BytecodeEmitter::HandleFormatMacro(FormatMacro* node) {
  if (!status_.ok()) {
    return;
  }

  for (auto* arg : node->args()) {
    arg->AcceptExpr(this);
  }

  bytecode_.push_back(
      Bytecode(node->span(), Bytecode::Op::kTrace, node->format()));
}

absl::StatusOr<int64_t> GetValueWidth(const TypeInfo* type_info, Expr* expr) {
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
    BitsType* bits_type = dynamic_cast<BitsType*>(type);
    if (bits_type == nullptr) {
      status_ = absl::InternalError(absl::StrCat(
          "Width slice type specifier isn't a BitsType: ", type->ToString()));
      return;
    }

    bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kWidthSlice,
                                 type->CloneToUnique()));
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

  if (NameRef* name_ref = dynamic_cast<NameRef*>(node->callee());
      name_ref != nullptr) {
    if (name_ref->identifier() == "trace!") {
      if (node->args().size() != 1) {
        status_ = absl::InternalError("`trace!` takes a single argument.");
        return;
      }

      node->args().at(0)->AcceptExpr(this);

      Bytecode::TraceData trace_data;
      trace_data.push_back(absl::StrCat("trace of ",
                                        node->args()[0]->ToString(), " @ ",
                                        node->span().ToString(), ": "));
      trace_data.push_back(FormatPreference::kDefault);
      bytecode_.push_back(
          Bytecode(node->span(), Bytecode::Op::kTrace, trace_data));
      return;
    }
  }

  for (auto* arg : node->args()) {
    arg->AcceptExpr(this);
  }

  node->callee()->AcceptExpr(this);

  absl::optional<const SymbolicBindings*> maybe_callee_bindings =
      type_info_->GetInstantiationCalleeBindings(
          node, caller_bindings_.has_value() ? caller_bindings_.value()
                                             : SymbolicBindings());
  absl::optional<SymbolicBindings> final_bindings = absl::nullopt;
  if (maybe_callee_bindings.has_value()) {
    final_bindings = *maybe_callee_bindings.value();
  }
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kCall,
                               Bytecode::InvocationData{node, final_bindings}));
}

absl::StatusOr<Bytecode::MatchArmItem> BytecodeEmitter::HandleNameDefTreeExpr(
    NameDefTree* tree) {
  if (tree->is_leaf()) {
    return absl::visit(
        Visitor{
            [&](NameRef* n) -> absl::StatusOr<Bytecode::MatchArmItem> {
              using ValueT = absl::variant<InterpValue, Bytecode::SlotIndex>;
              XLS_ASSIGN_OR_RETURN(ValueT item_value, HandleNameRefInternal(n));
              if (absl::holds_alternative<InterpValue>(item_value)) {
                return Bytecode::MatchArmItem::MakeInterpValue(
                    absl::get<InterpValue>(item_value));
              }

              return Bytecode::MatchArmItem::MakeLoad(
                  absl::get<Bytecode::SlotIndex>(item_value));
            },
            [&](Number* n) -> absl::StatusOr<Bytecode::MatchArmItem> {
              XLS_ASSIGN_OR_RETURN(InterpValue number, HandleNumberInternal(n));
              return Bytecode::MatchArmItem::MakeInterpValue(number);
            },
            [&](ColonRef* n) -> absl::StatusOr<Bytecode::MatchArmItem> {
              XLS_ASSIGN_OR_RETURN(InterpValue value,
                                   HandleColonRefInternal(n));
              return Bytecode::MatchArmItem::MakeInterpValue(value);
            },
            [&](NameDef* n) -> absl::StatusOr<Bytecode::MatchArmItem> {
              int64_t slot_index = namedef_to_slot_.size();
              namedef_to_slot_[n] = slot_index;
              return Bytecode::MatchArmItem::MakeStore(
                  Bytecode::SlotIndex(slot_index));
            },
            [&](WildcardPattern* n) -> absl::StatusOr<Bytecode::MatchArmItem> {
              return Bytecode::MatchArmItem::MakeWildcard();
            },
        },
        tree->leaf());
  }

  std::vector<Bytecode::MatchArmItem> elements;
  for (NameDefTree* node : tree->nodes()) {
    XLS_ASSIGN_OR_RETURN(Bytecode::MatchArmItem element,
                         HandleNameDefTreeExpr(node));
    elements.push_back(element);
  }
  return Bytecode::MatchArmItem::MakeTuple(std::move(elements));
}

void BytecodeEmitter::DestructureLet(NameDefTree* tree) {
  if (tree->is_leaf()) {
    if (absl::holds_alternative<WildcardPattern*>(tree->leaf())) {
      Add(Bytecode::MakePop(tree->span()));
      // We can just drop this one.
      return;
    }

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

  absl::StatusOr<absl::variant<InterpValue, Bytecode::SlotIndex>> result_or =
      HandleNameRefInternal(node);
  if (!result_or.ok()) {
    status_ = result_or.status();
    return;
  }

  auto result = result_or.value();
  if (absl::holds_alternative<InterpValue>(result)) {
    Add(Bytecode::MakeLiteral(node->span(), absl::get<InterpValue>(result)));
  } else {
    Add(Bytecode::MakeLoad(node->span(),
                           absl::get<Bytecode::SlotIndex>(result)));
  }
}

absl::StatusOr<absl::variant<InterpValue, Bytecode::SlotIndex>>
BytecodeEmitter::HandleNameRefInternal(NameRef* node) {
  if (absl::holds_alternative<BuiltinNameDef*>(node->name_def())) {
    // Builtins don't have NameDefs, so we can't use slots to store them. It's
    // simpler to just emit a literal InterpValue, anyway.
    BuiltinNameDef* builtin_def = absl::get<BuiltinNameDef*>(node->name_def());
    XLS_ASSIGN_OR_RETURN(Builtin builtin,
                         BuiltinFromString(builtin_def->identifier()));
    return InterpValue::MakeFunction(builtin);
  }

  // Emit function and constant refs directly so that they can be stack elements
  // without having to load slots with them.
  NameDef* name_def = absl::get<NameDef*>(node->name_def());
  if (auto* f = dynamic_cast<Function*>(name_def->definer()); f != nullptr) {
    return InterpValue::MakeFunction(InterpValue::UserFnData{f->owner(), f});
  }

  if (auto* cd = dynamic_cast<ConstantDef*>(name_def->definer());
      cd != nullptr) {
    absl::optional<InterpValue> const_value =
        type_info_->GetConstExpr(cd->value());
    if (!const_value.has_value()) {
      return absl::InternalError(
          absl::StrCat("Could not find value for constant: ", cd->ToString()));
    }
    return const_value.value();
  }

  // The value is either a local name or a symbolic binding.
  if (namedef_to_slot_.contains(name_def)) {
    return Bytecode::SlotIndex(namedef_to_slot_.at(name_def));
  }

  if (caller_bindings_.has_value()) {
    absl::flat_hash_map<std::string, InterpValue> bindings_map =
        caller_bindings_.value().ToMap();
    if (bindings_map.contains(name_def->identifier())) {
      return caller_bindings_.value().ToMap().at(name_def->identifier());
    }
  }

  return absl::InternalError(absl::StrCat(
      "Could not find slot or binding for name: ", name_def->ToString()));
}

void BytecodeEmitter::HandleNumber(Number* node) {
  if (!status_.ok()) {
    return;
  }

  absl::StatusOr<InterpValue> value_or = HandleNumberInternal(node);
  if (!value_or.ok()) {
    status_ = value_or.status();
    return;
  }

  Add(Bytecode::MakeLiteral(node->span(), value_or.value()));
}

absl::StatusOr<InterpValue> BytecodeEmitter::HandleNumberInternal(
    Number* node) {
  absl::optional<ConcreteType*> type_or = type_info_->GetItem(node);
  if (!type_or.has_value()) {
    return absl::InternalError(
        absl::StrCat("Could not find type for number: ", node->ToString()));
  }

  BitsType* bits_type = dynamic_cast<BitsType*>(type_or.value());
  if (bits_type == nullptr) {
    return absl::InternalError(
        "Error in type deduction; number did not have \"bits\" type.");
  }

  ConcreteTypeDim dim = bits_type->size();
  XLS_ASSIGN_OR_RETURN(int64_t dim_val, dim.GetAsInt64());
  XLS_ASSIGN_OR_RETURN(Bits bits, node->GetBits(dim_val));
  return InterpValue::MakeBits(bits_type->is_signed(), bits);
}

void BytecodeEmitter::HandleString(String* node) {
  if (!status_.ok()) {
    return;
  }

  // A string is just a fancy array literal.
  for (const unsigned char c : node->text()) {
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
                                   InterpValue::MakeU64(i)));
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kIndex));
    }
  }

  const StructDef& struct_def = struct_type.value()->nominal_type();
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kCreateTuple,
                               Bytecode::NumElements(struct_def.size())));
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
  bytecode_.at(jump_if_index).PatchJumpTarget(consequent_index - jump_if_index);
  bytecode_.at(jump_index).PatchJumpTarget(jumpdest_index - jump_index);
}

void BytecodeEmitter::HandleMatch(Match* node) {
  if (!status_.ok()) {
    return;
  }

  // Structure is: (value-to-match is at TOS0)
  //
  //  <value-to-match present>
  //  dup                       # Duplicate the value to match so the test can
  //                            # consume it and we can go to the next one if it
  //                            # fails.
  //  arm0_matcher_value        # The value to match against.
  //  ne                        # If it's not a match...
  //  jump_rel_if =>arm1_match  # Jump to the next arm!
  //  pop                       # Otherwise, we don't need the matchee any more,
  //                            # we found what arm we want to eval.
  //  $arm0_expr                # Eval the arm RHS.
  //  jump =>done               # Done evaluating the match, goto done.
  // arm1_match:                # Next match arm test...
  //  jump_dest                 # Is something we jump to.
  //  dup                       # It also needs to copy in case its test fails.
  //  arm1_matcher_value        # etc.
  //  ne
  //  jump_rel_if => wild_arm_match
  //  pop
  //  $arm1_expr
  //  jump =>done
  // wild_arm_match:            # Final arm must be wildcard.
  //  pop                       # Where we don't care about the matched value.
  //  $wild_arm_expr            # We just eval the RHS.
  // done:
  //  jump_dest                 # TOS0 holds result, matched value is gone.
  node->matched()->AcceptExpr(this);

  const std::vector<MatchArm*>& arms = node->arms();
  if (arms.empty()) {
    status_ = absl::InternalError("At least one match arm is required.");
    return;
  }

  std::vector<size_t> arm_offsets;
  std::vector<size_t> jumps_to_next;
  std::vector<size_t> jumps_to_done;

  for (size_t arm_idx = 0; arm_idx < node->arms().size(); ++arm_idx) {
    auto outer_scope_slots = namedef_to_slot_;
    auto cleanup = absl::MakeCleanup(
        [this, &outer_scope_slots]() { namedef_to_slot_ = outer_scope_slots; });

    MatchArm* arm = node->arms()[arm_idx];
    arm_offsets.push_back(bytecode_.size());
    // We don't jump to the first arm test, but we jump to all the subsequent
    // ones.
    if (arm_offsets.size() > 1) {
      Add(Bytecode::MakeJumpDest(node->span()));
    }

    const std::vector<NameDefTree*>& patterns = arm->patterns();
    // First, prime the stack with all the copies of the matchee we'll need.
    for (int pattern_idx = 0; pattern_idx < patterns.size(); pattern_idx++) {
      Add(Bytecode::MakeDup(node->matched()->span()));
    }

    for (int pattern_idx = 0; pattern_idx < patterns.size(); pattern_idx++) {
      // Then we match each arm. We OR with the prev. result (if there is one)
      // and swap to the next copy of the matchee, unless this is the last
      // pattern.
      NameDefTree* ndt = arm->patterns()[pattern_idx];
      absl::StatusOr<Bytecode::MatchArmItem> arm_item_or =
          HandleNameDefTreeExpr(ndt);
      if (!arm_item_or.ok()) {
        status_ = arm_item_or.status();
        return;
      }
      Add(Bytecode::MakeMatchArm(ndt->span(), arm_item_or.value()));

      if (pattern_idx != 0) {
        Add(Bytecode::MakeLogicalOr(ndt->span()));
      }
      if (pattern_idx != patterns.size() - 1) {
        Add(Bytecode::MakeSwap(ndt->span()));
      }
    }
    Add(Bytecode::MakeInvert(arm->span()));
    jumps_to_next.push_back(bytecode_.size());
    Add(Bytecode::MakeJumpRelIf(arm->span(), Bytecode::kPlaceholderJumpAmount));
    // Pop the value being matched since now we're going to produce the final
    // result.
    Add(Bytecode::MakePop(arm->span()));

    // The arm matched: calculate the resulting expression and jump
    // unconditionally to "done".
    arm->expr()->AcceptExpr(this);
    jumps_to_done.push_back(bytecode_.size());
    Add(Bytecode::MakeJumpRel(arm->span(), Bytecode::kPlaceholderJumpAmount));
  }

  // Finally, handle the case where no arm matched, which is a failure.
  // Theoretically, we could reduce bytecode size by omitting this when the last
  // arm is strictly wildcards, but it doesn't seem worth the effort.
  arm_offsets.push_back(bytecode_.size());
  Add(Bytecode::MakeJumpDest(node->span()));
  Bytecode::TraceData trace_data;
  trace_data.push_back("The value was not matched: value: ");
  trace_data.push_back(FormatPreference::kDefault);
  Add(Bytecode(node->span(), Bytecode::Op::kFail, trace_data));

  size_t done_offset = bytecode_.size();
  Add(Bytecode::MakeJumpDest(node->span()));

  // One extra for the fall-through failure case.
  XLS_CHECK_EQ(node->arms().size() + 1, arm_offsets.size());
  XLS_CHECK_EQ(node->arms().size(), jumps_to_next.size());
  for (size_t i = 0; i < jumps_to_next.size(); ++i) {
    size_t jump_offset = jumps_to_next[i];
    size_t next_offset = arm_offsets[i + 1];
    XLS_VLOG(5) << "Patching jump offset " << jump_offset
                << " to jump to next_offset " << next_offset;
    bytecode_.at(jump_offset).PatchJumpTarget(next_offset - jump_offset);
  }

  for (size_t offset : jumps_to_done) {
    XLS_VLOG(5) << "Patching jump-to-done offset " << offset
                << " to jump to done_offset " << done_offset;
    bytecode_.at(offset).PatchJumpTarget(done_offset - offset);
  }
}

}  // namespace xls::dslx
