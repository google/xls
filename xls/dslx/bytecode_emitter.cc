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
#include "absl/strings/str_format.h"
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
    const std::optional<SymbolicBindings>& caller_bindings)
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
                      const std::optional<SymbolicBindings>& caller_bindings) {
  return EmitProcNext(import_data, type_info, f, caller_bindings,
                      /*proc_members=*/{});
}

/* static */ absl::StatusOr<std::unique_ptr<BytecodeFunction>>
BytecodeEmitter::EmitProcNext(
    ImportData* import_data, const TypeInfo* type_info, const Function* f,
    const std::optional<SymbolicBindings>& caller_bindings,
    const std::vector<NameDef*>& proc_members) {
  BytecodeEmitter emitter(import_data, type_info, caller_bindings);
  for (const NameDef* name_def : proc_members) {
    emitter.namedef_to_slot_[name_def] = emitter.namedef_to_slot_.size();
  }
  XLS_RETURN_IF_ERROR(emitter.Init(f));
  XLS_RETURN_IF_ERROR(f->body()->AcceptExpr(&emitter));

  return BytecodeFunction::Create(f->owner(), f, type_info,
                                  std::move(emitter.bytecode_));
}

// Extracts all NameDefs "downstream" of a given AstNode. This
// is needed for Expr evaluation, so we can reserve slots for provided
// InterpValues (in namedef_to_slot_).
class NameDefCollector : public AstNodeVisitor {
 public:
  const absl::flat_hash_map<std::string, const NameDef*>& name_defs() {
    return name_defs_;
  }

#define DEFAULT_HANDLER(NODE)                                      \
  absl::Status Handle##NODE(const NODE* n) override {              \
    for (AstNode * child : n->GetChildren(/*want_types=*/false)) { \
      XLS_RETURN_IF_ERROR(child->Accept(this));                    \
    }                                                              \
    return absl::OkStatus();                                       \
  }

  DEFAULT_HANDLER(Array);
  DEFAULT_HANDLER(ArrayTypeAnnotation);
  absl::Status HandleAttr(const Attr* n) override {
    XLS_RETURN_IF_ERROR(n->lhs()->Accept(this));
    return absl::OkStatus();
  }
  DEFAULT_HANDLER(Binop);
  DEFAULT_HANDLER(Block);
  DEFAULT_HANDLER(BuiltinNameDef);
  DEFAULT_HANDLER(BuiltinTypeAnnotation);
  DEFAULT_HANDLER(Cast);
  DEFAULT_HANDLER(ChannelDecl);
  DEFAULT_HANDLER(ChannelTypeAnnotation);
  DEFAULT_HANDLER(ColonRef);
  DEFAULT_HANDLER(ConstantArray);
  DEFAULT_HANDLER(ConstantDef);
  absl::Status HandleConstRef(const ConstRef* n) override {
    return n->name_def()->Accept(this);
  }
  DEFAULT_HANDLER(EnumDef);
  DEFAULT_HANDLER(For);
  DEFAULT_HANDLER(FormatMacro);
  absl::Status HandleFunction(const Function* n) override {
    return absl::InternalError(
        absl::StrFormat(
            "Encountered nested Function: %s @ %s",
            n->identifier(), n->span().ToString()));
  }
  DEFAULT_HANDLER(Index);
  DEFAULT_HANDLER(Invocation);
  DEFAULT_HANDLER(Import);
  DEFAULT_HANDLER(Join);
  DEFAULT_HANDLER(Let);
  DEFAULT_HANDLER(Match);
  DEFAULT_HANDLER(MatchArm);
  DEFAULT_HANDLER(Module);
  absl::Status HandleNameDef(const NameDef* n) override {
    name_defs_[n->identifier()] = n;
    return absl::OkStatus();
  }
  DEFAULT_HANDLER(NameDefTree);
  absl::Status HandleNameRef(const NameRef* n) override {
    if (std::holds_alternative<const NameDef*>(n->name_def())) {
      return std::get<const NameDef*>(n->name_def())->Accept(this);
    }
    return absl::OkStatus();
  }
  DEFAULT_HANDLER(Number);
  DEFAULT_HANDLER(Param);
  DEFAULT_HANDLER(ParametricBinding);
  absl::Status HandleProc(const Proc* n) override {
    return absl::InternalError(
        absl::StrFormat(
            "Encountered nested Proc: %s @ %s",
            n->identifier(), n->span().ToString()));
  }
  DEFAULT_HANDLER(QuickCheck);
  DEFAULT_HANDLER(Range);
  DEFAULT_HANDLER(Recv);
  DEFAULT_HANDLER(RecvIf);
  DEFAULT_HANDLER(RecvIfNonBlocking);
  DEFAULT_HANDLER(RecvNonBlocking);
  DEFAULT_HANDLER(Send);
  DEFAULT_HANDLER(SendIf);
  DEFAULT_HANDLER(Slice);
  DEFAULT_HANDLER(Spawn);
  DEFAULT_HANDLER(SplatStructInstance);
  DEFAULT_HANDLER(String);
  DEFAULT_HANDLER(StructDef);
  absl::Status HandleTestFunction(const TestFunction* n) override {
    return absl::InternalError(
        absl::StrFormat(
            "Encountered nested TestFunction: %s @ %s",
            n->identifier(), n->GetSpan()->ToString()));
  }
  absl::Status HandleStructInstance(const StructInstance* n) override {
    for (const auto& member : n->GetUnorderedMembers()) {
      XLS_RETURN_IF_ERROR(member.second->Accept(this));
    }
    return absl::OkStatus();
  }
  DEFAULT_HANDLER(Ternary);
  DEFAULT_HANDLER(TestProc);
  DEFAULT_HANDLER(TupleIndex);
  DEFAULT_HANDLER(TupleTypeAnnotation);
  DEFAULT_HANDLER(TypeDef);
  DEFAULT_HANDLER(TypeRef);
  DEFAULT_HANDLER(TypeRefTypeAnnotation);
  DEFAULT_HANDLER(Unop);
  DEFAULT_HANDLER(UnrollFor);
  DEFAULT_HANDLER(WidthSlice);
  DEFAULT_HANDLER(WildcardPattern);
  DEFAULT_HANDLER(XlsTuple);

 private:
  absl::flat_hash_map<std::string, const NameDef*> name_defs_;
};

/* static */ absl::StatusOr<std::unique_ptr<BytecodeFunction>>
BytecodeEmitter::EmitExpression(
    ImportData* import_data, const TypeInfo* type_info, const Expr* expr,
    const absl::flat_hash_map<std::string, InterpValue>& env,
    const std::optional<SymbolicBindings>& caller_bindings) {
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

  XLS_RETURN_IF_ERROR(expr->AcceptExpr(&emitter));

  return BytecodeFunction::Create(expr->owner(), /*source_fn=*/nullptr,
                                  type_info, std::move(emitter.bytecode_));
}

absl::Status BytecodeEmitter::HandleArray(const Array* node) {
  int num_members = node->members().size();
  for (auto* member : node->members()) {
    XLS_RETURN_IF_ERROR(member->AcceptExpr(this));
  }

  // If we've got an ellipsis, then repeat the last element until we reach the
  // full array size.
  if (node->has_ellipsis()) {
    XLS_ASSIGN_OR_RETURN(ArrayType * array_type,
                         type_info_->GetItemAs<ArrayType>(node));
    const ConcreteTypeDim& dim = array_type->size();
    XLS_ASSIGN_OR_RETURN(num_members, dim.GetAsInt64());
    int64_t remaining_members = num_members - node->members().size();
    for (int i = 0; i < remaining_members; i++) {
      XLS_RETURN_IF_ERROR(node->members().back()->AcceptExpr(this));
    }
  }

  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kCreateArray,
                               Bytecode::NumElements(num_members)));
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleAttr(const Attr* node) {
  // Will place a struct instance on the stack.
  XLS_RETURN_IF_ERROR(node->lhs()->AcceptExpr(this));

  // Now we need the index of the attr NameRef in the struct def.
  XLS_ASSIGN_OR_RETURN(StructType * struct_type,
                       type_info_->GetItemAs<StructType>(node->lhs()));
  XLS_ASSIGN_OR_RETURN(int64_t member_index,
                       struct_type->GetMemberIndex(node->attr()->identifier()));

  // This indexing literal needs to be unsigned since InterpValue::Index
  // requires an unsigned value.
  Add(Bytecode::MakeLiteral(node->span(), InterpValue::MakeU64(member_index)));
  Add(Bytecode::MakeIndex(node->span()));
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleBinop(const Binop* node) {
  XLS_RETURN_IF_ERROR(node->lhs()->AcceptExpr(this));
  XLS_RETURN_IF_ERROR(node->rhs()->AcceptExpr(this));
  switch (node->binop_kind()) {
    case BinopKind::kAdd:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kAdd));
      break;
    case BinopKind::kAnd:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kAnd));
      break;
    case BinopKind::kConcat:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kConcat));
      break;
    case BinopKind::kDiv:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kDiv));
      break;
    case BinopKind::kEq:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kEq));
      break;
    case BinopKind::kGe:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kGe));
      break;
    case BinopKind::kGt:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kGt));
      break;
    case BinopKind::kLe:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kLe));
      break;
    case BinopKind::kLogicalAnd:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kLogicalAnd));
      break;
    case BinopKind::kLogicalOr:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kLogicalOr));
      break;
    case BinopKind::kLt:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kLt));
      break;
    case BinopKind::kMul:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kMul));
      break;
    case BinopKind::kNe:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kNe));
      break;
    case BinopKind::kOr:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kOr));
      break;
    case BinopKind::kShl:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kShl));
      break;
    case BinopKind::kShr:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kShr));
      break;
    case BinopKind::kSub:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kSub));
      break;
    case BinopKind::kXor:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kXor));
      break;
    default:
      return absl::UnimplementedError(
          absl::StrCat("Unimplemented binary operator: ",
                       BinopKindToString(node->binop_kind())));
  }
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleBlock(const Block* node) {
  return node->body()->AcceptExpr(this);
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

absl::Status BytecodeEmitter::HandleCast(const Cast* node) {
  XLS_RETURN_IF_ERROR(node->expr()->AcceptExpr(this));

  std::optional<ConcreteType*> maybe_from = type_info_->GetItem(node->expr());
  if (!maybe_from.has_value()) {
    return absl::InternalError(
        absl::StrCat("Could not find type for cast \"from\" arg: ",
                     node->expr()->ToString()));
  }
  ConcreteType* from = maybe_from.value();

  std::optional<ConcreteType*> maybe_to =
      type_info_->GetItem(node->type_annotation());
  if (!maybe_to.has_value()) {
    return absl::InternalError(absl::StrCat(
        "Could not find concrete type for cast \"to\" type: ",
        node->type_annotation()->ToString(), " : ", node->span().ToString()));
  }
  ConcreteType* to = maybe_to.value();

  if (ArrayType* from_array = dynamic_cast<ArrayType*>(from);
      from_array != nullptr) {
    BitsType* to_bits = dynamic_cast<BitsType*>(to);
    if (to_bits == nullptr) {
      return absl::InternalError(absl::StrCat(
          "The only valid array cast is to bits: ", node->ToString()));
    }

    return CastArrayToBits(node->span(), from_array, to_bits);
  }

  if (EnumType* from_enum = dynamic_cast<EnumType*>(from);
      from_enum != nullptr) {
    BitsType* to_bits = dynamic_cast<BitsType*>(to);
    if (to_bits == nullptr) {
      return absl::InternalError(absl::StrCat(
          "The only valid enum cast is to bits: ", node->ToString()));
    }

    bytecode_.push_back(
        Bytecode(node->span(), Bytecode::Op::kCast, to_bits->CloneToUnique()));
    return absl::OkStatus();
  }

  BitsType* from_bits = dynamic_cast<BitsType*>(from);
  if (from_bits == nullptr) {
    return absl::InternalError(
        "Only casts from arrays, enums, or bits are allowed.");
  }

  if (ArrayType* to_array = dynamic_cast<ArrayType*>(to); to_array != nullptr) {
    return CastBitsToArray(node->span(), from_bits, to_array);
  }

  if (EnumType* to_enum = dynamic_cast<EnumType*>(to); to_enum != nullptr) {
    bytecode_.push_back(
        Bytecode(node->span(), Bytecode::Op::kCast, to_enum->CloneToUnique()));
    return absl::OkStatus();
  }

  BitsType* to_bits = dynamic_cast<BitsType*>(to);
  if (to_bits == nullptr) {
    return absl::InternalError(
        "Only casts to arrays, enums, or bits are allowed.");
  }

  bytecode_.push_back(
      Bytecode(node->span(), Bytecode::Op::kCast, to_bits->CloneToUnique()));
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleChannelDecl(const ChannelDecl* node) {
  // Channels are created as constexpr values during type deduction/constexpr
  // evaluation, since they're concrete values that need to be shared amongst
  // two actors.
  XLS_ASSIGN_OR_RETURN(InterpValue channel, type_info_->GetConstExpr(node));
  Add(Bytecode::MakeLiteral(node->span(), channel));
  return absl::OkStatus();
}

absl::StatusOr<InterpValue> BytecodeEmitter::HandleColonRefToEnum(
    const ColonRef* colon_ref, EnumDef* enum_def, const TypeInfo* type_info) {
  // TODO(rspringer): 2022-01-26 We'll need to pull the right type info during
  // ResolveTypeDefToEnum.
  std::string_view attr = colon_ref->attr();
  XLS_ASSIGN_OR_RETURN(Expr * value_expr, enum_def->GetValue(attr));
  return type_info->GetConstExpr(value_expr);
}

absl::StatusOr<InterpValue> BytecodeEmitter::HandleColonRefToValue(
    Module* module, const ColonRef* colon_ref) {
  // TODO(rspringer): We'll need subject resolution to return the appropriate
  // TypeInfo for parametrics.
  XLS_ASSIGN_OR_RETURN(TypeInfo * type_info,
                       import_data_->GetRootTypeInfo(module));

  std::optional<ModuleMember*> member =
      module->FindMemberWithName(colon_ref->attr());
  if (!member.has_value()) {
    return absl::InternalError(
        absl::StrFormat("Could not find member %s of module %s.",
                        colon_ref->attr(), module->name()));
  }

  if (std::holds_alternative<Function*>(*member.value())) {
    Function* f = std::get<Function*>(*member.value());
    return InterpValue::MakeFunction(InterpValue::UserFnData{module, f});
  }

  XLS_RET_CHECK(std::holds_alternative<ConstantDef*>(*member.value()));
  ConstantDef* constant_def = std::get<ConstantDef*>(*member.value());
  return type_info->GetConstExpr(constant_def->value());
}

absl::Status BytecodeEmitter::HandleColonRef(const ColonRef* node) {
  XLS_ASSIGN_OR_RETURN(InterpValue value, HandleColonRefInternal(node));

  Add(Bytecode::MakeLiteral(node->span(), value));
  return absl::OkStatus();
}

absl::StatusOr<InterpValue> BytecodeEmitter::HandleColonRefInternal(
    const ColonRef* node) {
  XLS_ASSIGN_OR_RETURN(auto resolved_subject,
                       ResolveColonRefSubject(import_data_, type_info_, node));

  return absl::visit(
      Visitor{
          [&](EnumDef* enum_def) -> absl::StatusOr<InterpValue> {
            const TypeInfo* type_info = type_info_;
            if (enum_def->owner() != type_info_->module()) {
              type_info =
                  import_data_->GetRootTypeInfo(enum_def->owner()).value();
            }
            return HandleColonRefToEnum(node, enum_def, type_info);
          },
          [&](BuiltinNameDef* builtin_name_def) -> absl::StatusOr<InterpValue> {
            return GetBuiltinNameDefColonAttr(builtin_name_def, node->attr());
          },
          [&](ArrayTypeAnnotation* array_type) -> absl::StatusOr<InterpValue> {
            XLS_ASSIGN_OR_RETURN(
                TypeInfo * type_info,
                import_data_->GetRootTypeInfoForNode(array_type));
            XLS_ASSIGN_OR_RETURN(InterpValue value,
                                 type_info->GetConstExpr(array_type->dim()));
            XLS_ASSIGN_OR_RETURN(uint64_t dim_u64, value.GetBitValueUint64());
            return GetArrayTypeColonAttr(array_type, dim_u64, node->attr());
          },
          [&](Module* module) -> absl::StatusOr<InterpValue> {
            return HandleColonRefToValue(module, node);
          }},
      resolved_subject);
}

absl::Status BytecodeEmitter::HandleConstantArray(const ConstantArray* node) {
  return HandleArray(node);
}

absl::Status BytecodeEmitter::HandleConstRef(const ConstRef* node) {
  return HandleNameRef(node);
}

absl::Status BytecodeEmitter::HandleFor(const For* node) {
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
  std::optional<ConcreteType*> maybe_iterable_type =
      type_info_->GetItem(node->iterable());
  if (!maybe_iterable_type.has_value()) {
    return absl::InternalError(
        absl::StrCat("Could not find concrete type for loop iterable: ",
                     node->iterable()->ToString()));
  }

  ArrayType* array_type = dynamic_cast<ArrayType*>(maybe_iterable_type.value());
  if (array_type == nullptr) {
    return absl::InternalError(absl::StrCat("Iterable was not of array type: ",
                                            node->iterable()->ToString()));
  }

  ConcreteTypeDim iterable_size_dim = array_type->size();
  XLS_ASSIGN_OR_RETURN(int64_t iterable_size, iterable_size_dim.GetAsInt64());

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
  const NameDef* fake_name_def =
      reinterpret_cast<const NameDef*>(node->iterable());
  namedef_to_slot_[fake_name_def] = iterable_slot;
  XLS_RETURN_IF_ERROR(node->iterable()->AcceptExpr(this));
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kStore,
                               Bytecode::SlotIndex(iterable_slot)));

  int index_slot = namedef_to_slot_.size();
  fake_name_def = reinterpret_cast<const NameDef*>(node);
  namedef_to_slot_[fake_name_def] = index_slot;
  bytecode_.push_back(
      Bytecode(node->span(), Bytecode::Op::kLiteral, InterpValue::MakeU32(0)));
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kStore,
                               Bytecode::SlotIndex(index_slot)));

  // Evaluate the initial accumulator value & leave it on the stack.
  XLS_RETURN_IF_ERROR(node->init()->AcceptExpr(this));

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
  Add(Bytecode::MakeIndex(node->span()));
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kSwap));
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kCreateTuple,
                               Bytecode::NumElements(2)));
  DestructureLet(node->names());

  // Emit the loop body.
  XLS_RETURN_IF_ERROR(node->body()->AcceptExpr(this));

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
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleFormatMacro(const FormatMacro* node) {
  for (auto* arg : node->args()) {
    XLS_RETURN_IF_ERROR(arg->AcceptExpr(this));
  }

  bytecode_.push_back(
      Bytecode(node->span(), Bytecode::Op::kTrace, node->format()));
  return absl::OkStatus();
}

absl::StatusOr<int64_t> GetValueWidth(const TypeInfo* type_info, Expr* expr) {
  std::optional<ConcreteType*> maybe_type = type_info->GetItem(expr);
  if (!maybe_type.has_value()) {
    return absl::InternalError(
        "Could not find concrete type for slice component.");
  }
  return maybe_type.value()->GetTotalBitCount()->GetAsInt64();
}

absl::Status BytecodeEmitter::HandleIndex(const Index* node) {
  XLS_RETURN_IF_ERROR(node->lhs()->AcceptExpr(this));

  if (std::holds_alternative<Slice*>(node->rhs())) {
    Slice* slice = std::get<Slice*>(node->rhs());
    if (slice->start() == nullptr) {
      int64_t start_width;
      if (slice->limit() == nullptr) {
        // TODO(rspringer): Define a uniform `usize` to avoid specifying magic
        // numbers here. This is the default size used for untyped numbers in
        // the typechecker.
        start_width = 32;
      } else {
        XLS_ASSIGN_OR_RETURN(start_width,
                             GetValueWidth(type_info_, slice->limit()));
      }
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kLiteral,
                                   InterpValue::MakeSBits(start_width, 0)));
    } else {
      XLS_RETURN_IF_ERROR(slice->start()->AcceptExpr(this));
    }

    if (slice->limit() == nullptr) {
      std::optional<ConcreteType*> maybe_type =
          type_info_->GetItem(node->lhs());
      if (!maybe_type.has_value()) {
        return absl::InternalError("Could not find concrete type for slice.");
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
        XLS_ASSIGN_OR_RETURN(limit_width,
                             GetValueWidth(type_info_, slice->start()));
      }
      bytecode_.push_back(
          Bytecode(node->span(), Bytecode::Op::kLiteral,
                   InterpValue::MakeSBits(limit_width, width_or.value())));
    } else {
      XLS_RETURN_IF_ERROR(slice->limit()->AcceptExpr(this));
    }
    bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kSlice));
    return absl::OkStatus();
  }

  if (std::holds_alternative<WidthSlice*>(node->rhs())) {
    WidthSlice* width_slice = std::get<WidthSlice*>(node->rhs());
    XLS_RETURN_IF_ERROR(width_slice->start()->AcceptExpr(this));

    std::optional<ConcreteType*> maybe_type =
        type_info_->GetItem(width_slice->width());
    if (!maybe_type.has_value()) {
      return absl::InternalError(absl::StrCat(
          "Could not find concrete type for slice width parameter \"",
          width_slice->width()->ToString(), "\"."));
    }

    ConcreteType* type = maybe_type.value();
    BitsType* bits_type = dynamic_cast<BitsType*>(type);
    if (bits_type == nullptr) {
      return absl::InternalError(absl::StrCat(
          "Width slice type specifier isn't a BitsType: ", type->ToString()));
    }

    bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kWidthSlice,
                                 type->CloneToUnique()));
    return absl::OkStatus();
  }

  // Otherwise, it's a regular [array or tuple] index op.
  Expr* expr = std::get<Expr*>(node->rhs());
  XLS_RETURN_IF_ERROR(expr->AcceptExpr(this));
  Add(Bytecode::MakeIndex(node->span()));
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleInvocation(const Invocation* node) {
  if (NameRef* name_ref = dynamic_cast<NameRef*>(node->callee());
      name_ref != nullptr) {
    if (name_ref->identifier() == "trace!") {
      if (node->args().size() != 1) {
        return absl::InternalError("`trace!` takes a single argument.");
      }

      XLS_RETURN_IF_ERROR(node->args().at(0)->AcceptExpr(this));

      Bytecode::TraceData trace_data;
      trace_data.push_back(absl::StrCat("trace of ",
                                        node->args()[0]->ToString(), " @ ",
                                        node->span().ToString(), ": "));
      trace_data.push_back(FormatPreference::kDefault);
      bytecode_.push_back(
          Bytecode(node->span(), Bytecode::Op::kTrace, trace_data));
      return absl::OkStatus();
    }
  }

  for (auto* arg : node->args()) {
    XLS_RETURN_IF_ERROR(arg->AcceptExpr(this));
  }

  XLS_RETURN_IF_ERROR(node->callee()->AcceptExpr(this));

  std::optional<const SymbolicBindings*> maybe_callee_bindings =
      type_info_->GetInvocationCalleeBindings(
          node, caller_bindings_.has_value() ? caller_bindings_.value()
                                             : SymbolicBindings());
  std::optional<SymbolicBindings> final_bindings = absl::nullopt;
  if (maybe_callee_bindings.has_value()) {
    final_bindings = *maybe_callee_bindings.value();
  }
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kCall,
                               Bytecode::InvocationData{node, final_bindings}));
  return absl::OkStatus();
}

absl::StatusOr<Bytecode::MatchArmItem> BytecodeEmitter::HandleNameDefTreeExpr(
    NameDefTree* tree) {
  if (tree->is_leaf()) {
    return absl::visit(
        Visitor{
            [&](NameRef* n) -> absl::StatusOr<Bytecode::MatchArmItem> {
              using ValueT = std::variant<InterpValue, Bytecode::SlotIndex>;
              XLS_ASSIGN_OR_RETURN(ValueT item_value, HandleNameRefInternal(n));
              if (std::holds_alternative<InterpValue>(item_value)) {
                return Bytecode::MatchArmItem::MakeInterpValue(
                    std::get<InterpValue>(item_value));
              }

              return Bytecode::MatchArmItem::MakeLoad(
                  std::get<Bytecode::SlotIndex>(item_value));
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
    if (std::holds_alternative<WildcardPattern*>(tree->leaf())) {
      Add(Bytecode::MakePop(tree->span()));
      // We can just drop this one.
      return;
    }

    NameDef* name_def = std::get<NameDef*>(tree->leaf());
    if (!namedef_to_slot_.contains(name_def)) {
      namedef_to_slot_.insert({name_def, namedef_to_slot_.size()});
    }
    int64_t slot = namedef_to_slot_.at(name_def);
    Add(Bytecode::MakeStore(tree->span(), Bytecode::SlotIndex(slot)));
  } else {
    Add(Bytecode(tree->span(), Bytecode::Op::kExpandTuple));
    for (const auto& node : tree->nodes()) {
      DestructureLet(node);
    }
  }
}

absl::Status BytecodeEmitter::HandleJoin(const Join* node) {
  // Since we serially execute top-to-bottom, every node is an implicit join.
  Add(Bytecode::MakeLiteral(node->span(), InterpValue::MakeToken()));
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleLet(const Let* node) {
  XLS_RETURN_IF_ERROR(node->rhs()->AcceptExpr(this));
  DestructureLet(node->name_def_tree());
  return node->body()->AcceptExpr(this);
}

absl::Status BytecodeEmitter::HandleNameRef(const NameRef* node) {
  XLS_ASSIGN_OR_RETURN(auto result, HandleNameRefInternal(node));
  if (std::holds_alternative<InterpValue>(result)) {
    Add(Bytecode::MakeLiteral(node->span(), std::get<InterpValue>(result)));
  } else {
    Add(Bytecode::MakeLoad(node->span(),
                           std::get<Bytecode::SlotIndex>(result)));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::variant<InterpValue, Bytecode::SlotIndex>>
BytecodeEmitter::HandleNameRefInternal(const NameRef* node) {
  if (std::holds_alternative<BuiltinNameDef*>(node->name_def())) {
    // Builtins don't have NameDefs, so we can't use slots to store them. It's
    // simpler to just emit a literal InterpValue, anyway.
    BuiltinNameDef* builtin_def = std::get<BuiltinNameDef*>(node->name_def());
    XLS_ASSIGN_OR_RETURN(Builtin builtin,
                         BuiltinFromString(builtin_def->identifier()));
    return InterpValue::MakeFunction(builtin);
  }

  // Emit function and constant refs directly so that they can be stack elements
  // without having to load slots with them.
  const NameDef* name_def = std::get<const NameDef*>(node->name_def());
  if (auto* f = dynamic_cast<Function*>(name_def->definer()); f != nullptr) {
    return InterpValue::MakeFunction(InterpValue::UserFnData{f->owner(), f});
  }

  if (auto* cd = dynamic_cast<ConstantDef*>(name_def->definer());
      cd != nullptr) {
    return type_info_->GetConstExpr(cd->value());
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
      "Could not find slot or binding for name: ", name_def->ToString(), " @ ",
      name_def->span().ToString()));
}

absl::Status BytecodeEmitter::HandleNumber(const Number* node) {
  XLS_ASSIGN_OR_RETURN(InterpValue value, HandleNumberInternal(node));
  Add(Bytecode::MakeLiteral(node->span(), value));
  return absl::OkStatus();
}

absl::StatusOr<InterpValue> BytecodeEmitter::HandleNumberInternal(
    const Number* node) {
  std::optional<ConcreteType*> type_or = type_info_->GetItem(node);
  if (!type_or.has_value()) {
    return absl::InternalError(
        absl::StrCat("Could not find type for number: ", node->ToString()));
  }

  const ConcreteTypeDim* dim = nullptr;
  bool is_signed = false;
  if (auto* bits_type = dynamic_cast<BitsType*>(type_or.value());
      bits_type != nullptr) {
    dim = &bits_type->size();
    is_signed = bits_type->is_signed();
  } else if (auto* enum_type = dynamic_cast<EnumType*>(type_or.value());
             enum_type != nullptr) {
    dim = &enum_type->size();
    is_signed = enum_type->signedness();
  }

  XLS_RET_CHECK(dim != nullptr) << absl::StrCat(
      "Error in type deduction; number \"", node->ToString(),
      "\" did not have bits or enum type: ", type_or.value()->ToString(), ".");

  XLS_ASSIGN_OR_RETURN(int64_t dim_val, dim->GetAsInt64());
  XLS_ASSIGN_OR_RETURN(Bits bits, node->GetBits(dim_val));
  return InterpValue::MakeBits(is_signed, bits);
}

absl::Status BytecodeEmitter::HandleRange(const Range* node) {
  XLS_RETURN_IF_ERROR(node->start()->AcceptExpr(this));
  XLS_RETURN_IF_ERROR(node->end()->AcceptExpr(this));
  Add(Bytecode::MakeRange(node->span()));
  return absl::OkStatus();
}

namespace {

// Find concrete type of channel's payload.
absl::StatusOr<std::unique_ptr<ConcreteType>> GetChannelPayloadType(
    const TypeInfo* type_info, const Expr* channel) {
  std::optional<ConcreteType*> type = type_info->GetItem(channel);

  if (!type.has_value()) {
    return absl::InternalError(absl::StrFormat(
        "Could not retrieve type of channel %s", channel->ToString()));
  }

  ChannelType* channel_type = dynamic_cast<ChannelType*>(type.value());
  if (channel_type == nullptr) {
    return absl::InternalError(absl::StrFormat(
        "Channel %s type is not of type channel", channel->ToString()));
  }

  return channel_type->payload_type().CloneToUnique();
}

}  // namespace

absl::Status BytecodeEmitter::HandleRecv(const Recv* node) {
  XLS_RETURN_IF_ERROR(node->token()->AcceptExpr(this));
  XLS_RETURN_IF_ERROR(node->channel()->AcceptExpr(this));
  Add(Bytecode::MakeLiteral(node->span(), InterpValue::MakeUBits(1, 1)));

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> channel_payload_type,
                       GetChannelPayloadType(type_info_, node->channel()));
  Add(Bytecode::MakeRecv(node->span(), std::move(channel_payload_type)));
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleRecvNonBlocking(
    const RecvNonBlocking* node) {
  XLS_RETURN_IF_ERROR(node->token()->AcceptExpr(this));
  XLS_RETURN_IF_ERROR(node->channel()->AcceptExpr(this));
  Add(Bytecode::MakeLiteral(node->span(), InterpValue::MakeUBits(1, 1)));

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> channel_payload_type,
                       GetChannelPayloadType(type_info_, node->channel()));
  Add(Bytecode::MakeRecvNonBlocking(node->span(),
                                    std::move(channel_payload_type)));
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleRecvIf(const RecvIf* node) {
  XLS_RETURN_IF_ERROR(node->token()->AcceptExpr(this));
  XLS_RETURN_IF_ERROR(node->channel()->AcceptExpr(this));
  XLS_RETURN_IF_ERROR(node->condition()->AcceptExpr(this));

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> channel_payload_type,
                       GetChannelPayloadType(type_info_, node->channel()));
  Add(Bytecode::MakeRecv(node->span(), std::move(channel_payload_type)));
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleRecvIfNonBlocking(
    const RecvIfNonBlocking* node) {
  XLS_RETURN_IF_ERROR(node->token()->AcceptExpr(this));
  XLS_RETURN_IF_ERROR(node->channel()->AcceptExpr(this));
  XLS_RETURN_IF_ERROR(node->condition()->AcceptExpr(this));

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ConcreteType> channel_payload_type,
                       GetChannelPayloadType(type_info_, node->channel()));
  Add(Bytecode::MakeRecvNonBlocking(node->span(),
                                    std::move(channel_payload_type)));
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleSend(const Send* node) {
  XLS_RETURN_IF_ERROR(node->token()->AcceptExpr(this));
  XLS_RETURN_IF_ERROR(node->channel()->AcceptExpr(this));
  XLS_RETURN_IF_ERROR(node->payload()->AcceptExpr(this));
  Add(Bytecode::MakeLiteral(node->span(), InterpValue::MakeUBits(1, 1)));
  Add(Bytecode::MakeSend(node->span()));
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleSendIf(const SendIf* node) {
  XLS_RETURN_IF_ERROR(node->token()->AcceptExpr(this));
  XLS_RETURN_IF_ERROR(node->channel()->AcceptExpr(this));
  XLS_RETURN_IF_ERROR(node->payload()->AcceptExpr(this));
  XLS_RETURN_IF_ERROR(node->condition()->AcceptExpr(this));
  Add(Bytecode::MakeSend(node->span()));
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleSpawn(const Spawn* node) {
  XLS_ASSIGN_OR_RETURN(Proc * proc, ResolveProc(node->callee(), type_info_));

  auto convert_args = [this](const absl::Span<Expr* const> args)
      -> absl::StatusOr<std::vector<InterpValue>> {
    std::vector<InterpValue> arg_values;
    arg_values.reserve(args.size());
    for (const auto* arg : args) {
      XLS_ASSIGN_OR_RETURN(InterpValue arg_value,
                           type_info_->GetConstExpr(arg));
      arg_values.push_back(arg_value);
    }
    return arg_values;
  };

  XLS_ASSIGN_OR_RETURN(std::vector<InterpValue> config_args,
                       convert_args(node->config()->args()));
  XLS_RET_CHECK_EQ(node->next()->args().size(), 1);
  XLS_ASSIGN_OR_RETURN(InterpValue initial_state,
                       type_info_->GetConstExpr(node->next()->args()[0]));

  // The whole Proc is parameterized, not the individual invocations
  // (config/next), so we can use either invocation to get the bindings.
  std::optional<const SymbolicBindings*> maybe_callee_bindings =
      type_info_->GetInvocationCalleeBindings(node->config(),
                                                 caller_bindings_.has_value()
                                                     ? caller_bindings_.value()
                                                     : SymbolicBindings());
  std::optional<SymbolicBindings> final_bindings = absl::nullopt;
  if (maybe_callee_bindings.has_value()) {
    final_bindings = *maybe_callee_bindings.value();
  }

  Bytecode::SpawnData spawn_data{node, proc, config_args, initial_state,
                                 final_bindings};
  Add(Bytecode::MakeSpawn(node->span(), spawn_data));
  return node->body()->AcceptExpr(this);
}

absl::Status BytecodeEmitter::HandleString(const String* node) {
  // A string is just a fancy array literal.
  for (const unsigned char c : node->text()) {
    bytecode_.push_back(
        Bytecode(node->span(), Bytecode::Op::kLiteral,
                 InterpValue::MakeUBits(/*bit_count=*/8, static_cast<int>(c))));
  }

  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kCreateArray,
                               Bytecode::NumElements(node->text().size())));
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleStructInstance(const StructInstance* node) {
  XLS_ASSIGN_OR_RETURN(StructType * struct_type,
                       type_info_->GetItemAs<StructType>(node));

  const StructDef& struct_def = struct_type->nominal_type();
  for (const std::pair<std::string, Expr*>& member :
       node->GetOrderedMembers(&struct_def)) {
    XLS_RETURN_IF_ERROR(member.second->AcceptExpr(this));
  }

  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kCreateTuple,
                               Bytecode::NumElements(struct_def.size())));
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleSplatStructInstance(
    const SplatStructInstance* node) {
  // For each field in the struct:
  //   If it's specified in the SplatStructInstance, then use it,
  //   otherwise extract it from the base struct.
  XLS_ASSIGN_OR_RETURN(StructType * struct_type,
                       type_info_->GetItemAs<StructType>(node));

  XLS_ASSIGN_OR_RETURN(std::vector<std::string> member_names,
                       struct_type->GetMemberNames());

  absl::flat_hash_map<std::string, Expr*> new_members;
  for (const std::pair<std::string, Expr*>& new_member : node->members()) {
    new_members[new_member.first] = new_member.second;
  }

  // To extract values from the base struct, we need to visit it and extract the
  // appropriate attr (i.e., index into the representative tuple).
  // Referencing a base struct doesn't have side effects, so we can just
  // visit it as often as we need.

  // Member names is ordered.
  for (int i = 0; i < member_names.size(); i++) {
    std::string& member_name = member_names.at(i);
    if (new_members.contains(member_name)) {
      XLS_RETURN_IF_ERROR(new_members.at(member_name)->AcceptExpr(this));
    } else {
      XLS_RETURN_IF_ERROR(node->splatted()->AcceptExpr(this));
      Add(Bytecode::MakeLiteral(node->span(), InterpValue::MakeU64(i)));
      Add(Bytecode::MakeIndex(node->span()));
    }
  }

  const StructDef& struct_def = struct_type->nominal_type();
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kCreateTuple,
                               Bytecode::NumElements(struct_def.size())));
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleTupleIndex(const TupleIndex* node) {
  XLS_RETURN_IF_ERROR(node->lhs()->AcceptExpr(this));
  XLS_RETURN_IF_ERROR(node->index()->AcceptExpr(this));
  Add(Bytecode::MakeIndex(node->span()));
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleUnop(const Unop* node) {
  XLS_RETURN_IF_ERROR(node->operand()->AcceptExpr(this));
  switch (node->unop_kind()) {
    case UnopKind::kInvert:
      Add(Bytecode::MakeInvert(node->span()));
      break;
    case UnopKind::kNegate:
      bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kNegate));
      break;
  }
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleXlsTuple(const XlsTuple* node) {
  for (auto* member : node->members()) {
    XLS_RETURN_IF_ERROR(member->AcceptExpr(this));
  }

  // Pop the N elements and push the result as a single value.
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kCreateTuple,
                               Bytecode::NumElements(node->members().size())));
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleTernary(const Ternary* node) {
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
  XLS_RETURN_IF_ERROR(node->test()->AcceptExpr(this));
  size_t jump_if_index = bytecode_.size();
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kJumpRelIf,
                               Bytecode::kPlaceholderJumpAmount));
  XLS_RETURN_IF_ERROR(node->alternate()->AcceptExpr(this));
  size_t jump_index = bytecode_.size();
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kJumpRel,
                               Bytecode::kPlaceholderJumpAmount));
  size_t consequent_index = bytecode_.size();
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kJumpDest));
  XLS_RETURN_IF_ERROR(node->consequent()->AcceptExpr(this));
  size_t jumpdest_index = bytecode_.size();

  // Now patch up the jumps since we now know the relative PCs we're jumping to.
  // TODO(leary): 2021-12-10 Keep track of the stack depth so we can validate it
  // at the jump destination.
  bytecode_.push_back(Bytecode(node->span(), Bytecode::Op::kJumpDest));
  bytecode_.at(jump_if_index).PatchJumpTarget(consequent_index - jump_if_index);
  bytecode_.at(jump_index).PatchJumpTarget(jumpdest_index - jump_index);
  return absl::OkStatus();
}

absl::Status BytecodeEmitter::HandleUnrollFor(const UnrollFor* node) {
  return absl::UnimplementedError(
      "UnrollFor nodes aren't interpretable/emittable. "
      "Such nodes should have been unrolled into flat statements.");
}

absl::Status BytecodeEmitter::HandleMatch(const Match* node) {
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
  XLS_RETURN_IF_ERROR(node->matched()->AcceptExpr(this));

  const std::vector<MatchArm*>& arms = node->arms();
  if (arms.empty()) {
    return absl::InternalError("At least one match arm is required.");
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
      XLS_ASSIGN_OR_RETURN(Bytecode::MatchArmItem arm_item,
                           HandleNameDefTreeExpr(ndt));
      Add(Bytecode::MakeMatchArm(ndt->span(), arm_item));

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
    XLS_RETURN_IF_ERROR(arm->expr()->AcceptExpr(this));
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

  return absl::OkStatus();
}

}  // namespace xls::dslx
