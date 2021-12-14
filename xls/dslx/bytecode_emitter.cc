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

#include "absl/strings/str_split.h"
#include "xls/common/status/ret_check.h"
#include "xls/dslx/ast.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/number_parser.h"
#include "re2/re2.h"

namespace xls::dslx {
namespace {

std::string OpToString(Bytecode::Op op) {
  switch (op) {
    case Bytecode::Op::kAdd:
      return "add";
    case Bytecode::Op::kAnd:
      return "and";
    case Bytecode::Op::kCall:
      return "call";
    case Bytecode::Op::kConcat:
      return "concat";
    case Bytecode::Op::kCreateTuple:
      return "create_tuple";
    case Bytecode::Op::kDiv:
      return "div";
    case Bytecode::Op::kExpandTuple:
      return "expand_tuple";
    case Bytecode::Op::kEq:
      return "eq";
    case Bytecode::Op::kGe:
      return "ge";
    case Bytecode::Op::kGt:
      return "gt";
    case Bytecode::Op::kInvert:
      return "invert";
    case Bytecode::Op::kJumpRel:
      return "jump_rel";
    case Bytecode::Op::kJumpRelIf:
      return "jump_rel_if";
    case Bytecode::Op::kJumpDest:
      return "jump_dest";
    case Bytecode::Op::kLe:
      return "le";
    case Bytecode::Op::kLoad:
      return "load";
    case Bytecode::Op::kLt:
      return "lt";
    case Bytecode::Op::kLiteral:
      return "literal";
    case Bytecode::Op::kLogicalAnd:
      return "logical_and";
    case Bytecode::Op::kLogicalOr:
      return "logical_or";
    case Bytecode::Op::kMul:
      return "mul";
    case Bytecode::Op::kNe:
      return "ne";
    case Bytecode::Op::kNegate:
      return "negate";
    case Bytecode::Op::kOr:
      return "or";
    case Bytecode::Op::kShll:
      return "shl";
    case Bytecode::Op::kShrl:
      return "shr";
    case Bytecode::Op::kStore:
      return "store";
    case Bytecode::Op::kSub:
      return "sub";
    case Bytecode::Op::kXor:
      return "xor";
  }
  return absl::StrCat("<invalid: ", static_cast<int>(op), ">");
}

absl::StatusOr<Bytecode::Op> OpFromString(std::string_view s) {
  if (s == "add") {
    return Bytecode::Op::kAdd;
  }
  if (s == "call") {
    return Bytecode::Op::kCall;
  }
  if (s == "create_tuple") {
    return Bytecode::Op::kCreateTuple;
  }
  if (s == "expand_tuple") {
    return Bytecode::Op::kExpandTuple;
  }
  if (s == "eq") {
    return Bytecode::Op::kEq;
  }
  if (s == "invert") {
    return Bytecode::Op::kInvert;
  }
  if (s == "load") {
    return Bytecode::Op::kLoad;
  }
  if (s == "literal") {
    return Bytecode::Op::kLiteral;
  }
  if (s == "negate") {
    return Bytecode::Op::kNegate;
  }
  if (s == "store") {
    return Bytecode::Op::kStore;
  }
  if (s == "jump_rel") {
    return Bytecode::Op::kJumpRel;
  }
  if (s == "jump_rel_if") {
    return Bytecode::Op::kJumpRelIf;
  }
  if (s == "jump_dest") {
    return Bytecode::Op::kJumpDest;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("String was not a bytecode op: `", s, "`"));
}

}  // namespace

std::string BytecodesToString(absl::Span<const Bytecode> bytecodes,
                              bool source_locs) {
  std::string program;
  for (size_t i = 0; i < bytecodes.size(); ++i) {
    if (i != 0) {
      absl::StrAppend(&program, "\n");
    }
    absl::StrAppendFormat(&program, "%03d %s", i,
                          bytecodes.at(i).ToString(source_locs));
  }
  return program;
}

std::string Bytecode::ToString(bool source_locs) const {
  std::string op_string = OpToString(op_);
  std::string loc_string;
  if (source_locs) {
    loc_string = " @ " + source_span_.ToString();
  }

  if (op_ == Op::kJumpRel || op_ == Op::kJumpRelIf) {
    return absl::StrFormat("%s %+d%s", OpToString(op_),
                           absl::get<int64_t>(data_.value()), loc_string);
  }

  if (data_.has_value()) {
    std::string data_string;
    if (absl::holds_alternative<int64_t>(data_.value())) {
      int64_t data = absl::get<int64_t>(data_.value());
      data_string = absl::StrCat(data);
    } else {
      data_string = absl::get<InterpValue>(data_.value()).ToString();
    }
    return absl::StrFormat("%s %s%s", op_string, data_string, loc_string);
  }
  return absl::StrFormat("%s%s", op_string, loc_string);
}

static absl::StatusOr<InterpValue> ParseInterpValue(absl::string_view text) {
  std::vector<std::string_view> pieces =
      absl::StrSplit(text, absl::MaxSplits(':', 1));
  if (pieces.size() != 2) {
    return absl::UnimplementedError(absl::StrCat(
        "Could not find type annotation in InterpValue text: `", text, "`"));
  }
  std::string_view type = pieces.at(0);
  std::string_view data = pieces.at(1);
  char signedness;
  int64_t bit_count;
  if (RE2::FullMatch(type, "([us])(\\d+)", &signedness, &bit_count)) {
    bool is_signed = signedness == 's';
    XLS_ASSIGN_OR_RETURN(Bits bits, ParseNumber(data));
    XLS_RET_CHECK(bits.bit_count() <= bit_count);
    if (is_signed) {
      bits = bits_ops::SignExtend(bits, bit_count);
    } else {
      bits = bits_ops::ZeroExtend(bits, bit_count);
    }
    return InterpValue::MakeBits(
        is_signed ? InterpValueTag::kSBits : InterpValueTag::kUBits,
        std::move(bits));
  }
  return absl::UnimplementedError(
      absl::StrCat("Cannot parse literal value: `", text, "`"));
}

absl::StatusOr<std::vector<Bytecode>> BytecodesFromString(
    absl::string_view text) {
  std::vector<Bytecode> result;
  std::vector<std::string_view> lines = absl::StrSplit(text, '\n');
  for (std::string_view line : lines) {
    std::vector<std::string_view> pieces =
        absl::StrSplit(line, absl::MaxSplits(' ', 1));
    int64_t pc;
    if (!absl::SimpleAtoi(pieces.at(0), &pc)) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Could not parse PC at start of line: `%s`", line));
    }
    std::vector<std::string_view> op_and_rest =
        absl::StrSplit(pieces.at(1), absl::MaxSplits(' ', 1));
    std::string_view op_str = op_and_rest.at(0);
    XLS_ASSIGN_OR_RETURN(Bytecode::Op op, OpFromString(op_str));
    std::optional<Bytecode::Data> data;
    if (op_and_rest.size() > 1) {
      std::string_view value_str = op_and_rest.at(1);
      if (op == Bytecode::Op::kLiteral) {
        XLS_ASSIGN_OR_RETURN(data, ParseInterpValue(value_str));
      } else if (!value_str.empty()) {
        int64_t integer_data;
        if (!absl::SimpleAtoi(value_str, &integer_data)) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Could not parse data payload in line: `%s`", line));
        }
        data = integer_data;
      }
    }
    result.push_back(Bytecode(Span::Fake(), op, data));
  }
  return result;
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

void BytecodeEmitter::HandleUnop(Unop* node) {
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
                               static_cast<int64_t>(node->members().size())));
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
