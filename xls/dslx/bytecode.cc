// Copyright 2022 The XLS Authors
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

#include "xls/dslx/bytecode.h"

#include <memory>

#include "absl/strings/str_split.h"
#include "absl/types/variant.h"
#include "xls/common/status/ret_check.h"
#include "xls/dslx/ast.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/number_parser.h"
#include "re2/re2.h"

namespace xls::dslx {
namespace {

absl::StatusOr<Bytecode::Op> OpFromString(std::string_view s) {
  if (s == "add") {
    return Bytecode::Op::kAdd;
  }
  if (s == "and") {
    return Bytecode::Op::kAnd;
  }
  if (s == "call") {
    return Bytecode::Op::kCall;
  }
  if (s == "cast") {
    return Bytecode::Op::kCast;
  }
  if (s == "concat") {
    return Bytecode::Op::kConcat;
  }
  if (s == "create_array") {
    return Bytecode::Op::kCreateArray;
  }
  if (s == "create_tuple") {
    return Bytecode::Op::kCreateTuple;
  }
  if (s == "div") {
    return Bytecode::Op::kDiv;
  }
  if (s == "dup") {
    return Bytecode::Op::kDup;
  }
  if (s == "expand_tuple") {
    return Bytecode::Op::kExpandTuple;
  }
  if (s == "eq") {
    return Bytecode::Op::kEq;
  }
  if (s == "fail") {
    return Bytecode::Op::kFail;
  }
  if (s == "ge") {
    return Bytecode::Op::kGe;
  }
  if (s == "gt") {
    return Bytecode::Op::kGt;
  }
  if (s == "index") {
    return Bytecode::Op::kIndex;
  }
  if (s == "invert") {
    return Bytecode::Op::kInvert;
  }
  if (s == "jump_dest") {
    return Bytecode::Op::kJumpDest;
  }
  if (s == "jump_rel") {
    return Bytecode::Op::kJumpRel;
  }
  if (s == "jump_rel_if") {
    return Bytecode::Op::kJumpRelIf;
  }
  if (s == "le") {
    return Bytecode::Op::kLe;
  }
  if (s == "load") {
    return Bytecode::Op::kLoad;
  }
  if (s == "literal") {
    return Bytecode::Op::kLiteral;
  }
  if (s == "logical_and") {
    return Bytecode::Op::kLogicalAnd;
  }
  if (s == "logical_or") {
    return Bytecode::Op::kLogicalOr;
  }
  if (s == "lt") {
    return Bytecode::Op::kLt;
  }
  if (s == "match_arm") {
    return Bytecode::Op::kMatchArm;
  }
  if (s == "mul") {
    return Bytecode::Op::kMul;
  }
  if (s == "ne") {
    return Bytecode::Op::kNe;
  }
  if (s == "negate") {
    return Bytecode::Op::kNegate;
  }
  if (s == "or") {
    return Bytecode::Op::kOr;
  }
  if (s == "pop") {
    return Bytecode::Op::kPop;
  }
  if (s == "range") {
    return Bytecode::Op::kRange;
  }
  if (s == "recv") {
    return Bytecode::Op::kRecv;
  }
  if (s == "recv_nonblocking") {
    return Bytecode::Op::kRecvNonBlocking;
  }
  if (s == "send") {
    return Bytecode::Op::kSend;
  }
  if (s == "shl") {
    return Bytecode::Op::kShl;
  }
  if (s == "shr") {
    return Bytecode::Op::kShr;
  }
  if (s == "slice") {
    return Bytecode::Op::kSlice;
  }
  if (s == "spawn") {
    return Bytecode::Op::kSpawn;
  }
  if (s == "store") {
    return Bytecode::Op::kStore;
  }
  if (s == "sub") {
    return Bytecode::Op::kSub;
  }
  if (s == "swap") {
    return Bytecode::Op::kSwap;
  }
  if (s == "trace") {
    return Bytecode::Op::kTrace;
  }
  if (s == "width_slice") {
    return Bytecode::Op::kWidthSlice;
  }
  if (s == "xor") {
    return Bytecode::Op::kXor;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("String was not a bytecode op: `", s, "`"));
}

}  // namespace

std::string OpToString(Bytecode::Op op) {
  switch (op) {
    case Bytecode::Op::kAdd:
      return "add";
    case Bytecode::Op::kAnd:
      return "and";
    case Bytecode::Op::kCall:
      return "call";
    case Bytecode::Op::kCast:
      return "cast";
    case Bytecode::Op::kConcat:
      return "concat";
    case Bytecode::Op::kCreateArray:
      return "create_array";
    case Bytecode::Op::kCreateTuple:
      return "create_tuple";
    case Bytecode::Op::kDiv:
      return "div";
    case Bytecode::Op::kDup:
      return "dup";
    case Bytecode::Op::kExpandTuple:
      return "expand_tuple";
    case Bytecode::Op::kEq:
      return "eq";
    case Bytecode::Op::kFail:
      return "fail";
    case Bytecode::Op::kGe:
      return "ge";
    case Bytecode::Op::kGt:
      return "gt";
    case Bytecode::Op::kIndex:
      return "index";
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
    case Bytecode::Op::kMatchArm:
      return "match_arm";
    case Bytecode::Op::kMul:
      return "mul";
    case Bytecode::Op::kNe:
      return "ne";
    case Bytecode::Op::kNegate:
      return "negate";
    case Bytecode::Op::kOr:
      return "or";
    case Bytecode::Op::kPop:
      return "pop";
    case Bytecode::Op::kRange:
      return "range";
    case Bytecode::Op::kRecv:
      return "recv";
    case Bytecode::Op::kRecvNonBlocking:
      return "recv_nonblocking";
    case Bytecode::Op::kSend:
      return "send";
    case Bytecode::Op::kShl:
      return "shl";
    case Bytecode::Op::kShr:
      return "shr";
    case Bytecode::Op::kSlice:
      return "slice";
    case Bytecode::Op::kSpawn:
      return "spawn";
    case Bytecode::Op::kStore:
      return "store";
    case Bytecode::Op::kSub:
      return "sub";
    case Bytecode::Op::kSwap:
      return "swap";
    case Bytecode::Op::kTrace:
      return "trace";
    case Bytecode::Op::kWidthSlice:
      return "width_slice";
    case Bytecode::Op::kXor:
      return "xor";
  }
  return absl::StrCat("<invalid: ", static_cast<int>(op), ">");
}

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

/* static */ Bytecode::MatchArmItem Bytecode::MatchArmItem::MakeInterpValue(
    const InterpValue& interp_value) {
  return MatchArmItem(Kind::kInterpValue, interp_value);
}

/* static */ Bytecode::MatchArmItem Bytecode::MatchArmItem::MakeLoad(
    SlotIndex slot_index) {
  return MatchArmItem(Kind::kLoad, slot_index);
}

/* static */ Bytecode::MatchArmItem Bytecode::MatchArmItem::MakeStore(
    SlotIndex slot_index) {
  return MatchArmItem(Kind::kStore, slot_index);
}

/* static */ Bytecode::MatchArmItem Bytecode::MatchArmItem::MakeTuple(
    std::vector<MatchArmItem> elements) {
  return MatchArmItem(Kind::kTuple, std::move(elements));
}

/* static */ Bytecode::MatchArmItem Bytecode::MatchArmItem::MakeWildcard() {
  return MatchArmItem(Kind::kWildcard);
}

Bytecode::MatchArmItem::MatchArmItem(Kind kind)
    : kind_(kind), data_(absl::nullopt) {}

Bytecode::MatchArmItem::MatchArmItem(
    Kind kind,
    std::variant<InterpValue, SlotIndex, std::vector<MatchArmItem>> data)
    : kind_(kind), data_(std::move(data)) {}

absl::StatusOr<InterpValue> Bytecode::MatchArmItem::interp_value() const {
  if (!data_.has_value()) {
    return absl::InvalidArgumentError("MatchArmItem does not hold data.");
  }
  if (!std::holds_alternative<InterpValue>(data_.value())) {
    return absl::InvalidArgumentError("Bytecode data is not an InterpValue.");
  }

  return std::get<InterpValue>(data_.value());
}

absl::StatusOr<Bytecode::SlotIndex> Bytecode::MatchArmItem::slot_index() const {
  if (!data_.has_value()) {
    return absl::InvalidArgumentError("MatchArmItem does not hold data.");
  }
  if (!std::holds_alternative<SlotIndex>(data_.value())) {
    return absl::InvalidArgumentError("Bytecode data is not a SlotIndex.");
  }

  return std::get<SlotIndex>(data_.value());
}

absl::StatusOr<std::vector<Bytecode::MatchArmItem>>
Bytecode::MatchArmItem::tuple_elements() const {
  if (!data_.has_value()) {
    return absl::InvalidArgumentError("MatchArmItem does not hold data.");
  }
  if (!std::holds_alternative<std::vector<MatchArmItem>>(data_.value())) {
    return absl::InvalidArgumentError("Bytecode data is not a SlotIndex.");
  }

  return std::get<std::vector<MatchArmItem>>(data_.value());
}

std::string Bytecode::MatchArmItem::ToString() const {
  switch (kind_) {
    case Kind::kInterpValue:
      return absl::StrCat("value:",
                          std::get<InterpValue>(data_.value()).ToString());
    case Kind::kLoad:
      return absl::StrCat("load:", std::get<SlotIndex>(data_.value()).value());
    case Kind::kStore:
      return absl::StrCat("store:", std::get<SlotIndex>(data_.value()).value());
    case Kind::kTuple: {
      std::vector<MatchArmItem> elements =
          std::get<std::vector<MatchArmItem>>(data_.value());
      std::vector<std::string> pieces;
      pieces.reserve(elements.size());
      for (const auto& element : elements) {
        pieces.push_back(element.ToString());
      }
      return absl::StrCat("tuple: (", absl::StrJoin(pieces, ", "), ")");
    }
    case Kind::kWildcard:
      return "wildcard";
  }

  return "<invalid MatchArmItem>";
}

#define DEF_UNARY_BUILDER(OP_NAME)                           \
  /* static */ Bytecode Bytecode::Make##OP_NAME(Span span) { \
    return Bytecode(span, Op::k##OP_NAME);                   \
  }

DEF_UNARY_BUILDER(Dup);
DEF_UNARY_BUILDER(Index);
DEF_UNARY_BUILDER(Invert);
DEF_UNARY_BUILDER(JumpDest);
DEF_UNARY_BUILDER(LogicalOr);
DEF_UNARY_BUILDER(Pop);
DEF_UNARY_BUILDER(Range);
DEF_UNARY_BUILDER(Send);
DEF_UNARY_BUILDER(Swap);

#undef DEF_UNARY_BUILDER

/* static */ Bytecode Bytecode::MakeCreateTuple(Span span,
                                                NumElements num_elements) {
  return Bytecode(span, Op::kCreateTuple, num_elements);
}

/* static */ Bytecode Bytecode::MakeJumpRelIf(Span span, JumpTarget target) {
  return Bytecode(span, Op::kJumpRelIf, target);
}

/* static */ Bytecode Bytecode::MakeJumpRel(Span span, JumpTarget target) {
  return Bytecode(span, Op::kJumpRel, target);
}

/* static */ Bytecode Bytecode::MakeLiteral(Span span, InterpValue literal) {
  return Bytecode(span, Op::kLiteral, std::move(literal));
}

/* static */ Bytecode Bytecode::MakeLoad(Span span, SlotIndex slot_index) {
  return Bytecode(span, Op::kLoad, slot_index);
}

/* static */ Bytecode Bytecode::MakeMatchArm(Span span, MatchArmItem item) {
  return Bytecode(span, Op::kMatchArm, std::move(item));
}

/* static */ Bytecode Bytecode::MakeRecv(Span span,
                                         std::unique_ptr<ConcreteType> type) {
  return Bytecode(span, Op::kRecv, std::move(type));
}

/* static */ Bytecode Bytecode::MakeRecvNonBlocking(
    Span span, std::unique_ptr<ConcreteType> type) {
  return Bytecode(span, Op::kRecvNonBlocking, std::move(type));
}

/* static */ Bytecode Bytecode::MakeSpawn(Span span, SpawnData spawn_data) {
  return Bytecode(span, Op::kSpawn, spawn_data);
}

/* static */ Bytecode Bytecode::MakeStore(Span span, SlotIndex slot_index) {
  return Bytecode(span, Op::kStore, slot_index);
}

absl::StatusOr<Bytecode::JumpTarget> Bytecode::jump_target() const {
  if (!data_.has_value()) {
    return absl::InvalidArgumentError("Bytecode does not hold data.");
  }
  if (!std::holds_alternative<JumpTarget>(data_.value())) {
    return absl::InvalidArgumentError("Bytecode data is not a JumpTarget.");
  }

  return std::get<JumpTarget>(data_.value());
}

absl::StatusOr<const Bytecode::MatchArmItem*> Bytecode::match_arm_item() const {
  if (!data_.has_value()) {
    return absl::InvalidArgumentError("Bytecode does not hold data.");
  }
  if (!std::holds_alternative<MatchArmItem>(data_.value())) {
    return absl::InvalidArgumentError("Bytecode data is not MatchArmItem.");
  }
  return &std::get<MatchArmItem>(data_.value());
}

absl::StatusOr<Bytecode::NumElements> Bytecode::num_elements() const {
  if (!data_.has_value()) {
    return absl::InvalidArgumentError("Bytecode does not hold data.");
  }
  if (!std::holds_alternative<NumElements>(data_.value())) {
    return absl::InvalidArgumentError("Bytecode data is not a NumElements.");
  }

  return std::get<NumElements>(data_.value());
}

absl::StatusOr<const Bytecode::TraceData*> Bytecode::trace_data() const {
  if (!data_.has_value()) {
    return absl::InvalidArgumentError("Bytecode does not hold data.");
  }
  if (!std::holds_alternative<TraceData>(data_.value())) {
    return absl::InvalidArgumentError("Bytecode data is not a TraceData.");
  }
  return &std::get<TraceData>(data_.value());
}

absl::StatusOr<Bytecode::SlotIndex> Bytecode::slot_index() const {
  if (!data_.has_value()) {
    return absl::InvalidArgumentError("Bytecode does not hold data.");
  }
  if (!std::holds_alternative<SlotIndex>(data_.value())) {
    return absl::InvalidArgumentError("Bytecode data is not a SlotIndex.");
  }

  return std::get<SlotIndex>(data_.value());
}

absl::StatusOr<Bytecode::InvocationData> Bytecode::invocation_data() const {
  if (!data_.has_value()) {
    return absl::InvalidArgumentError("Bytecode does not hold data.");
  }
  if (!std::holds_alternative<InvocationData>(data_.value())) {
    return absl::InvalidArgumentError(
        "Bytecode data is not a SymbolicBindings.");
  }

  return std::get<InvocationData>(data_.value());
}

absl::StatusOr<const Bytecode::SpawnData*> Bytecode::spawn_data() const {
  if (!data_.has_value()) {
    return absl::InvalidArgumentError("Bytecode does not hold data.");
  }
  if (!std::holds_alternative<SpawnData>(data_.value())) {
    return absl::InvalidArgumentError("Bytecode data is not an InterpValue.");
  }

  return &std::get<SpawnData>(data_.value());
}

absl::StatusOr<InterpValue> Bytecode::value_data() const {
  if (!data_.has_value()) {
    return absl::InvalidArgumentError("Bytecode does not hold data.");
  }
  if (!std::holds_alternative<InterpValue>(data_.value())) {
    return absl::InvalidArgumentError("Bytecode data is not an InterpValue.");
  }

  return std::get<InterpValue>(data_.value());
}

absl::StatusOr<const ConcreteType*> Bytecode::type_data() const {
  if (!data_.has_value()) {
    return absl::InvalidArgumentError("Bytecode does not hold data.");
  }
  if (!std::holds_alternative<std::unique_ptr<ConcreteType>>(data_.value())) {
    return absl::InvalidArgumentError("Bytecode data is not a ConcreteType.");
  }
  return std::get<std::unique_ptr<ConcreteType>>(data_.value()).get();
}

void Bytecode::PatchJumpTarget(int64_t value) {
  XLS_CHECK(op_ == Op::kJumpRelIf || op_ == Op::kJumpRel)
      << "Cannot patch non-jump op: " << OpToString(op_);
  XLS_CHECK(data_.has_value());
  JumpTarget jump_target = std::get<JumpTarget>(data_.value());
  XLS_CHECK_EQ(jump_target, kPlaceholderJumpAmount);
  data_ = JumpTarget(value);
}

std::string Bytecode::ToString(bool source_locs) const {
  std::string op_string = OpToString(op_);
  std::string loc_string;
  if (source_locs) {
    loc_string = " @ " + source_span_.ToString();
  }

  if (op_ == Op::kJumpRel || op_ == Op::kJumpRelIf) {
    XLS_CHECK(std::holds_alternative<JumpTarget>(data_.value()));
    JumpTarget target = std::get<JumpTarget>(data_.value());
    return absl::StrFormat("%s %+d%s", OpToString(op_), target.value(),
                           loc_string);
  }

  if (data_.has_value()) {
    struct DataVisitor {
      std::string operator()(const std::unique_ptr<ConcreteType>& v) {
        return v->ToString();
      }

      std::string operator()(const InvocationData& iv) {
        if (iv.bindings.has_value()) {
          return absl::StrCat(iv.invocation->ToString(), " : ",
                              iv.bindings.value().ToString());
        }

        return iv.invocation->ToString();
      }

      std::string operator()(const InterpValue& v) { return v.ToString(); }

      std::string operator()(const NumElements& v) {
        return absl::StrCat(v.value());
      }

      std::string operator()(const SlotIndex& v) {
        return absl::StrCat(v.value());
      }

      std::string operator()(const TraceData& trace_data) {
        std::vector<std::string> pieces;
        pieces.reserve(trace_data.size());
        for (const auto& step : trace_data) {
          if (std::holds_alternative<std::string>(step)) {
            pieces.push_back(std::get<std::string>(step));
          } else {
            pieces.push_back(std::string(
                FormatPreferenceToString(std::get<FormatPreference>(step))));
          }
        }
        return absl::StrCat("trace data: ", absl::StrJoin(pieces, ", "));
      }

      std::string operator()(const JumpTarget& target) {
        if (target == kPlaceholderJumpAmount) {
          return "<placeholder>";
        }

        return absl::StrCat(target.value());
      }

      std::string operator()(const MatchArmItem& v) { return v.ToString(); }

      std::string operator()(const SpawnData& spawn_data) {
        return spawn_data.spawn->ToString();
      }
    };

    std::string data_string = std::visit(DataVisitor(), data_.value());

    return absl::StrFormat("%s %s%s", op_string, data_string, loc_string);
  }

  return absl::StrFormat("%s%s", op_string, loc_string);
}

static absl::StatusOr<InterpValue> ParseInterpValue(std::string_view text) {
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
    std::string_view text) {
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
        // ParseInterpValue works for DSLX literal-typed numbers, but jump
        // offsets won't have type prefixes or optional leading '+' signs, so we
        // use SimpleAtoi() instead (rather than updating ParseNumber()).
        int64_t integer_data;
        if (!absl::SimpleAtoi(value_str, &integer_data)) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Could not parse data payload in line: `%s`", line));
        }
        data = Bytecode::JumpTarget(integer_data);
      }
    }
    result.push_back(Bytecode(Span::Fake(), op, std::move(data)));
  }
  return result;
}

absl::StatusOr<std::unique_ptr<BytecodeFunction>> BytecodeFunction::Create(
    const Module* owner, const Function* source_fn, const TypeInfo* type_info,
    std::vector<Bytecode> bytecodes) {
  auto bf = absl::WrapUnique(
      new BytecodeFunction(owner, source_fn, type_info, std::move(bytecodes)));
  XLS_RETURN_IF_ERROR(bf->Init());
  return bf;
}

BytecodeFunction::BytecodeFunction(const Module* owner,
                                   const Function* source_fn,
                                   const TypeInfo* type_info,
                                   std::vector<Bytecode> bytecodes)
    : owner_(owner),
      source_fn_(source_fn),
      type_info_(type_info),
      bytecodes_(std::move(bytecodes)) {}

absl::Status BytecodeFunction::Init() {
  num_slots_ = 0;
  for (const auto& bc : bytecodes_) {
    if (bc.op() == Bytecode::Op::kLoad || bc.op() == Bytecode::Op::kStore) {
      XLS_ASSIGN_OR_RETURN(Bytecode::SlotIndex slot, bc.slot_index());
      num_slots_ = std::max(num_slots_, slot.value() + 1);
    }
  }
  return absl::OkStatus();
}

std::vector<Bytecode> BytecodeFunction::CloneBytecodes() const {
  // Create a modifiable copy of the bytecodes.
  std::vector<Bytecode> bytecodes;
  for (const auto& bc : bytecodes_) {
    if (bc.has_data()) {
      if (std::holds_alternative<Bytecode::SlotIndex>(bc.data().value())) {
        bytecodes.emplace_back(
            Bytecode(bc.source_span(), bc.op(),
                     std::get<Bytecode::SlotIndex>(bc.data().value())));
      } else if (std::holds_alternative<InterpValue>(bc.data().value())) {
        bytecodes.emplace_back(
            Bytecode(bc.source_span(), bc.op(),
                     std::get<InterpValue>(bc.data().value())));
      } else {
        const std::unique_ptr<ConcreteType>& type =
            std::get<std::unique_ptr<ConcreteType>>(bc.data().value());
        bytecodes.emplace_back(
            Bytecode(bc.source_span(), bc.op(), type->CloneToUnique()));
      }
    } else {
      bytecodes.emplace_back(Bytecode(bc.source_span(), bc.op()));
    }
  }

  return bytecodes;
}

std::string BytecodeFunction::ToString() const {
  std::vector<std::string> lines;
  lines.reserve(bytecodes_.size());
  for (const auto& bytecode : bytecodes_) {
    lines.push_back(bytecode.ToString());
  }
  return absl::StrJoin(lines, "\n");
}

}  // namespace xls::dslx
