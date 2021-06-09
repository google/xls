// Copyright 2020 The XLS Authors
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

#include "xls/codegen/vast.h"

#include "absl/flags/flag.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/strip.h"
#include "xls/common/indent.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "re2/re2.h"

namespace xls {
namespace verilog {

std::string SanitizeIdentifier(absl::string_view name) {
  if (name.empty()) {
    return "_";
  }
  // Numbers can appear anywhere in the identifier except the first
  // character. Handle this case by prefixing the sanitized name with an
  // underscore.
  std::string sanitized = absl::ascii_isdigit(name[0]) ? absl::StrCat("_", name)
                                                       : std::string(name);
  for (int i = 0; i < sanitized.size(); ++i) {
    if (!absl::ascii_isalnum(sanitized[i])) {
      sanitized[i] = '_';
    }
  }
  return sanitized;
}

std::string ToString(Direction direction) {
  switch (direction) {
    case Direction::kInput:
      return "input";
    case Direction::kOutput:
      return "output";
    default:
      return "<invalid direction>";
  }
}

std::string MacroRef::Emit() const { return absl::StrCat("`", name_); }

std::string Include::Emit() const {
  return absl::StrFormat("`include \"%s\"", path_);
}

DataType* VerilogFile::BitVectorTypeNoScalar(int64_t bit_count,
                                             bool is_signed) {
  return Make<DataType>(PlainLiteral(bit_count), is_signed);
}

DataType* VerilogFile::BitVectorType(int64_t bit_count, bool is_signed) {
  XLS_CHECK_GT(bit_count, 0);
  if (bit_count == 1) {
    if (is_signed) {
      return Make<DataType>(/*width=*/nullptr, /*is_signed=*/true);
    } else {
      return Make<DataType>();
    }
  }
  return BitVectorTypeNoScalar(bit_count, is_signed);
}

DataType* VerilogFile::PackedArrayType(int64_t element_bit_count,
                                       absl::Span<const int64_t> dims,
                                       bool is_signed) {
  XLS_CHECK_GT(element_bit_count, 0);
  std::vector<Expression*> dim_exprs;
  for (int64_t d : dims) {
    dim_exprs.push_back(PlainLiteral(d));
  }
  // For packed arrays we always use a bitvector (non-scalar) for the element
  // type when the element bit width is 1. For example, if element bit width is
  // one and dims is {42} we generate the following type:
  //   reg [0:0][41:0] foo;
  // If we emitted a scalar type, it would look like:
  //   reg [41:0] foo;
  // Which would generate invalid verilog if we index into an element
  // (e.g. foo[2][0]) because scalars are not indexable.
  return Make<DataType>(
      PlainLiteral(element_bit_count), /*packed_dims=*/dim_exprs,
      /*unpacked_dims=*/std::vector<Expression*>(), is_signed);
}

DataType* VerilogFile::UnpackedArrayType(int64_t element_bit_count,
                                         absl::Span<const int64_t> dims,
                                         bool is_signed) {
  XLS_CHECK_GT(element_bit_count, 0);
  std::vector<Expression*> dim_exprs;
  for (int64_t d : dims) {
    dim_exprs.push_back(PlainLiteral(d));
  }
  return Make<DataType>(
      element_bit_count == 1 ? nullptr : PlainLiteral(element_bit_count),
      /*packed_dims=*/std::vector<Expression*>(),
      /*unpacked_dims=*/dim_exprs, is_signed);
}

std::string VerilogFile::Emit() const {
  auto file_member_str = [](const FileMember& member) -> std::string {
    return absl::visit(
        Visitor{[](Include* m) -> std::string { return m->Emit(); },
                [](Module* m) -> std::string { return m->Emit(); }},
        member);
  };

  std::string out;
  for (const FileMember& member : members_) {
    absl::StrAppend(&out, file_member_str(member), "\n");
  }
  return out;
}

LocalParamItemRef* LocalParam::AddItem(absl::string_view name,
                                       Expression* value) {
  items_.push_back(file()->Make<LocalParamItem>(name, value));
  return file()->Make<LocalParamItemRef>(items_.back());
}

CaseArm::CaseArm(CaseLabel label, VerilogFile* file)
    : VastNode(file),
      label_(label),
      statements_(file->Make<StatementBlock>()) {}

std::string CaseArm::Emit() const {
  return absl::visit(
      Visitor{[](Expression* named) { return named->Emit(); },
              [](DefaultSentinel) { return std::string("default"); }},
      label_);
}

std::string StatementBlock::Emit() const {
  // TODO(meheff): We can probably be smarter about optionally emitting the
  // begin/end.
  if (statements_.empty()) {
    return "begin end";
  }
  std::string result = "begin\n";
  std::vector<std::string> lines;
  for (const auto& statement : statements_) {
    lines.push_back(statement->Emit());
  }
  absl::StrAppend(&result, Indent(absl::StrJoin(lines, "\n")), "\nend");
  return result;
}

Port Port::FromProto(const PortProto& proto, VerilogFile* f) {
  Port port;
  port.direction = proto.direction() == DIRECTION_INPUT ? Direction::kInput
                                                        : Direction::kOutput;
  port.wire = f->Make<WireDef>(proto.name(), f->BitVectorType(proto.width()));
  return port;
}

std::string Port::ToString() const {
  return absl::StrFormat("Port(dir=%s, name=\"%s\")",
                         verilog::ToString(direction), name());
}

absl::StatusOr<PortProto> Port::ToProto() const {
  PortProto proto;
  proto.set_direction(direction == Direction::kInput ? DIRECTION_INPUT
                                                     : DIRECTION_OUTPUT);
  proto.set_name(wire->GetName());
  XLS_ASSIGN_OR_RETURN(int64_t width, wire->data_type()->FlatBitCountAsInt64());
  proto.set_width(width);
  return proto;
}

VerilogFunction::VerilogFunction(absl::string_view name, DataType* result_type,
                                 VerilogFile* file)
    : VastNode(file),
      name_(name),
      return_value_def_(file->Make<RegDef>(name, result_type)),
      statement_block_(file->Make<StatementBlock>()) {}

LogicRef* VerilogFunction::AddArgument(absl::string_view name, DataType* type) {
  argument_defs_.push_back(file()->Make<RegDef>(name, type));
  return file()->Make<LogicRef>(argument_defs_.back());
}

LogicRef* VerilogFunction::return_value_ref() {
  return file()->Make<LogicRef>(return_value_def_);
}

std::string VerilogFunction::Emit() const {
  std::vector<std::string> lines;
  for (RegDef* reg_def : block_reg_defs_) {
    lines.push_back(reg_def->Emit());
  }
  lines.push_back(statement_block_->Emit());
  return absl::StrCat(
      absl::StrFormat(
          "function automatic%s (%s);\n",
          return_value_def_->data_type()->EmitWithIdentifier(name()),
          absl::StrJoin(argument_defs_, ", ",
                        [](std::string* out, RegDef* d) {
                          absl::StrAppend(out, "input ", d->EmitNoSemi());
                        })),
      Indent(absl::StrJoin(lines, "\n")), "\nendfunction");
}

std::string VerilogFunctionCall::Emit() const {
  return absl::StrFormat(
      "%s(%s)", func_->name(),
      absl::StrJoin(args_, ", ", [](std::string* out, Expression* e) {
        absl::StrAppend(out, e->Emit());
      }));
}

LogicRef* Module::AddPortDef(Direction direction, Def* def) {
  ports_.push_back(Port{direction, def});
  return file()->Make<LogicRef>(def);
}

LogicRef* Module::AddInput(absl::string_view name, DataType* type) {
  return AddPortDef(Direction::kInput,
                    file()->Make<WireDef>(name, std::move(type)));
}

LogicRef* Module::AddOutput(absl::string_view name, DataType* type) {
  return AddPortDef(Direction::kOutput,
                    file()->Make<WireDef>(name, std::move(type)));
}

LogicRef* Module::AddReg(absl::string_view name, DataType* type,
                         Expression* init, ModuleSection* section) {
  if (section == nullptr) {
    section = &top_;
  }
  return file()->Make<LogicRef>(
      section->Add<RegDef>(name, std::move(type), init));
}

LogicRef* Module::AddWire(absl::string_view name, DataType* type,
                          ModuleSection* section) {
  if (section == nullptr) {
    section = &top_;
  }
  return file()->Make<LogicRef>(section->Add<WireDef>(name, std::move(type)));
}

ParameterRef* Module::AddParameter(absl::string_view name, Expression* rhs) {
  Parameter* param = AddModuleMember(file()->Make<Parameter>(name, rhs));
  return file()->Make<ParameterRef>(param);
}

Literal* Expression::AsLiteralOrDie() {
  XLS_CHECK(IsLiteral());
  return static_cast<Literal*>(this);
}

IndexableExpression* Expression::AsIndexableExpressionOrDie() {
  XLS_CHECK(IsIndexableExpression());
  return static_cast<IndexableExpression*>(this);
}

Unary* Expression::AsUnaryOrDie() {
  XLS_CHECK(IsUnary());
  return static_cast<Unary*>(this);
}

LogicRef* Expression::AsLogicRefOrDie() {
  XLS_CHECK(IsLogicRef());
  return static_cast<LogicRef*>(this);
}

std::string XSentinel::Emit() const { return absl::StrFormat("%d'dx", width_); }

// Returns a string representation of the given expression minus one.
static std::string WidthToLimit(Expression* expr) {
  if (expr->IsLiteral()) {
    // If the expression is a literal, then we can emit the value - 1 directly.
    uint64_t value = expr->AsLiteralOrDie()->bits().ToUint64().value();
    return absl::StrCat(value - 1);
  }
  Literal* one = expr->file()->PlainLiteral(1);
  Expression* width_minus_one = expr->file()->Sub(expr, one);
  return width_minus_one->Emit();
}

std::string DataType::EmitWithIdentifier(absl::string_view identifier) const {
  std::string result = is_signed_ ? " signed" : "";
  if (width_ != nullptr) {
    absl::StrAppendFormat(&result, " [%s:0]", WidthToLimit(width()));
  }
  for (Expression* dim : packed_dims()) {
    absl::StrAppendFormat(&result, "[%s:0]", WidthToLimit(dim));
  }
  absl::StrAppend(&result, " ", identifier);
  for (Expression* dim : unpacked_dims()) {
    // In SystemVerilog unpacked arrays can be specified using only the size
    // rather than a range.
    if (file()->use_system_verilog()) {
      absl::StrAppendFormat(&result, "[%s]", dim->Emit());
    } else {
      absl::StrAppendFormat(&result, "[0:%s]", WidthToLimit(dim));
    }
  }
  return result;
}

absl::StatusOr<int64_t> DataType::WidthAsInt64() const {
  if (width() == nullptr) {
    // No width indicates a single-bit signal.
    return 1;
  }

  if (!width()->IsLiteral()) {
    return absl::FailedPreconditionError("Width is not a literal: " +
                                         width()->Emit());
  }
  return width()->AsLiteralOrDie()->bits().ToUint64();
}

absl::StatusOr<int64_t> DataType::FlatBitCountAsInt64() const {
  XLS_ASSIGN_OR_RETURN(int64_t bit_count, WidthAsInt64());
  for (Expression* dim : packed_dims()) {
    if (!dim->IsLiteral()) {
      return absl::FailedPreconditionError(
          "Packed dimension is not a literal:" + dim->Emit());
    }
    XLS_ASSIGN_OR_RETURN(int64_t dim_size,
                         dim->AsLiteralOrDie()->bits().ToUint64());
    bit_count = bit_count * dim_size;
  }
  for (Expression* dim : unpacked_dims()) {
    if (!dim->IsLiteral()) {
      return absl::FailedPreconditionError(
          "Unpacked dimension is not a literal:" + dim->Emit());
    }
    XLS_ASSIGN_OR_RETURN(int64_t dim_size,
                         dim->AsLiteralOrDie()->bits().ToUint64());
    bit_count = bit_count * dim_size;
  }
  return bit_count;
}

std::string Def::Emit() const { return EmitNoSemi() + ";"; }

std::string Def::EmitNoSemi() const {
  std::string kind_str;
  switch (data_kind()) {
    case DataKind::kReg:
      kind_str = "reg";
      break;
    case DataKind::kWire:
      kind_str = "wire";
      break;
    case DataKind::kLogic:
      kind_str = "logic";
      break;
  }
  return absl::StrCat(kind_str, data_type()->EmitWithIdentifier(GetName()));
}

std::string RegDef::Emit() const {
  std::string result = Def::EmitNoSemi();
  if (init_ != nullptr) {
    absl::StrAppend(&result, " = ", init_->Emit());
  }
  absl::StrAppend(&result, ";");
  return result;
}

namespace {

// "Match" statement for emitting a ModuleMember.
std::string EmitModuleMember(const ModuleMember& member) {
  return absl::visit(Visitor{[](Def* d) { return d->Emit(); },
                             [](LocalParam* p) { return p->Emit(); },
                             [](Parameter* p) { return p->Emit(); },
                             [](Instantiation* i) { return i->Emit(); },
                             [](ContinuousAssignment* c) { return c->Emit(); },
                             [](Comment* c) { return c->Emit(); },
                             [](BlankLine* b) { return b->Emit(); },
                             [](RawStatement* s) { return s->Emit(); },
                             [](StructuredProcedure* sp) { return sp->Emit(); },
                             [](AlwaysComb* ac) { return ac->Emit(); },
                             [](AlwaysFf* af) { return af->Emit(); },
                             [](AlwaysFlop* af) { return af->Emit(); },
                             [](VerilogFunction* f) { return f->Emit(); },
                             [](ModuleSection* s) { return s->Emit(); }},
                     member);
}

}  // namespace

std::vector<ModuleMember> ModuleSection::GatherMembers() const {
  std::vector<ModuleMember> all_members;
  for (const ModuleMember& member : members_) {
    if (absl::holds_alternative<ModuleSection*>(member)) {
      std::vector<ModuleMember> submembers =
          absl::get<ModuleSection*>(member)->GatherMembers();
      all_members.insert(all_members.end(), submembers.begin(),
                         submembers.end());
    } else {
      all_members.push_back(member);
    }
  }
  return all_members;
}

std::string ModuleSection::Emit() const {
  std::vector<std::string> elements;
  for (const ModuleMember& member : GatherMembers()) {
    elements.push_back(EmitModuleMember(member));
  }
  return absl::StrJoin(elements, "\n");
}

std::string ContinuousAssignment::Emit() const {
  return absl::StrFormat("assign %s = %s;", lhs_->Emit(), rhs_->Emit());
}

std::string Comment::Emit() const {
  return absl::StrCat("// ", absl::StrReplaceAll(text_, {{"\n", "\n// "}}));
}

std::string RawStatement::Emit() const { return text_; }

std::string Assert::Emit() const {
  // The $fatal statement takes finish_number as the first argument which is a
  // value in the set {0, 1, 2}. This value "may be used in an
  // implementation-specific manner" (from the SystemVerilog LRM). We choose
  // zero arbitrarily.
  constexpr int64_t kFinishNumber = 0;
  return absl::StrFormat(
      "assert (%s) else $fatal(%d%s);", condition_->Emit(), kFinishNumber,
      error_message_.empty() ? ""
                             : absl::StrFormat(", \"%s\"", error_message_));
}

std::string Cover::Emit() const {
  // Coverpoints don't work without clock sources. Don't emit them in that case.
  return absl::StrFormat("%s: cover property (%s%s);", label_,
                         absl::StrCat("@(posedge ", clk_->Emit(), ") "),
                         condition_->Emit());
}

std::string SystemTaskCall::Emit() const {
  if (args_.has_value()) {
    return absl::StrFormat(
        "$%s(%s);", name_,
        absl::StrJoin(*args_, ", ", [](std::string* out, Expression* e) {
          absl::StrAppend(out, e->Emit());
        }));
  } else {
    return absl::StrFormat("$%s;", name_);
  }
}

std::string SystemFunctionCall::Emit() const {
  if (args_.has_value()) {
    return absl::StrFormat(
        "$%s(%s)", name_,
        absl::StrJoin(*args_, ", ", [](std::string* out, Expression* e) {
          absl::StrAppend(out, e->Emit());
        }));
  } else {
    return absl::StrFormat("$%s", name_);
  }
}

std::string Module::Emit() const {
  std::string result = absl::StrCat("module ", name_);
  if (ports_.empty()) {
    absl::StrAppend(&result, ";\n");
  } else {
    absl::StrAppend(&result, "(\n  ");
    absl::StrAppend(
        &result,
        absl::StrJoin(ports_, ",\n  ", [](std::string* out, const Port& port) {
          absl::StrAppendFormat(out, "%s %s", ToString(port.direction),
                                port.wire->EmitNoSemi());
        }));
    absl::StrAppend(&result, "\n);\n");
  }
  absl::StrAppend(&result, Indent(top_.Emit()), "\n");
  absl::StrAppend(&result, "endmodule");
  return result;
}

std::string Literal::Emit() const {
  if (format_ == FormatPreference::kDefault) {
    XLS_CHECK_LE(bits_.bit_count(), 32);
    return absl::StrFormat("%s", bits_.ToString(FormatPreference::kDecimal));
  }
  if (format_ == FormatPreference::kDecimal) {
    std::string prefix;
    if (emit_bit_count_) {
      prefix = absl::StrFormat("%d'd", bits_.bit_count());
    }
    return absl::StrFormat("%s%s", prefix,
                           bits_.ToString(FormatPreference::kDecimal));
  }
  if (format_ == FormatPreference::kBinary) {
    return absl::StrFormat(
        "%d'b%s", bits_.bit_count(),
        bits_.ToRawDigits(format_, /*emit_leading_zeros=*/true));
  }
  XLS_CHECK_EQ(format_, FormatPreference::kHex);
  return absl::StrFormat(
      "%d'h%s", bits_.bit_count(),
      bits_.ToRawDigits(FormatPreference::kHex, /*emit_leading_zeros=*/true));
}

bool Literal::IsLiteralWithValue(int64_t target) const {
  // VAST Literals are always unsigned. Signed literal values are created by
  // casting a VAST Literal to a signed type.
  if (target < 0) {
    return false;
  }
  if (!bits().FitsInUint64()) {
    return false;
  }
  return bits().ToUint64().value() == target;
}

// TODO(meheff): Escape string.
std::string QuotedString::Emit() const {
  return absl::StrFormat("\"%s\"", str_);
}

static bool IsScalarLogicRef(IndexableExpression* expr) {
  auto* logic_ref = dynamic_cast<LogicRef*>(expr);
  return logic_ref != nullptr && logic_ref->def()->data_type()->IsScalar();
}

std::string Slice::Emit() const {
  if (IsScalarLogicRef(subject_)) {
    // If subject is scalar (no width given in declaration) then avoid slicing
    // as this is invalid Verilog. The only valid hi/lo values are zero.
    // TODO(https://github.com/google/xls/issues/43): Avoid this special case
    // and perform the equivalent logic at a higher abstraction level than VAST.
    XLS_CHECK(hi_->IsLiteralWithValue(0)) << hi_->Emit();
    XLS_CHECK(lo_->IsLiteralWithValue(0)) << lo_->Emit();
    return subject_->Emit();
  }
  return absl::StrFormat("%s[%s:%s]", subject_->Emit(), hi_->Emit(),
                         lo_->Emit());
}

std::string PartSelect::Emit() const {
  return absl::StrFormat("%s[%s +: %s]", subject_->Emit(), start_->Emit(),
                         width_->Emit());
}

std::string Index::Emit() const {
  if (IsScalarLogicRef(subject_)) {
    // If subject is scalar (no width given in declaration) then avoid indexing
    // as this is invalid Verilog. The only valid index values are zero.
    // TODO(https://github.com/google/xls/issues/43): Avoid this special case
    // and perform the equivalent logic at a higher abstraction level than VAST.
    XLS_CHECK(index_->IsLiteralWithValue(0))
        << absl::StreamFormat("%s[%s]", subject_->Emit(), index_->Emit());
    return subject_->Emit();
  }
  return absl::StrFormat("%s[%s]", subject_->Emit(), index_->Emit());
}

// Returns the given string wrappend in parentheses.
static std::string ParenWrap(absl::string_view s) {
  return absl::StrFormat("(%s)", s);
}

std::string Ternary::Emit() const {
  auto maybe_paren_wrap = [this](Expression* e) {
    if (e->precedence() <= precedence()) {
      return ParenWrap(e->Emit());
    }
    return e->Emit();
  };
  return absl::StrFormat("%s ? %s : %s", maybe_paren_wrap(test_),
                         maybe_paren_wrap(consequent_),
                         maybe_paren_wrap(alternate_));
}

std::string Parameter::Emit() const {
  return absl::StrFormat("parameter %s = %s;", name_, rhs_->Emit());
}

std::string LocalParamItem::Emit() const {
  return absl::StrFormat("%s = %s", name_, rhs_->Emit());
}

std::string LocalParam::Emit() const {
  std::string result = "localparam";
  if (items_.size() == 1) {
    absl::StrAppend(&result, " ", items_[0]->Emit(), ";");
    return result;
  }
  auto append_item = [](std::string* out, LocalParamItem* item) {
    absl::StrAppend(out, item->Emit());
  };
  absl::StrAppend(&result, "\n  ", absl::StrJoin(items_, ",\n  ", append_item),
                  ";");
  return result;
}

std::string BinaryInfix::Emit() const {
  // Equal precedence operators are evaluated left-to-right so LHS only needs to
  // be wrapped if its precedence is strictly less than this operators. The
  // RHS, however, must be wrapped if its less than or equal precedence.
  std::string lhs_string = lhs_->precedence() < precedence()
                               ? ParenWrap(lhs_->Emit())
                               : lhs_->Emit();
  std::string rhs_string = rhs_->precedence() <= precedence()
                               ? ParenWrap(rhs_->Emit())
                               : rhs_->Emit();
  return absl::StrFormat("%s %s %s", lhs_string, op_, rhs_string);
}

std::string Concat::Emit() const {
  std::string arg_string = absl::StrFormat(
      "{%s}", absl::StrJoin(args_, ", ", [](std::string* out, Expression* e) {
        absl::StrAppend(out, e->Emit());
      }));

  if (replication_ != nullptr) {
    return absl::StrFormat("{%s%s}", replication_->Emit(), arg_string);
  } else {
    return arg_string;
  }
}

std::string ArrayAssignmentPattern::Emit() const {
  return absl::StrFormat(
      "'{%s}", absl::StrJoin(args_, ", ", [](std::string* out, Expression* e) {
        absl::StrAppend(out, e->Emit());
      }));
}

std::string Unary::Emit() const {
  // Nested unary ops should be wrapped in parentheses as this is required by
  // some consumers of Verilog.
  return absl::StrFormat(
      "%s%s", op_,
      ((arg_->precedence() < precedence()) || arg_->IsUnary())
          ? ParenWrap(arg_->Emit())
          : arg_->Emit());
}

StatementBlock* Case::AddCaseArm(CaseLabel label) {
  arms_.push_back(file()->Make<CaseArm>(label));
  return arms_.back()->statements();
}

std::string Case::Emit() const {
  std::string result = absl::StrFormat("case (%s)\n", subject_->Emit());
  for (auto& arm : arms_) {
    absl::StrAppend(&result,
                    Indent(absl::StrFormat("%s: %s", arm->Emit(),
                                           arm->statements()->Emit())),
                    "\n");
  }
  absl::StrAppend(&result, "endcase");
  return result;
}

Conditional::Conditional(Expression* condition, VerilogFile* file)
    : Statement(file),
      condition_(condition),
      consequent_(file->Make<StatementBlock>()) {}

StatementBlock* Conditional::AddAlternate(Expression* condition) {
  // The conditional must not have been previously closed with an unconditional
  // alternate ("else").
  XLS_CHECK(alternates_.empty() || alternates_.back().first != nullptr);
  alternates_.push_back({condition, file()->Make<StatementBlock>()});
  return alternates_.back().second;
}

std::string Conditional::Emit() const {
  std::string result;
  absl::StrAppendFormat(&result, "if (%s) %s", condition_->Emit(),
                        consequent()->Emit());
  for (auto& alternate : alternates_) {
    absl::StrAppend(&result, " else ");
    if (alternate.first != nullptr) {
      absl::StrAppendFormat(&result, "if (%s) ", alternate.first->Emit());
    }
    absl::StrAppend(&result, alternate.second->Emit());
  }
  return result;
}

WhileStatement::WhileStatement(Expression* condition, VerilogFile* file)
    : Statement(file),
      condition_(condition),
      statements_(file->Make<StatementBlock>()) {}

std::string WhileStatement::Emit() const {
  return absl::StrFormat("while (%s) %s", condition_->Emit(),
                         statements()->Emit());
}

std::string RepeatStatement::Emit() const {
  return absl::StrFormat("repeat (%s) %s;", repeat_count_->Emit(),
                         statement_->Emit());
}

std::string EventControl::Emit() const {
  return absl::StrFormat("@(%s);", event_expression_->Emit());
}

std::string PosEdge::Emit() const {
  return absl::StrFormat("posedge %s", expression_->Emit());
}

std::string NegEdge::Emit() const {
  return absl::StrFormat("negedge %s", expression_->Emit());
}

std::string DelayStatement::Emit() const {
  std::string delay_str = delay_->precedence() < Expression::kMaxPrecedence
                              ? ParenWrap(delay_->Emit())
                              : delay_->Emit();
  if (delayed_statement_) {
    return absl::StrFormat("#%s %s", delay_str, delayed_statement_->Emit());
  } else {
    return absl::StrFormat("#%s;", delay_str);
  }
}

std::string WaitStatement::Emit() const {
  return absl::StrFormat("wait(%s);", event_->Emit());
}

std::string Forever::Emit() const {
  return absl::StrCat("forever ", statement_->Emit());
}

std::string BlockingAssignment::Emit() const {
  return absl::StrFormat("%s = %s;", lhs_->Emit(), rhs_->Emit());
}

std::string NonblockingAssignment::Emit() const {
  return absl::StrFormat("%s <= %s;", lhs_->Emit(), rhs_->Emit());
}

StructuredProcedure::StructuredProcedure(VerilogFile* file)
    : VastNode(file), statements_(file->Make<StatementBlock>()) {}

namespace {

std::string EmitSensitivityListElement(const SensitivityListElement& element) {
  return absl::visit(
      Visitor{[](ImplicitEventExpression e) -> std::string { return "*"; },
              [](PosEdge* p) -> std::string { return p->Emit(); },
              [](NegEdge* n) -> std::string { return n->Emit(); }},
      element);
}

}  // namespace

std::string AlwaysBase::Emit() const {
  return absl::StrFormat(
      "%s @ (%s) %s", name(),
      absl::StrJoin(sensitivity_list_, " or ",
                    [](std::string* out, const SensitivityListElement& e) {
                      absl::StrAppend(out, EmitSensitivityListElement(e));
                    }),
      statements_->Emit());
}

std::string AlwaysComb::Emit() const {
  return absl::StrFormat("%s %s", name(), statements_->Emit());
}

std::string Initial::Emit() const {
  std::string result = "initial ";
  absl::StrAppend(&result, statements_->Emit());
  return result;
}

AlwaysFlop::AlwaysFlop(LogicRef* clk, Reset rst, VerilogFile* file)
    : VastNode(file),
      clk_(clk),
      rst_(rst),
      top_block_(file->Make<StatementBlock>()) {
  // Reset signal specified. Construct conditional which switches the reset
  // signal.
  Expression* rst_condition;
  if (rst_->active_low) {
    rst_condition = file->LogicalNot(rst_->signal);
  } else {
    rst_condition = rst_->signal;
  }
  Conditional* conditional = top_block_->Add<Conditional>(rst_condition);
  reset_block_ = conditional->consequent();
  assignment_block_ = conditional->AddAlternate();
}

AlwaysFlop::AlwaysFlop(LogicRef* clk, VerilogFile* file)
    : VastNode(file), clk_(clk), top_block_(file->Make<StatementBlock>()) {
  // No reset signal specified.
  reset_block_ = nullptr;
  assignment_block_ = top_block_;
}

void AlwaysFlop::AddRegister(LogicRef* reg, Expression* reg_next,
                             Expression* reset_value) {
  if (reset_value != nullptr) {
    XLS_CHECK(reset_block_ != nullptr);
    reset_block_->Add<NonblockingAssignment>(reg, reset_value);
  }
  assignment_block_->Add<NonblockingAssignment>(reg, reg_next);
}

std::string AlwaysFlop::Emit() const {
  std::string result;
  std::string sensitivity_list = absl::StrCat("posedge ", clk_->Emit());
  if (rst_.has_value() && rst_->asynchronous) {
    absl::StrAppendFormat(&sensitivity_list, " or %s %s",
                          (rst_->active_low ? "negedge" : "posedge"),
                          rst_->signal->Emit());
  }
  absl::StrAppendFormat(&result, "always @ (%s) %s", sensitivity_list,
                        top_block_->Emit());
  return result;
}

std::string Instantiation::Emit() const {
  std::string result = absl::StrCat(module_name_, " ");
  auto append_connection = [](std::string* out, const Connection& parameter) {
    absl::StrAppendFormat(out, ".%s(%s)", parameter.port_name,
                          parameter.expression->Emit());
  };
  if (!parameters_.empty()) {
    absl::StrAppend(&result, "#(\n  ");
    absl::StrAppend(&result,
                    absl::StrJoin(parameters_, ",\n  ", append_connection),
                    "\n) ");
  }
  absl::StrAppend(&result, instance_name_);
  absl::StrAppend(&result, " (\n  ");
  absl::StrAppend(
      &result, absl::StrJoin(connections_, ",\n  ", append_connection), "\n)");
  absl::StrAppend(&result, ";");
  return result;
}

}  // namespace verilog
}  // namespace xls
