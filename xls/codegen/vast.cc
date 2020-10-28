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

using absl::StrJoin;

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

std::string MacroRef::Emit() { return absl::StrCat("`", name_); }

std::string Include::Emit() {
  return absl::StrFormat("`include \"%s\"", path_);
}

std::string VerilogFile::Emit() {
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
  items_.push_back(parent_->Make<LocalParamItem>(name, value));
  return parent_->Make<LocalParamItemRef>(items_.back());
}

CaseArm::CaseArm(VerilogFile* f, CaseLabel label)
    : label_(label), statements_(f->Make<StatementBlock>(f)) {}

std::string CaseArm::GetLabelString() {
  return absl::visit(
      Visitor{[](Expression* named) { return named->Emit(); },
              [](DefaultSentinel) { return std::string("default"); }},
      label_);
}

std::string StatementBlock::Emit() {
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
  port.wire = f->Make<WireDef>(proto.name(), f->PlainLiteral(proto.width()));
  return port;
}

std::string Port::ToString() const {
  return absl::StrFormat("Port(dir=%s, name=\"%s\")",
                         verilog::ToString(direction), name());
}

absl::StatusOr<PortProto> Port::ToProto() const {
  if (!wire->width()->IsLiteral()) {
    return absl::FailedPreconditionError(
        "Width of port is not a literal, cannot convert to proto: " +
        wire->Emit());
  }
  PortProto proto;
  proto.set_direction(direction == Direction::kInput ? DIRECTION_INPUT
                                                     : DIRECTION_OUTPUT);
  proto.set_name(wire->GetName());
  XLS_ASSIGN_OR_RETURN(int64 width,
                       wire->width()->AsLiteralOrDie()->bits().ToUint64());
  proto.set_width(width);
  return proto;
}

namespace {
absl::StatusOr<int64> GetBitsForDirection(absl::Span<const Port> ports,
                                          Direction direction) {
  int64 result = 0;
  for (const Port& port : ports) {
    if (port.direction != direction) {
      continue;
    }
    if (!port.wire->width()->IsLiteral()) {
      return absl::FailedPreconditionError(absl::StrFormat(
          "Port with name \"%s\" has width that is not a literal value.",
          port.wire->GetName()));
    }
    XLS_ASSIGN_OR_RETURN(
        int64 width, port.wire->width()->AsLiteralOrDie()->bits().ToUint64());
    result += width;
  }
  return result;
}
}  // namespace

absl::StatusOr<int64> GetInputBits(absl::Span<const Port> ports) {
  return GetBitsForDirection(ports, Direction::kInput);
}

absl::StatusOr<int64> GetOutputBits(absl::Span<const Port> ports) {
  return GetBitsForDirection(ports, Direction::kOutput);
}

VerilogFunction::VerilogFunction(absl::string_view name, int64 result_width,
                                 VerilogFile* file)
    : name_(name),
      result_width_(result_width),
      return_value_def_(
          file->Make<RegDef>(name, file->PlainLiteral(result_width_))),
      statement_block_(file->Make<StatementBlock>(file)),
      file_(file) {}

LogicRef* VerilogFunction::AddArgument(absl::string_view name, int64 width) {
  argument_defs_.push_back(
      file_->Make<RegDef>(name, file_->PlainLiteral(width)));
  return file_->Make<LogicRef>(argument_defs_.back());
}

LogicRef* VerilogFunction::return_value_ref() {
  return file_->Make<LogicRef>(return_value_def_);
}

// For the given expression returns a string " [e - 1:0]". As a special case if
// 'e' is the literal 1, then returns the empty string.
static std::string WidthToString(Expression* e) {
  if (e->IsLiteral()) {
    uint64 value = e->AsLiteralOrDie()->bits().ToUint64().value();
    // Elide the width if it is one.
    // TODO(https://github.com/google/xls/issues/43): Avoid this special case
    // and perform the equivalent logic at a higher abstraction level than VAST.
    return value == 1 ? ""
                      : absl::StrFormat(" [%s:0]", absl::StrCat(value - 1));
  }
  Literal literal(UBits(1, 32), FormatPreference::kDefault,
                  /*emit_bit_count=*/false);
  // TODO(meheff): It'd be better to use VerilogFile::Sub here to keep
  // precedence values in one place but we don't have a VerilogFile.
  const int64 kBinarySubPrecedence = 9;
  BinaryInfix b(e, "-", &literal, /*precedence=*/kBinarySubPrecedence);
  return absl::StrFormat(" [%s:0]", b.Emit());
}

std::string VerilogFunction::Emit() {
  std::vector<std::string> lines;
  for (RegDef* reg_def : block_reg_defs_) {
    lines.push_back(reg_def->Emit());
  }
  lines.push_back(statement_block_->Emit());
  return absl::StrCat(
      absl::StrFormat("function automatic%s %s (%s);\n",
                      // Special case a single bit return value because
                      // WidthToString returns the empty string in this case.
                      return_value_ref()->IsScalarReg()
                          ? " [0:0]"
                          : WidthToString(return_value_ref()->width()),
                      name(),
                      absl::StrJoin(argument_defs_, ", ",
                                    [](std::string* out, RegDef* d) {
                                      absl::StrAppend(out, "input ",
                                                      d->EmitNoSemi());
                                    })),
      Indent(absl::StrJoin(lines, "\n")), "\nendfunction");
}

std::string VerilogFunctionCall::Emit() {
  return absl::StrFormat(
      "%s(%s)", func_->name(),
      absl::StrJoin(args_, ", ", [](std::string* out, Expression* e) {
        absl::StrAppend(out, e->Emit());
      }));
}

LogicRef* Module::AddPortAsExpression(Direction direction,
                                      absl::string_view name,
                                      Expression* width) {
  Def* def = parent_->Make<WireDef>(name, width);
  ports_.push_back(Port{direction, def});
  return parent_->Make<LogicRef>(def);
}

LogicRef* Module::AddPort(Direction direction, absl::string_view name,
                          int64 width) {
  return AddPortAsExpression(direction, name, parent()->PlainLiteral(width));
}

LogicRef1* Module::AddInput(absl::string_view name) {
  return AddPort(Direction::kInput, name, 1)->AsLogicRefNOrDie<1>();
}

LogicRef1* Module::AddOutput(absl::string_view name) {
  return AddPort(Direction::kOutput, name, 1)->AsLogicRefNOrDie<1>();
}

LogicRef* Module::AddRegAsExpression(absl::string_view name, Expression* width,
                                     RegInit init, ModuleSection* section) {
  if (section == nullptr) {
    section = &top_;
  }
  return parent_->Make<LogicRef>(section->Add<RegDef>(name, width, init));
}

LogicRef* Module::AddUnpackedArrayReg(
    absl::string_view name, Expression* element_width,
    absl::Span<const UnpackedArrayBound> bounds, RegInit init,
    ModuleSection* section) {
  if (section == nullptr) {
    section = &top_;
  }
  return parent_->Make<LogicRef>(
      section->Add<UnpackedArrayRegDef>(name, element_width, bounds, init));
}

LogicRef* Module::AddReg(absl::string_view name, int64 width,
                         absl::optional<int64> init, ModuleSection* section) {
  if (init.has_value()) {
    return AddRegAsExpression(name, parent_->PlainLiteral(width),
                              parent_->Literal(*init, width), section);
  } else {
    return AddRegAsExpression(name, parent_->PlainLiteral(width),
                              UninitializedSentinel(), section);
  }
}

LogicRef* Module::AddWireAsExpression(absl::string_view name, Expression* width,
                                      ModuleSection* section) {
  if (section == nullptr) {
    section = &top_;
  }
  return parent_->Make<LogicRef>(section->Add<WireDef>(name, width));
}

LogicRef* Module::AddWire(absl::string_view name, int64 width,
                          ModuleSection* section) {
  return AddWireAsExpression(name, parent_->PlainLiteral(width), section);
}

// TODO(meheff): Consider removing subtyped-by-bitcount reg and wire defs as
// they are lightly used in the generators and may not add enough value to
// justify their existence.
#define ADD_REG_WIRE_N(__n)                                                 \
  LogicRef##__n* Module::AddReg##__n(absl::string_view name,                \
                                     absl::optional<int64> init,            \
                                     ModuleSection* section) {              \
    if (section == nullptr) {                                               \
      section = &top_;                                                      \
    }                                                                       \
    RegInit init_expr = UninitializedSentinel();                            \
    if (init.has_value()) {                                                 \
      init_expr = parent_->Literal(*init, __n);                             \
    }                                                                       \
    return parent_->Make<LogicRef##__n>(                                    \
        section->Add<RegDef>(name, parent_->PlainLiteral(__n), init_expr)); \
  }                                                                         \
  LogicRef##__n* Module::AddWire##__n(absl::string_view name,               \
                                      ModuleSection* section) {             \
    if (section == nullptr) {                                               \
      section = &top_;                                                      \
    }                                                                       \
    return parent_->Make<LogicRef##__n>(                                    \
        section->Add<WireDef>(name, parent_->PlainLiteral(__n)));           \
  }

ADD_REG_WIRE_N(1)
ADD_REG_WIRE_N(8)

ParameterRef* Module::AddParameter(absl::string_view name, Expression* rhs) {
  Parameter* param = AddModuleMember(parent_->Make<Parameter>(name, rhs));
  return parent_->Make<ParameterRef>(param);
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

std::string XSentinel::Emit() { return absl::StrFormat("%d'dx", width_); }

std::string ToString(const RegInit& init) {
  return absl::visit(Visitor{[](Expression* e) { return e->Emit(); },
                             [](UninitializedSentinel) {
                               return std::string("<uninitialized>");
                             }},
                     init);
}

std::string WireDef::Emit() {
  return absl::StrFormat("wire%s%s %s;", is_signed() ? " signed " : "",
                         WidthToString(width()), name());
}

std::string RegDef::Emit() {
  std::string result =
      absl::StrFormat("reg%s%s %s", is_signed() ? " signed " : "",
                      WidthToString(width()), name());
  if (!absl::holds_alternative<UninitializedSentinel>(init_)) {
    absl::StrAppend(&result, " = ", ToString(init_));
  }
  absl::StrAppend(&result, ";");
  return result;
}

// Returns a string appropriate for defining the array bounds of a unpacked
// array reg declaration. The string is a sequence of sizes (e.g.,
// "[0:41][0:122]" or "[42][123]" depending upon whether the bounds are defined
// with ranges or sizes) with the first size corresponding to the outer most
// dimension of the array.
static std::string UnpackedArrayBoundsToString(
    absl::Span<const UnpackedArrayBound> bounds) {
  XLS_CHECK_GE(bounds.size(), 1);
  std::string result;
  for (const UnpackedArrayBound& bound : bounds) {
    absl::visit(Visitor{[&](Expression* size) {
                          absl::StrAppendFormat(&result, "[%s]", size->Emit());
                        },
                        [&](std::pair<Expression*, Expression*> pair) {
                          absl::StrAppendFormat(&result, "[%s:%s]",
                                                pair.first->Emit(),
                                                pair.second->Emit());
                        }},
                bound);
  }
  return result;
}

std::string UnpackedArrayRegDef::Emit() {
  std::string result =
      absl::StrFormat("reg%s %s%s", WidthToString(width()), name(),
                      UnpackedArrayBoundsToString(bounds()));
  if (!absl::holds_alternative<UninitializedSentinel>(init_)) {
    absl::StrAppend(&result, " = ", ToString(init_));
  }
  absl::StrAppend(&result, ";");
  return result;
}

std::string UnpackedArrayWireDef::Emit() {
  std::string result =
      absl::StrFormat("wire%s %s%s", WidthToString(width()), name(),
                      UnpackedArrayBoundsToString(bounds()));
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
                             [](StructuredProcedure* sp) { return sp->Emit(); },
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

std::string ContinuousAssignment::Emit() {
  return absl::StrFormat("assign %s = %s;", lhs_->Emit(), rhs_->Emit());
}

std::string Comment::Emit() {
  return absl::StrCat("// ", absl::StrReplaceAll(text_, {{"\n", "\n// "}}));
}

std::string SystemTaskCall::Emit() {
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

std::string SystemFunctionCall::Emit() {
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

std::string Module::Emit() {
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

std::string Literal::Emit() {
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

bool Literal::IsLiteralWithValue(int64 target) const {
  if (!bits().FitsInInt64()) {
    return false;
  }
  return bits().ToInt64().value() == target;
}

// TODO(meheff): Escape string.
std::string QuotedString::Emit() { return absl::StrFormat("\"%s\"", str_); }

std::string Slice::Emit() {
  if (subject_->IsScalarReg()) {
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

std::string DynamicSlice::Emit() {
  return absl::StrFormat("%s[%s +: %s]", subject_->Emit(), start_->Emit(),
                         width_->Emit());
}

std::string Index::Emit() {
  if (subject_->IsScalarReg()) {
    // If subject is scalar (no width given in declaration) then avoid indexing
    // as this is invalid Verilog. The only valid index values are zero.
    // TODO(https://github.com/google/xls/issues/43): Avoid this special case
    // and perform the equivalent logic at a higher abstraction level than VAST.
    XLS_CHECK(index_->IsLiteralWithValue(0)) << index_->Emit();
    return subject_->Emit();
  }
  return absl::StrFormat("%s[%s]", subject_->Emit(), index_->Emit());
}

// Returns the given string wrappend in parentheses.
static std::string ParenWrap(absl::string_view s) {
  return absl::StrFormat("(%s)", s);
}

std::string Ternary::Emit() {
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

std::string Parameter::Emit() {
  return absl::StrFormat("parameter %s = %s;", name_, rhs_->Emit());
}

std::string LocalParamItem::Emit() {
  return absl::StrFormat("%s = %s", name_, rhs_->Emit());
}

std::string LocalParam::Emit() {
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

std::string BinaryInfix::Emit() {
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

std::string Concat::Emit() {
  std::string arg_string = absl::StrFormat(
      "{%s}", absl::StrJoin(args_, ", ", [](std::string* out, Expression* e) {
        absl::StrAppend(out, e->Emit());
      }));

  if (replication_.has_value()) {
    return absl::StrFormat("{%s%s}", (*replication_)->Emit(), arg_string);
  } else {
    return arg_string;
  }
}

std::string ArrayAssignmentPattern::Emit() {
  return absl::StrFormat(
      "'{%s}", absl::StrJoin(args_, ", ", [](std::string* out, Expression* e) {
        absl::StrAppend(out, e->Emit());
      }));
}

std::string Unary::Emit() {
  // Nested unary ops should be wrapped in parentheses as this is required by
  // some consumers of Verilog.
  return absl::StrFormat(
      "%s%s", op_,
      ((arg_->precedence() < precedence()) || arg_->IsUnary())
          ? ParenWrap(arg_->Emit())
          : arg_->Emit());
}

StatementBlock* Case::AddCaseArm(CaseLabel label) {
  arms_.push_back(parent_->Make<CaseArm>(parent_, label));
  return arms_.back()->statements();
}

std::string Case::Emit() {
  std::string result = absl::StrFormat("case (%s)\n", subject_->Emit());
  for (auto& arm : arms_) {
    absl::StrAppend(&result,
                    Indent(absl::StrFormat("%s: %s", arm->GetLabelString(),
                                           arm->statements()->Emit())),
                    "\n");
  }
  absl::StrAppend(&result, "endcase");
  return result;
}

Conditional::Conditional(VerilogFile* f, Expression* condition)
    : parent_(f),
      condition_(condition),
      consequent_(f->Make<StatementBlock>(f)) {}

StatementBlock* Conditional::AddAlternate(Expression* condition) {
  // The conditional must not have been previously closed with an unconditional
  // alternate ("else").
  XLS_CHECK(alternates_.empty() || alternates_.back().first != nullptr);
  alternates_.push_back({condition, parent_->Make<StatementBlock>(parent_)});
  return alternates_.back().second;
}

std::string Conditional::Emit() {
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

WhileStatement::WhileStatement(VerilogFile* f, Expression* condition)
    : condition_(condition), statements_(f->Make<StatementBlock>(f)) {}

std::string WhileStatement::Emit() {
  return absl::StrFormat("while (%s) %s", condition_->Emit(),
                         statements()->Emit());
}

std::string RepeatStatement::Emit() {
  return absl::StrFormat("repeat (%s) %s;", repeat_count_->Emit(),
                         statement_->Emit());
}

std::string EventControl::Emit() {
  return absl::StrFormat("@(%s);", event_expression_->Emit());
}

std::string PosEdge::Emit() {
  return absl::StrFormat("posedge %s", expression_->Emit());
}

std::string NegEdge::Emit() {
  return absl::StrFormat("negedge %s", expression_->Emit());
}

std::string DelayStatement::Emit() {
  std::string delay_str = delay_->precedence() < Expression::kMaxPrecedence
                              ? ParenWrap(delay_->Emit())
                              : delay_->Emit();
  if (delayed_statement_) {
    return absl::StrFormat("#%s %s", delay_str, delayed_statement_->Emit());
  } else {
    return absl::StrFormat("#%s;", delay_str);
  }
}

std::string WaitStatement::Emit() {
  return absl::StrFormat("wait(%s);", event_->Emit());
}

std::string Forever::Emit() {
  return absl::StrCat("forever ", statement_->Emit());
}

std::string BlockingAssignment::Emit() {
  return absl::StrFormat("%s = %s;", lhs_->Emit(), rhs_->Emit());
}

std::string NonblockingAssignment::Emit() {
  return absl::StrFormat("%s <= %s;", lhs_->Emit(), rhs_->Emit());
}

StructuredProcedure::StructuredProcedure(VerilogFile* f)
    : statements_(f->Make<StatementBlock>(f)) {}

namespace {

std::string EmitSensitivityListElement(const SensitivityListElement& element) {
  return absl::visit(
      Visitor{[](ImplicitEventExpression e) -> std::string { return "*"; },
              [](PosEdge* p) -> std::string { return p->Emit(); },
              [](NegEdge* n) -> std::string { return n->Emit(); }},
      element);
}

}  // namespace

std::string AlwaysBase::Emit() {
  return absl::StrFormat(
      "%s @ (%s) %s", name(),
      absl::StrJoin(sensitivity_list_, " or ",
                    [](std::string* out, const SensitivityListElement& e) {
                      absl::StrAppend(out, EmitSensitivityListElement(e));
                    }),
      statements_->Emit());
}

std::string AlwaysComb::Emit() {
  return absl::StrFormat("%s %s", name(), statements_->Emit());
}

std::string Initial::Emit() {
  std::string result = "initial ";
  absl::StrAppend(&result, statements_->Emit());
  return result;
}

AlwaysFlop::AlwaysFlop(VerilogFile* file, LogicRef* clk,
                       absl::optional<Reset> rst)
    : file_(file),
      clk_(clk),
      rst_(rst),
      top_block_(file_->Make<StatementBlock>(file)) {
  if (rst_.has_value()) {
    // Reset signal specified. Construct conditional which switches the reset
    // signal.
    Expression* rst_condition;
    if (rst_->active_low) {
      rst_condition = file_->LogicalNot(rst_->signal);
    } else {
      rst_condition = rst_->signal;
    }
    Conditional* conditional =
        top_block_->Add<Conditional>(file_, rst_condition);
    reset_block_ = conditional->consequent();
    assignment_block_ = conditional->AddAlternate();
  } else {
    // No reset signal specified.
    reset_block_ = nullptr;
    assignment_block_ = top_block_;
  }
}

void AlwaysFlop::AddRegister(LogicRef* reg, Expression* reg_next,
                             Expression* reset_value) {
  if (reset_value) {
    XLS_CHECK(reset_block_ != nullptr);
    reset_block_->Add<NonblockingAssignment>(reg, reset_value);
  }
  assignment_block_->Add<NonblockingAssignment>(reg, reg_next);
}

std::string AlwaysFlop::Emit() {
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

std::string Instantiation::Emit() {
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
