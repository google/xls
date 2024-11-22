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

#include "xls/codegen/vast/vast.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/common/indent.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/code_template.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/source_location.h"

namespace xls {
namespace verilog {

namespace {

int64_t NumberOfNewlines(std::string_view string) {
  int64_t number_of_newlines = 0;
  for (char c : string) {
    if (c == '\n') {
      ++number_of_newlines;
    }
  }
  return number_of_newlines;
}

void LineInfoStart(LineInfo* line_info, const VastNode* node) {
  if (line_info != nullptr) {
    line_info->Start(node);
  }
}

void LineInfoEnd(LineInfo* line_info, const VastNode* node) {
  if (line_info != nullptr) {
    line_info->End(node);
  }
}

void LineInfoIncrease(LineInfo* line_info, int64_t delta) {
  if (line_info != nullptr) {
    line_info->Increase(delta);
  }
}

// Converts a `DataKind` to its SystemVerilog name, if any. Emitting a data type
// in most contexts requires the containing entity to emit both the `DataKind`
// and the `DataType`, at least one of which should emit as nonempty.
std::string DataKindToString(DataKind kind) {
  switch (kind) {
    case DataKind::kReg:
      return "reg";
    case DataKind::kWire:
      return "wire";
    case DataKind::kLogic:
      return "logic";
    case DataKind::kInteger:
      return "integer";
    default:
      // For any other type, the `DataType->Emit()` output is sufficient.
      return "";
  }
}

std::string EmitNothing(const VastNode* node, LineInfo* line_info) {
  LineInfoStart(line_info, node);
  LineInfoEnd(line_info, node);
  return "";
}

// Helper for validating that there is not leading/trailing whitespace that can
// be stripped from string view `s` -- this helps us maintain our invariants
// that our Emit() calls generally do not return things with leading or
// trailing whitespace, so callers uniformly know what to expect in terms of
// spacing.
//
// This routine is made without performing string copies or performing a strcmp
// so should be usable in CHECKs instead of resorting to DCHECKs everywhere.
//
// i.e. callers are expected to call `CHECK(CannotStripWhitespace(s));`
bool CannotStripWhitespace(std::string_view s) {
  std::string_view stripped = absl::StripAsciiWhitespace(s);
  return stripped.size() == s.size();
}

// Helper that combines DataKind and DataType strings -- since either of these
// can be empty under certain circumstances we consolidate the handling of
// inserting spaces appropriately in this routine.
std::string CombineKindAndDataType(std::string_view kind_str,
                                   std::string_view data_type_str) {
  CHECK(CannotStripWhitespace(kind_str));
  CHECK(CannotStripWhitespace(data_type_str));

  std::string result;
  if (kind_str.empty()) {
    result = data_type_str;
  } else if (data_type_str.empty()) {
    result = kind_str;
  } else {
    result = absl::StrCat(kind_str, " ", data_type_str);
  }
  return result;
}

}  // namespace

int Precedence(OperatorKind kind) {
  switch (kind) {
    case OperatorKind::kNegate:
      return 12;
    case OperatorKind::kBitwiseNot:
      return 12;
    case OperatorKind::kLogicalNot:
      return 12;
    case OperatorKind::kAndReduce:
      return 12;
    case OperatorKind::kOrReduce:
      return 12;
    case OperatorKind::kXorReduce:
      return 12;
    case OperatorKind::kPower:
      return 11;
    case OperatorKind::kDiv:
      return 10;
    case OperatorKind::kMod:
      return 10;
    case OperatorKind::kMul:
      return 10;
    case OperatorKind::kAdd:
      return 9;
    case OperatorKind::kSub:
      return 9;
    case OperatorKind::kShll:
      return 8;
    case OperatorKind::kShra:
      return 8;
    case OperatorKind::kShrl:
      return 8;
    case OperatorKind::kGe:
      return 7;
    case OperatorKind::kGt:
      return 7;
    case OperatorKind::kLe:
      return 7;
    case OperatorKind::kLt:
      return 7;
    case OperatorKind::kNe:
      return 6;
    case OperatorKind::kEq:
      return 6;
    case OperatorKind::kNeX:
      return 6;
    case OperatorKind::kEqX:
      return 6;
    case OperatorKind::kBitwiseAnd:
      return 5;
    case OperatorKind::kBitwiseXor:
      return 4;
    case OperatorKind::kBitwiseOr:
      return 3;
    case OperatorKind::kLogicalAnd:
      return 2;
    case OperatorKind::kLogicalOr:
      return 1;
  }
}

std::string_view OperatorString(OperatorKind kind) {
  switch (kind) {
    case OperatorKind::kAdd:
      return "+";
    case OperatorKind::kLogicalAnd:
      return "&&";
    case OperatorKind::kBitwiseAnd:
      return "&";
    case OperatorKind::kNe:
      return "!=";
    case OperatorKind::kEq:
      return "==";
    case OperatorKind::kGe:
      return ">=";
    case OperatorKind::kGt:
      return ">";
    case OperatorKind::kLe:
      return "<=";
    case OperatorKind::kLt:
      return "<";
    case OperatorKind::kDiv:
      return "/";
    case OperatorKind::kMod:
      return "%";
    case OperatorKind::kMul:
      return "*";
    case OperatorKind::kPower:
      return "**";
    case OperatorKind::kBitwiseOr:
      return "|";
    case OperatorKind::kLogicalOr:
      return "||";
    case OperatorKind::kBitwiseXor:
      return "^";
    case OperatorKind::kShll:
      return "<<";
    case OperatorKind::kShra:
      return ">>>";
    case OperatorKind::kShrl:
      return ">>";
    case OperatorKind::kSub:
      return "-";
    case OperatorKind::kNeX:
      return "!==";
    case OperatorKind::kEqX:
      return "===";
    case OperatorKind::kNegate:
      return "-";
    case OperatorKind::kBitwiseNot:
      return "~";
    case OperatorKind::kLogicalNot:
      return "!";
    case OperatorKind::kAndReduce:
      return "&";
    case OperatorKind::kOrReduce:
      return "|";
    case OperatorKind::kXorReduce:
      return "^";
  }
}

std::string PartialLineSpans::ToString() const {
  return absl::StrCat(
      "[",
      absl::StrJoin(completed_spans, ", ",
                    [](std::string* out, const LineSpan& line_span) {
                      return line_span.ToString();
                    }),
      hanging_start_line.has_value()
          ? absl::StrCat("; ", hanging_start_line.value())
          : "",
      "]");
}

void LineInfo::Start(const VastNode* node) {
  if (!spans_.contains(node)) {
    nodes_.push_back(node);
    spans_[node];
  }
  CHECK(!spans_.at(node).hanging_start_line.has_value())
      << "LineInfoStart can't be called twice in a row on the same node!";
  spans_.at(node).hanging_start_line = current_line_number_;
}

void LineInfo::End(const VastNode* node) {
  CHECK(spans_.contains(node))
      << "LineInfoEnd called without corresponding LineInfoStart!";
  CHECK(spans_.at(node).hanging_start_line.has_value())
      << "LineInfoEnd can't be called twice in a row on the same node!";
  int64_t start_line = spans_.at(node).hanging_start_line.value();
  int64_t end_line = current_line_number_;
  spans_.at(node).completed_spans.push_back(LineSpan(start_line, end_line));
  spans_.at(node).hanging_start_line = std::nullopt;
}

void LineInfo::Increase(int64_t delta) { current_line_number_ += delta; }

std::optional<std::vector<LineSpan>> LineInfo::LookupNode(
    const VastNode* node) const {
  if (!spans_.contains(node)) {
    return std::nullopt;
  }
  if (spans_.at(node).hanging_start_line.has_value()) {
    return std::nullopt;
  }
  return spans_.at(node).completed_spans;
}

std::string SanitizeIdentifier(std::string_view name) {
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

std::string DataType::EmitWithIdentifier(LineInfo* line_info,
                                         std::string_view identifier) const {
  std::string base = Emit(line_info);
  if (base.empty()) {
    return std::string{identifier};
  }
  CHECK(CannotStripWhitespace(base));
  return absl::StrCat(base, " ", identifier);
}

std::string ScalarType::Emit(LineInfo* line_info) const {
  // The `DataKind` preceding the type is enough.
  if (!is_signed_) {
    return EmitNothing(this, line_info);
  }
  LineInfoStart(line_info, this);
  LineInfoEnd(line_info, this);
  return "signed";
}

std::string IntegerType::Emit(LineInfo* line_info) const {
  // The `DataKind` preceding the type is enough if it's signed, which is the
  // default.
  if (is_signed_) {
    return EmitNothing(this, line_info);
  }
  LineInfoStart(line_info, this);
  LineInfoEnd(line_info, this);
  return "unsigned";
}

std::string MacroRef::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  LineInfoIncrease(line_info, NumberOfNewlines(name_));
  LineInfoEnd(line_info, this);
  return absl::StrCat("`", name_);
}

std::string Include::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  LineInfoIncrease(line_info, NumberOfNewlines(path_));
  LineInfoEnd(line_info, this);
  return absl::StrFormat("`include \"%s\"", path_);
}

DataType* VerilogFile::ExternType(DataType* punable_type, std::string_view name,
                                  const SourceInfo& loc) {
  return Make<verilog::ExternType>(loc, punable_type, name);
}

BitVectorType* VerilogFile::BitVectorTypeNoScalar(int64_t bit_count,
                                                  const SourceInfo& loc,
                                                  bool is_signed) {
  return Make<verilog::BitVectorType>(
      loc, PlainLiteral(static_cast<int32_t>(bit_count), loc), is_signed);
}

DataType* VerilogFile::BitVectorType(int64_t bit_count, const SourceInfo& loc,
                                     bool is_signed) {
  CHECK_GT(bit_count, 0);
  if (bit_count == 1) {
    return Make<verilog::ScalarType>(loc, is_signed);
  }
  return BitVectorTypeNoScalar(bit_count, loc, is_signed);
}

PackedArrayType* VerilogFile::PackedArrayType(int64_t element_bit_count,
                                              absl::Span<const int64_t> dims,
                                              const SourceInfo& loc,
                                              bool is_signed) {
  CHECK_GT(element_bit_count, 0);
  std::vector<Expression*> dim_exprs;
  // For packed arrays we always use a bitvector (non-scalar) for the element
  // type when the element bit width is 1. For example, if element bit width is
  // one and dims is {42} we generate the following type:
  //   reg [0:0][41:0] foo;
  // If we emitted a scalar type, it would look like:
  //   reg [41:0] foo;
  // Which would generate invalid verilog if we index into an element
  // (e.g. foo[2][0]) because scalars are not indexable.
  return Make<verilog::PackedArrayType>(loc, element_bit_count, dims,
                                        is_signed);
}

UnpackedArrayType* VerilogFile::UnpackedArrayType(
    int64_t element_bit_count, absl::Span<const int64_t> dims,
    const SourceInfo& loc, bool is_signed) {
  CHECK_GT(element_bit_count, 0);
  std::vector<Expression*> dim_exprs;
  for (int64_t d : dims) {
    dim_exprs.push_back(PlainLiteral(static_cast<int32_t>(d), loc));
  }
  DataType* element_type = BitVectorType(element_bit_count, loc, is_signed);
  return Make<verilog::UnpackedArrayType>(loc, element_type, dims);
}

std::string VerilogFile::Emit(LineInfo* line_info) const {
  auto file_member_str = [=](const FileMember& member) -> std::string {
    return absl::visit([=](auto* m) { return m->Emit(line_info); }, member);
  };

  std::string out;
  for (const FileMember& member : members_) {
    absl::StrAppend(&out, file_member_str(member), "\n");
    LineInfoIncrease(line_info, 1);
  }
  return out;
}

LocalParamItemRef* LocalParam::AddItem(std::string_view name, Expression* value,
                                       const SourceInfo& loc) {
  items_.push_back(file()->Make<LocalParamItem>(loc, name, value));
  return file()->Make<LocalParamItemRef>(loc, items_.back());
}

CaseArm::CaseArm(CaseLabel label, VerilogFile* file, const SourceInfo& loc)
    : VastNode(file, loc),
      label_(label),
      statements_(file->Make<StatementBlock>(loc)) {}

std::string CaseArm::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  std::string result = absl::visit(
      Visitor{[=](Expression* named) { return named->Emit(line_info); },
              [](DefaultSentinel) { return std::string("default"); }},
      label_);
  LineInfoEnd(line_info, this);
  return result;
}

std::string StatementBlock::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  // TODO(meheff): We can probably be smarter about optionally emitting the
  // begin/end.
  if (statements_.empty()) {
    LineInfoEnd(line_info, this);
    return "begin end";
  }
  std::string result = "begin\n";
  LineInfoIncrease(line_info, 1);
  std::vector<std::string> lines;
  for (const auto& statement : statements_) {
    lines.push_back(statement->Emit(line_info));
    LineInfoIncrease(line_info, 1);
  }
  absl::StrAppend(&result, Indent(absl::StrJoin(lines, "\n")), "\nend");
  LineInfoEnd(line_info, this);
  return result;
}

std::string MacroStatementBlock::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  std::string result;
  for (const auto& statement : statements_) {
    absl::StrAppend(&result, Indent(statement->Emit(line_info)), "\n");
    LineInfoIncrease(line_info, 1);
  }
  LineInfoEnd(line_info, this);
  return result;
}

Port Port::FromProto(const PortProto& proto, VerilogFile* f) {
  Port port;
  port.direction = proto.direction() == DIRECTION_INPUT ? Direction::kInput
                                                        : Direction::kOutput;
  port.wire = f->Make<WireDef>(SourceInfo(), proto.name(),
                               f->BitVectorType(proto.width(), SourceInfo()));
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

VerilogFunction::VerilogFunction(std::string_view name, DataType* result_type,
                                 VerilogFile* file, const SourceInfo& loc)
    : VastNode(file, loc),
      name_(name),
      return_value_def_(file->Make<RegDef>(loc, name, result_type)),
      statement_block_(file->Make<StatementBlock>(loc)) {}

LogicRef* VerilogFunction::AddArgument(std::string_view name, DataType* type,
                                       const SourceInfo& loc) {
  argument_defs_.push_back(file()->Make<RegDef>(loc, name, type));
  return file()->Make<LogicRef>(loc, argument_defs_.back());
}

LogicRef* VerilogFunction::AddArgument(Def* def, const SourceInfo& loc) {
  argument_defs_.push_back(def);
  return file()->Make<LogicRef>(loc, argument_defs_.back());
}

LogicRef* VerilogFunction::return_value_ref() {
  return file()->Make<LogicRef>(return_value_def_->loc(), return_value_def_);
}

std::string VerilogFunction::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);

  // Construct the return type for the function, sometimes we want to put
  // "logic" in front when in SV mode.
  std::string return_type =
      return_value_def_->data_type()->EmitWithIdentifier(line_info, name());
  if (return_value_def_->data_type()->IsScalar() &&
      file()->use_system_verilog()) {
    // Preface the return type with "logic", so there's always a type provided.
    return_type =
        absl::StrCat(DataKindToString(DataKind::kLogic), " ", return_type);
  }
  if (!return_type.empty()) {
    return_type = absl::StrCat(" ", return_type);
  }

  std::string parameters =
      absl::StrJoin(argument_defs_, ", ", [=](std::string* out, Def* d) {
        absl::StrAppend(out, "input ", d->EmitNoSemi(line_info));
      });
  LineInfoIncrease(line_info, 1);
  std::vector<std::string> lines;
  for (RegDef* reg_def : block_reg_defs_) {
    lines.push_back(reg_def->Emit(line_info));
    LineInfoIncrease(line_info, 1);
  }
  lines.push_back(statement_block_->Emit(line_info));
  LineInfoIncrease(line_info, 1);
  LineInfoEnd(line_info, this);
  return absl::StrCat(
      absl::StrFormat("function automatic%s (%s);\n", return_type, parameters),
      Indent(absl::StrJoin(lines, "\n")), "\nendfunction");
}

std::string VerilogFunctionCall::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  std::string result = absl::StrFormat(
      "%s(%s)", func_->name(),
      absl::StrJoin(args_, ", ", [=](std::string* out, Expression* e) {
        absl::StrAppend(out, e->Emit(line_info));
      }));
  LineInfoEnd(line_info, this);
  return result;
}

LogicRef* Module::AddPortDef(Direction direction, Def* def,
                             const SourceInfo& loc) {
  ports_.push_back(Port{.direction = direction, .wire = def});
  return file()->Make<LogicRef>(loc, def);
}

LogicRef* Module::AddInput(std::string_view name, DataType* type,
                           const SourceInfo& loc) {
  return AddPortDef(
      Direction::kInput,
      type->IsUserDefined()
          ? static_cast<Def*>(file()->Make<UserDefinedDef>(loc, name, type))
          : static_cast<Def*>(file()->Make<WireDef>(loc, name, type)),
      loc);
}

LogicRef* Module::AddOutput(std::string_view name, DataType* type,
                            const SourceInfo& loc) {
  return AddPortDef(
      Direction::kOutput,
      type->IsUserDefined()
          ? static_cast<Def*>(file()->Make<UserDefinedDef>(loc, name, type))
          : static_cast<Def*>(file()->Make<WireDef>(loc, name, type)),
      loc);
}

LogicRef* Module::AddReg(std::string_view name, DataType* type,
                         const SourceInfo& loc, Expression* init,
                         ModuleSection* section) {
  if (section == nullptr) {
    section = &top_;
  }
  return file()->Make<LogicRef>(loc,
                                section->Add<RegDef>(loc, name, type, init));
}

LogicRef* Module::AddWire(std::string_view name, DataType* type,
                          const SourceInfo& loc, ModuleSection* section) {
  if (section == nullptr) {
    section = &top_;
  }
  return file()->Make<LogicRef>(loc, section->Add<WireDef>(loc, name, type));
}

LogicRef* Module::AddWire(std::string_view name, DataType* type,
                          Expression* init, const SourceInfo& loc,
                          ModuleSection* section) {
  if (section == nullptr) {
    section = &top_;
  }
  return file()->Make<LogicRef>(loc,
                                section->Add<WireDef>(loc, name, type, init));
}

LogicRef* Module::AddInteger(std::string_view name, const SourceInfo& loc,
                             ModuleSection* section) {
  if (section == nullptr) {
    section = &top_;
  }
  return file()->Make<LogicRef>(loc, section->Add<IntegerDef>(loc, name));
}

ParameterRef* Module::AddParameter(std::string_view name, Expression* rhs,
                                   const SourceInfo& loc) {
  Parameter* param = AddModuleMember(file()->Make<Parameter>(loc, name, rhs));
  return file()->Make<ParameterRef>(loc, param);
}

ParameterRef* Module::AddParameter(Def* def, Expression* rhs,
                                   const SourceInfo& loc) {
  Parameter* param = AddModuleMember(file()->Make<Parameter>(loc, def, rhs));
  return file()->Make<ParameterRef>(loc, param);
}

Typedef* Module::AddTypedef(Def* def, const SourceInfo& loc) {
  return AddModuleMember(file()->Make<Typedef>(loc, def));
}

TypedefType* VerilogPackageSection::AddEnumTypedef(std::string_view name,
                                                   Enum* enum_data_type,
                                                   const SourceInfo& loc) {
  Typedef* def = Add<Typedef>(
      loc, file()->Make<Def>(loc, name, DataKind::kUser, enum_data_type));
  return file()->Make<TypedefType>(loc, def);
}

TypedefType* VerilogPackageSection::AddEnumTypedef(
    std::string_view name, DataKind kind, DataType* data_type,
    absl::Span<EnumMember*> enum_members, const SourceInfo& loc) {
  return AddEnumTypedef(
      name, file()->Make<Enum>(loc, kind, data_type, enum_members), loc);
}

TypedefType* VerilogPackageSection::AddStructTypedef(std::string_view name,
                                                     Struct* struct_data_type,
                                                     const SourceInfo& loc) {
  Typedef* def = Add<Typedef>(
      loc, file()->Make<Def>(loc, name, DataKind::kUser, struct_data_type));
  return file()->Make<TypedefType>(loc, def);
}

TypedefType* VerilogPackageSection::AddStructTypedef(
    std::string_view name, absl::Span<Def*> struct_members,
    const SourceInfo& loc) {
  return AddStructTypedef(name, file()->Make<Struct>(loc, struct_members), loc);
}

ParameterRef* VerilogPackageSection::AddParameter(std::string_view name,
                                                  Expression* rhs,
                                                  const SourceInfo& loc) {
  Parameter* param =
      AddVerilogPackageMember(file()->Make<Parameter>(loc, name, rhs));
  return file()->Make<ParameterRef>(loc, param);
}

ParameterRef* VerilogPackageSection::AddParameter(Def* def, Expression* rhs,
                                                  const SourceInfo& loc) {
  Parameter* param =
      AddVerilogPackageMember(file()->Make<Parameter>(loc, def, rhs));
  return file()->Make<ParameterRef>(loc, param);
}

ParameterRef* ParameterRef::Duplicate() const {
  return file()->Make<ParameterRef>(loc(), parameter());
}

Literal* Expression::AsLiteralOrDie() {
  CHECK(IsLiteral());
  return static_cast<Literal*>(this);
}

IndexableExpression* Expression::AsIndexableExpressionOrDie() {
  CHECK(IsIndexableExpression());
  return static_cast<IndexableExpression*>(this);
}

Unary* Expression::AsUnaryOrDie() {
  CHECK(IsUnary());
  return static_cast<Unary*>(this);
}

LogicRef* Expression::AsLogicRefOrDie() {
  CHECK(IsLogicRef());
  return static_cast<LogicRef*>(this);
}

LogicRef* LogicRef::Duplicate() const {
  return file()->Make<LogicRef>(loc(), def());
}

std::string XSentinel::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  LineInfoEnd(line_info, this);
  return absl::StrFormat("%d'dx", width_);
}

static void FourValueFormatter(std::string* out, FourValueBit value) {
  char value_as_char = '\0';
  switch (value) {
    case FourValueBit::kZero:
      value_as_char = '0';
      break;
    case FourValueBit::kOne:
      value_as_char = '1';
      break;
    case FourValueBit::kUnknown:
      value_as_char = 'X';
      break;
    case FourValueBit::kHighZ:
      value_as_char = '?';
      break;
  }
  CHECK_NE(value_as_char, '\0') << "Internal Error";
  out->push_back(value_as_char);
}

std::string FourValueBinaryLiteral::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  LineInfoEnd(line_info, this);
  return absl::StrCat(bits_.size(), "'b",
                      absl::StrJoin(bits_, "", FourValueFormatter));
}

// Returns a string representation of the given expression minus one.
static std::string WidthToLimit(LineInfo* line_info, Expression* expr) {
  if (expr->IsLiteral()) {
    // If the expression is a literal, then we can emit the value - 1 directly.
    uint64_t value = expr->AsLiteralOrDie()->bits().ToUint64().value();
    return absl::StrCat(value - 1);
  }
  Literal* one = expr->file()->PlainLiteral(1, expr->loc());
  Expression* width_minus_one = expr->file()->Sub(expr, one, expr->loc());
  return width_minus_one->Emit(line_info);
}

BitVectorType::BitVectorType(int64_t width, bool is_signed, VerilogFile* file,
                             const SourceInfo& loc)
    : DataType(file, loc),
      size_expr_(file->PlainLiteral(static_cast<int32_t>(width), loc)),
      is_signed_(is_signed) {}

absl::StatusOr<int64_t> BitVectorType::WidthAsInt64() const {
  if (!size_expr_->IsLiteral()) {
    return absl::FailedPreconditionError("Width is not a literal: " +
                                         size_expr_->Emit(nullptr));
  }
  XLS_ASSIGN_OR_RETURN(int64_t size,
                       size_expr_->AsLiteralOrDie()->bits().ToUint64());
  return size + (size_expr_is_max_ ? 1 : 0);
}

absl::StatusOr<int64_t> BitVectorType::FlatBitCountAsInt64() const {
  return WidthAsInt64();
}

std::string BitVectorType::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  std::string result = is_signed_ ? "signed " : "";
  absl::StrAppendFormat(&result, "[%s:0]",
                        size_expr_is_max_
                            ? size_expr_->Emit(line_info)
                            : WidthToLimit(line_info, size_expr_));
  LineInfoEnd(line_info, this);
  return result;
}

ArrayTypeBase::ArrayTypeBase(DataType* element_type,
                             absl::Span<const int64_t> dims, bool dims_are_max,
                             VerilogFile* file, const SourceInfo& loc)
    : DataType(file, loc),
      element_type_(element_type),
      dims_are_max_(dims_are_max) {
  CHECK(!dims.empty());
  for (int64_t dim : dims) {
    dims_.push_back(file->PlainLiteral(static_cast<int32_t>(dim), loc));
  }
}

Expression* ArrayDimToWidth(Expression* dim, bool dim_is_max) {
  if (dim_is_max) {
    Literal* one = dim->file()->PlainLiteral(1, dim->loc());
    return dim->file()->Add(dim, one, dim->loc());
  }
  return dim;
}

PackedArrayType::PackedArrayType(Expression* width,
                                 absl::Span<Expression* const> packed_dims,
                                 bool is_signed, VerilogFile* file,
                                 const SourceInfo& loc)
    : ArrayTypeBase(file->Make<BitVectorType>(loc, width, is_signed),
                    packed_dims, /*dims_are_max=*/false, file, loc) {}

PackedArrayType::PackedArrayType(int64_t width,
                                 absl::Span<const int64_t> packed_dims,
                                 bool is_signed, VerilogFile* file,
                                 const SourceInfo& loc)
    : ArrayTypeBase(file->Make<BitVectorType>(loc, static_cast<int32_t>(width),
                                              is_signed),
                    packed_dims, /*dims_are_max=*/false, file, loc) {}

PackedArrayType::PackedArrayType(DataType* element_type,
                                 absl::Span<const int64_t> packed_dims,
                                 bool dims_are_max, VerilogFile* file,
                                 const SourceInfo& loc)
    : ArrayTypeBase(element_type, packed_dims, /*dims_are_max=*/dims_are_max,
                    file, loc) {}

absl::StatusOr<int64_t> PackedArrayType::FlatBitCountAsInt64() const {
  XLS_ASSIGN_OR_RETURN(int64_t bit_count, WidthAsInt64());
  for (Expression* dim : dims()) {
    if (!dim->IsLiteral()) {
      return absl::FailedPreconditionError(
          "Packed dimension is not a literal:" + dim->Emit(nullptr));
    }
    XLS_ASSIGN_OR_RETURN(int64_t dim_size,
                         dim->AsLiteralOrDie()->bits().ToUint64());
    bit_count *= dim_size + (dims_are_max() ? 1 : 0);
  }
  return bit_count;
}

std::string PackedArrayType::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  std::string result = element_type()->Emit(line_info);
  if (element_type()->IsUserDefined()) {
    // Imitate the space that a bit vector emits between the kind and innermost
    // dimension.
    absl::StrAppend(&result, " ");
  }
  for (Expression* dim : dims()) {
    absl::StrAppendFormat(
        &result, "[%s:0]",
        dims_are_max() ? dim->Emit(line_info) : WidthToLimit(line_info, dim));
  }
  LineInfoEnd(line_info, this);
  return result;
}

UnpackedArrayType::UnpackedArrayType(DataType* element_type,
                                     absl::Span<const int64_t> unpacked_dims,
                                     VerilogFile* file, const SourceInfo& loc)
    : ArrayTypeBase(element_type, unpacked_dims, /*dims_are_max=*/false, file,
                    loc) {
  CHECK(dynamic_cast<UnpackedArrayType*>(element_type) == nullptr);
}

absl::StatusOr<int64_t> UnpackedArrayType::FlatBitCountAsInt64() const {
  XLS_ASSIGN_OR_RETURN(int64_t bit_count,
                       element_type()->FlatBitCountAsInt64());
  for (Expression* dim : dims()) {
    if (!dim->IsLiteral()) {
      return absl::FailedPreconditionError(
          "Packed dimension is not a literal:" + dim->Emit(nullptr));
    }
    XLS_ASSIGN_OR_RETURN(int64_t dim_size,
                         dim->AsLiteralOrDie()->bits().ToUint64());
    bit_count *= dim_size + (dims_are_max() ? 1 : 0);
  }
  return bit_count;
}

std::string UnpackedArrayType::EmitWithIdentifier(
    LineInfo* line_info, std::string_view identifier) const {
  LineInfoStart(line_info, this);
  std::string result =
      element_type()->EmitWithIdentifier(line_info, identifier);
  for (Expression* dim : dims()) {
    // In SystemVerilog unpacked arrays can be specified using only the size
    // rather than a range.
    if (file()->use_system_verilog()) {
      absl::StrAppendFormat(&result, "[%s]", dim->Emit(line_info));
    } else {
      absl::StrAppendFormat(&result, "[0:%s]", WidthToLimit(line_info, dim));
    }
  }
  LineInfoEnd(line_info, this);
  return result;
}

std::string Def::Emit(LineInfo* line_info) const {
  std::string result = EmitNoSemi(line_info);
  if (init().has_value()) {
    absl::StrAppend(&result, " = ", (*init())->Emit(line_info));
  }
  absl::StrAppend(&result, ";");
  return result;
}

std::string Def::EmitNoSemi(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  std::string kind_str = DataKindToString(data_kind());
  std::string data_type_str =
      data_type()->EmitWithIdentifier(line_info, GetName());
  std::string result = CombineKindAndDataType(kind_str, data_type_str);

  LineInfoEnd(line_info, this);
  return result;
}

IntegerDef::IntegerDef(std::string_view name, VerilogFile* file,
                       const SourceInfo& loc)
    : Def(name, DataKind::kInteger, file->IntegerType(loc), file, loc) {}

IntegerDef::IntegerDef(std::string_view name, DataType* data_type,
                       Expression* init, VerilogFile* file,
                       const SourceInfo& loc)
    : Def(name, DataKind::kInteger, data_type, init, file, loc) {}

namespace {

// "Match" statement for emitting a ModuleMember.
std::string EmitModuleMember(LineInfo* line_info, const ModuleMember& member) {
  return absl::visit([=](auto* d) { return d->Emit(line_info); }, member);
}

// Visitor for emitting a VerilogPackageMember.
std::string EmitVerilogPackageMember(LineInfo* line_info,
                                     const VerilogPackageMember& member) {
  return absl::visit([=](auto* d) { return d->Emit(line_info); }, member);
}

}  // namespace

std::string ModuleSection::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  std::vector<std::string> elements;
  for (const ModuleMember& member : members_) {
    if (std::holds_alternative<ModuleSection*>(member)) {
      if (std::get<ModuleSection*>(member)->members_.empty()) {
        continue;
      }
    }
    elements.push_back(EmitModuleMember(line_info, member));
    LineInfoIncrease(line_info, 1);
  }
  if (!elements.empty()) {
    LineInfoIncrease(line_info, -1);
  }
  LineInfoEnd(line_info, this);
  return absl::StrJoin(elements, "\n");
}

std::string VerilogPackageSection::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  std::vector<std::string> elements;
  for (const VerilogPackageMember& member : members_) {
    if (std::holds_alternative<VerilogPackageSection*>(member)) {
      if (std::get<VerilogPackageSection*>(member)->members_.empty()) {
        continue;
      }
    }
    elements.push_back(EmitVerilogPackageMember(line_info, member));
    LineInfoIncrease(line_info, 1);
  }
  if (!elements.empty()) {
    LineInfoIncrease(line_info, -1);
  }
  LineInfoEnd(line_info, this);
  return absl::StrJoin(elements, "\n");
}

std::string ContinuousAssignment::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  std::string lhs = lhs_->Emit(line_info);
  std::string rhs = rhs_->Emit(line_info);
  LineInfoEnd(line_info, this);
  return absl::StrFormat("assign %s = %s;", lhs, rhs);
}

std::string Comment::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  LineInfoIncrease(line_info, NumberOfNewlines(text_));
  LineInfoEnd(line_info, this);
  return absl::StrCat("// ", absl::StrReplaceAll(text_, {{"\n", "\n// "}}));
}

std::string InlineVerilogStatement::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  LineInfoIncrease(line_info, NumberOfNewlines(text_));
  LineInfoEnd(line_info, this);
  return text_;
}

std::string InlineVerilogRef::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  LineInfoIncrease(line_info, NumberOfNewlines(name_));
  LineInfoEnd(line_info, this);
  return name_;
}

std::string ConcurrentAssertion::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  LineInfoIncrease(line_info, 1);

  // The $fatal statement takes finish_number as the first argument which is a
  // value in the set {0, 1, 2}. This value "sets the level of diagnostic
  // information reported by the tool" (from IEEE Std 1800-2017, 20.10).
  //
  // XLS emits asserts taking combinational inputs, so a deferred
  // immediate assertion is used.
  constexpr int64_t kFinishNumber = 0;
  std::string result;
  if (!label_.empty()) {
    absl::StrAppendFormat(&result, "%s: ", label_);
  }
  absl::StrAppendFormat(&result, "assert property (@(%s) ",
                        clocking_event_->Emit(line_info));
  if (disable_iff_.has_value()) {
    absl::StrAppendFormat(&result, "disable iff (%s) ",
                          disable_iff_.value()->Emit(line_info));
  }
  absl::StrAppendFormat(&result, "%s) else $fatal(%d%s);",
                        condition_->Emit(line_info), kFinishNumber,
                        error_message_.empty()
                            ? ""
                            : absl::StrFormat(", \"%s\"", error_message_));
  LineInfoEnd(line_info, this);
  return result;
}

DeferredImmediateAssertion::DeferredImmediateAssertion(
    Expression* condition, std::optional<Expression*> disable_iff,
    std::string_view label, std::string_view error_message, VerilogFile* file,
    const SourceInfo& loc)
    : Statement(file, loc), label_(label), error_message_(error_message) {
  if (disable_iff.has_value()) {
    condition_ = file->LogicalOr(disable_iff.value(), condition, loc);
  } else {
    condition_ = condition;
  }
}

std::string DeferredImmediateAssertion::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  LineInfoIncrease(line_info, 1);

  // The $fatal statement takes finish_number as the first argument which is a
  // value in the set {0, 1, 2}. This value "sets the level of diagnostic
  // information reported by the tool" (from IEEE Std 1800-2017, 20.10).
  constexpr int64_t kFinishNumber = 0;
  std::string result;
  if (!label_.empty()) {
    absl::StrAppendFormat(&result, "%s: ", label_);
  }

  absl::StrAppendFormat(&result, "assert final (%s) else $fatal(%d%s);",
                        condition_->Emit(line_info), kFinishNumber,
                        error_message_.empty()
                            ? ""
                            : absl::StrFormat(", \"%s\"", error_message_));
  LineInfoEnd(line_info, this);
  return result;
}

std::string Cover::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  // Coverpoints don't work without clock sources. Don't emit them in that case.
  LineInfoIncrease(line_info, NumberOfNewlines(label_));
  std::string clock = clk_->Emit(line_info);
  std::string condition = condition_->Emit(line_info);
  LineInfoEnd(line_info, this);
  return absl::StrFormat("%s: cover property (@(posedge %s) %s);", label_,
                         clock, condition);
}

std::string SystemTaskCall::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  if (args_.has_value()) {
    std::string result = absl::StrFormat(
        "$%s(%s);", name_,
        absl::StrJoin(*args_, ", ", [=](std::string* out, Expression* e) {
          absl::StrAppend(out, e->Emit(line_info));
        }));
    LineInfoEnd(line_info, this);
    return result;
  }
  LineInfoEnd(line_info, this);
  return absl::StrFormat("$%s;", name_);
}

std::string SystemFunctionCall::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  if (args_.has_value()) {
    LineInfoIncrease(line_info, NumberOfNewlines(name_));
    std::string arg_list =
        absl::StrJoin(*args_, ", ", [=](std::string* out, Expression* e) {
          absl::StrAppend(out, e->Emit(line_info));
        });
    LineInfoEnd(line_info, this);
    return absl::StrFormat("$%s(%s)", name_, arg_list);
  }
  LineInfoIncrease(line_info, NumberOfNewlines(name_));
  LineInfoEnd(line_info, this);
  return absl::StrFormat("$%s", name_);
}

std::string Module::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  std::string result = absl::StrCat("module ", name_);
  if (ports_.empty()) {
    absl::StrAppend(&result, ";\n");
    LineInfoIncrease(line_info, 1);
  } else {
    absl::StrAppend(&result, "(\n  ");
    LineInfoIncrease(line_info, 1);
    absl::StrAppend(
        &result,
        absl::StrJoin(ports_, ",\n  ", [=](std::string* out, const Port& port) {
          std::string wire_str = port.wire->EmitNoSemi(line_info);
          CHECK(CannotStripWhitespace(wire_str));
          absl::StrAppendFormat(out, "%s %s", ToString(port.direction),
                                wire_str);
          LineInfoIncrease(line_info, 1);
        }));
    absl::StrAppend(&result, "\n);\n");
    LineInfoIncrease(line_info, 1);
  }
  absl::StrAppend(&result, Indent(top_.Emit(line_info)), "\n");
  LineInfoIncrease(line_info, 1);
  absl::StrAppend(&result, "endmodule");
  LineInfoEnd(line_info, this);
  return result;
}

std::string VerilogPackage::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);

  std::string result = absl::StrCat("package ", name_, ";\n");
  LineInfoIncrease(line_info, 1);

  absl::StrAppend(&result, Indent(top_.Emit(line_info)), "\n");
  LineInfoIncrease(line_info, 1);

  absl::StrAppend(&result, "endpackage");
  LineInfoEnd(line_info, this);
  return result;
}

std::string Literal::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  LineInfoEnd(line_info, this);
  // Large zero constants can be emitted as a simple zero-extend.
  constexpr int64_t kLargeBitCount = 1024;
  if (bits_.bit_count() > kLargeBitCount && bits_.IsZero()) {
    std::string_view format_specifier;
    switch (format_) {
      case FormatPreference::kBinary:
        format_specifier = "b";
        break;
      case FormatPreference::kHex:
        format_specifier = declared_as_signed_ ? "sh" : "h";
        break;
      case FormatPreference::kUnsignedDecimal:
      default:
        format_specifier = "d";
        break;
    }
    return absl::StrFormat("%d'%s0", bits_.bit_count(), format_specifier);
  }
  if (format_ == FormatPreference::kDefault) {
    CHECK_LE(bits_.bit_count(), 32);
    return absl::StrFormat(
        "%s", BitsToString(bits_, FormatPreference::kUnsignedDecimal));
  }
  if (format_ == FormatPreference::kUnsignedDecimal) {
    std::string prefix;
    if (emit_bit_count_) {
      prefix = absl::StrFormat("%d'd", effective_bit_count_);
    }
    return absl::StrFormat(
        "%s%s", prefix,
        BitsToString(bits_, FormatPreference::kUnsignedDecimal));
  }
  if (format_ == FormatPreference::kBinary) {
    return absl::StrFormat(
        "%d'b%s", effective_bit_count_,
        BitsToRawDigits(bits_, format_, /*emit_leading_zeros=*/true));
  }
  CHECK_EQ(format_, FormatPreference::kHex);
  const std::string raw_digits = BitsToRawDigits(bits_, FormatPreference::kHex,
                                                 /*emit_leading_zeros=*/true);
  return declared_as_signed_
             ? absl::StrFormat("%d'sh%s", effective_bit_count_, raw_digits)
             : absl::StrFormat("%d'h%s", effective_bit_count_, raw_digits);
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
std::string QuotedString::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  LineInfoIncrease(line_info, NumberOfNewlines(str_));
  LineInfoEnd(line_info, this);
  return absl::StrFormat("\"%s\"", str_);
}

static bool IsScalarLogicRef(IndexableExpression* expr) {
  auto* logic_ref = dynamic_cast<LogicRef*>(expr);
  return logic_ref != nullptr && logic_ref->def()->data_type()->IsScalar();
}

std::string Slice::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  if (IsScalarLogicRef(subject_)) {
    // If subject is scalar (no width given in declaration) then avoid slicing
    // as this is invalid Verilog. The only valid hi/lo values are zero.
    // TODO(https://github.com/google/xls/issues/43): Avoid this special case
    // and perform the equivalent logic at a higher abstraction level than VAST.
    CHECK(hi_->IsLiteralWithValue(0)) << hi_->Emit(nullptr);
    CHECK(lo_->IsLiteralWithValue(0)) << lo_->Emit(nullptr);
    std::string result = subject_->Emit(line_info);
    LineInfoEnd(line_info, this);
    return result;
  }
  std::string subject = subject_->Emit(line_info);
  std::string hi = hi_->Emit(line_info);
  std::string lo = lo_->Emit(line_info);
  LineInfoEnd(line_info, this);
  return absl::StrFormat("%s[%s:%s]", subject, hi, lo);
}

std::string PartSelect::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  std::string subject = subject_->Emit(line_info);
  std::string start = start_->Emit(line_info);
  std::string width = width_->Emit(line_info);
  LineInfoEnd(line_info, this);
  return absl::StrFormat("%s[%s +: %s]", subject, start, width);
}

std::string Index::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  if (IsScalarLogicRef(subject_)) {
    // If subject is scalar (no width given in declaration) then avoid indexing
    // as this is invalid Verilog. The only valid index values are zero.
    // TODO(https://github.com/google/xls/issues/43): Avoid this special case
    // and perform the equivalent logic at a higher abstraction level than VAST.
    CHECK(index_->IsLiteralWithValue(0)) << absl::StreamFormat(
        "%s[%s]", subject_->Emit(nullptr), index_->Emit(nullptr));
    std::string result = subject_->Emit(line_info);
    LineInfoEnd(line_info, this);
    return result;
  }
  std::string subject = subject_->Emit(line_info);
  std::string index = index_->Emit(line_info);
  LineInfoEnd(line_info, this);
  return absl::StrFormat("%s[%s]", subject, index);
}

// Returns the given string wrapped in parentheses.
static std::string ParenWrap(std::string_view s) {
  return absl::StrFormat("(%s)", s);
}

std::string Ternary::Emit(LineInfo* line_info) const {
  auto maybe_paren_wrap = [this, line_info](Expression* e) {
    if (e->precedence() <= precedence()) {
      return ParenWrap(e->Emit(line_info));
    }
    return e->Emit(line_info);
  };
  LineInfoStart(line_info, this);
  std::string result = absl::StrFormat("%s ? %s : %s", maybe_paren_wrap(test_),
                                       maybe_paren_wrap(consequent_),
                                       maybe_paren_wrap(alternate_));
  LineInfoEnd(line_info, this);
  return result;
}

std::string Parameter::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  LineInfoIncrease(line_info, NumberOfNewlines(name_));
  std::string result = absl::StrFormat(
      "parameter %s = %s;", def_ ? def_->EmitNoSemi(line_info) : name_,
      rhs_->Emit(line_info));
  LineInfoEnd(line_info, this);
  return result;
}

std::string Typedef::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  LineInfoIncrease(line_info, NumberOfNewlines(def_->GetName()));
  std::string result = absl::StrFormat("typedef %s", def_->Emit(line_info));
  LineInfoEnd(line_info, this);
  return result;
}

std::string TypedefType::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  std::string result = type_def_->GetName();
  LineInfoEnd(line_info, this);
  return result;
}

std::string ExternType::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  std::string result = name_;
  LineInfoEnd(line_info, this);
  return result;
}

std::string ExternPackageType::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  std::string result = absl::StrCat(package_name_, "::", type_name_);
  LineInfoEnd(line_info, this);
  return result;
}

std::string Enum::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  std::string result = "enum {\n";
  if (kind_ != DataKind::kUntypedEnum) {
    std::string kind_str = DataKindToString(kind_);
    std::string data_type_str = BaseType()->Emit(line_info);
    std::string underlying_type_str =
        CombineKindAndDataType(kind_str, data_type_str);
    result = absl::StrFormat("enum %s {\n", underlying_type_str);
  }
  LineInfoIncrease(line_info, 1);
  for (int i = 0; i < members_.size(); i++) {
    LineInfoIncrease(line_info, 1);
    std::string member_str = members_[i]->Emit(line_info);
    if (i == members_.size() - 1) {
      absl::StrAppend(&member_str, "\n");
    } else {
      absl::StrAppend(&member_str, ",\n");
    }
    absl::StrAppend(&result, Indent(member_str));
  }
  absl::StrAppend(&result, "}");
  LineInfoEnd(line_info, this);
  return result;
}

EnumMemberRef* Enum::AddMember(std::string_view name, Expression* rhs,
                               const SourceInfo& loc) {
  members_.push_back(file()->Make<EnumMember>(loc, name, rhs));
  return file()->Make<EnumMemberRef>(loc, this, members_.back());
}

EnumMemberRef* EnumMemberRef::Duplicate() const {
  return file()->Make<EnumMemberRef>(loc(), enum_def(), member());
}

std::string EnumMember::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  std::string result = absl::StrFormat("%s = %s", name_, rhs_->Emit(line_info));
  LineInfoEnd(line_info, this);
  return result;
}

absl::StatusOr<int64_t> Struct::FlatBitCountAsInt64() const {
  int64_t result = 0;
  for (const Def* next : members_) {
    XLS_ASSIGN_OR_RETURN(int64_t def_bit_count,
                         next->data_type()->FlatBitCountAsInt64());
    result += def_bit_count;
  }
  return result;
}

std::string Struct::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  std::string result = "struct packed {\n";
  LineInfoIncrease(line_info, 1);
  for (const Def* next : members_) {
    LineInfoIncrease(line_info, 1);
    absl::StrAppend(&result, Indent(next->Emit(line_info)), "\n");
  }
  absl::StrAppend(&result, "}");
  LineInfoEnd(line_info, this);
  return result;
}

std::string LocalParamItem::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  LineInfoIncrease(line_info, NumberOfNewlines(name_));
  std::string result = absl::StrFormat("%s = %s", name_, rhs_->Emit(line_info));
  LineInfoEnd(line_info, this);
  return result;
}

std::string LocalParam::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  std::string result = "localparam";
  if (items_.size() == 1) {
    absl::StrAppend(&result, " ", items_[0]->Emit(line_info), ";");
    LineInfoEnd(line_info, this);
    return result;
  }
  absl::StrAppend(&result, "\n  ");
  LineInfoIncrease(line_info, 1);
  auto append_item = [=](std::string* out, LocalParamItem* item) {
    absl::StrAppend(out, item->Emit(line_info));
    LineInfoIncrease(line_info, 1);
  };
  absl::StrAppend(&result, absl::StrJoin(items_, ",\n  ", append_item), ";");
  if (items_.size() > 1) {
    // StrJoin adds a fencepost number of newlines, so we need to subtract 1
    // to get the total number correct.
    LineInfoIncrease(line_info, -1);
  }
  LineInfoEnd(line_info, this);
  return result;
}

std::string BinaryInfix::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  auto is_unary_reduction = [](Expression* e) {
    return e->IsUnary() && e->AsUnaryOrDie()->IsReduction();
  };

  // Equal precedence operators are evaluated left-to-right so LHS only needs to
  // be wrapped if its precedence is strictly less than this operators. The RHS,
  // however, must be wrapped if its less than or equal precedence. Unary
  // reduction operations should be wrapped in parenthesis unconditionally
  // because some consumers of verilog emit warnings/errors for this
  // error-prone construct (e.g., `|x || |y`)
  std::string lhs_string =
      (lhs_->precedence() < precedence() || is_unary_reduction(lhs_))
          ? ParenWrap(lhs_->Emit(line_info))
          : lhs_->Emit(line_info);
  std::string rhs_string =
      (rhs_->precedence() <= precedence() || is_unary_reduction(rhs_))
          ? ParenWrap(rhs_->Emit(line_info))
          : rhs_->Emit(line_info);
  LineInfoEnd(line_info, this);
  return absl::StrFormat("%s %s %s", lhs_string, op_, rhs_string);
}

std::string Concat::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  std::string result;
  if (replication_ != nullptr) {
    absl::StrAppend(&result, "{", replication_->Emit(line_info));
  }
  absl::StrAppendFormat(
      &result, "{%s}",
      absl::StrJoin(args_, ", ", [=](std::string* out, Expression* e) {
        absl::StrAppend(out, e->Emit(line_info));
      }));
  if (replication_ != nullptr) {
    absl::StrAppend(&result, "}");
  }
  LineInfoEnd(line_info, this);
  return result;
}

std::string ArrayAssignmentPattern::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  std::string result = absl::StrFormat(
      "'{%s}", absl::StrJoin(args_, ", ", [=](std::string* out, Expression* e) {
        absl::StrAppend(out, e->Emit(line_info));
      }));
  LineInfoEnd(line_info, this);
  return result;
}

std::string Unary::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  // Nested unary ops should be wrapped in parentheses as this is required by
  // some consumers of Verilog.
  std::string result =
      absl::StrFormat("%s%s", op_,
                      ((arg_->precedence() < precedence()) || arg_->IsUnary())
                          ? ParenWrap(arg_->Emit(line_info))
                          : arg_->Emit(line_info));
  LineInfoEnd(line_info, this);
  return result;
}

StatementBlock* Case::AddCaseArm(CaseLabel label) {
  arms_.push_back(file()->Make<CaseArm>(SourceInfo(), label));
  return arms_.back()->statements();
}

static std::string CaseTypeToString(CaseType case_type) {
  std::string keyword;
  switch (case_type.keyword) {
    case CaseKeyword::kCase:
      keyword = "case";
      break;
    case CaseKeyword::kCasez:
      keyword = "casez";
      break;
    default:
      LOG(FATAL) << "Unexpected CaseKeyword with value "
                 << static_cast<int>(case_type.keyword);
  }

  if (case_type.modifier.has_value()) {
    std::string modifier;
    switch (*case_type.modifier) {
      case CaseModifier::kUnique:
        modifier = "unique";
        break;
      default:
        LOG(FATAL) << "Unexpected CaseModifier with value "
                   << static_cast<int>(*case_type.modifier);
    }
    return absl::StrCat(modifier, " ", keyword);
  }
  return keyword;
}

std::string Case::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  std::string result = absl::StrFormat(
      "%s (%s)\n", CaseTypeToString(case_type_), subject_->Emit(line_info));
  LineInfoIncrease(line_info, 1);
  for (auto& arm : arms_) {
    std::string arm_string = arm->Emit(line_info);
    std::string stmts_string = arm->statements()->Emit(line_info);
    absl::StrAppend(&result,
                    Indent(absl::StrFormat("%s: %s", arm_string, stmts_string)),
                    "\n");
    LineInfoIncrease(line_info, 1);
  }
  absl::StrAppend(&result, "endcase");
  LineInfoEnd(line_info, this);
  return result;
}

Conditional::Conditional(Expression* condition, VerilogFile* file,
                         const SourceInfo& loc)
    : Statement(file, loc),
      condition_(condition),
      consequent_(file->Make<StatementBlock>(SourceInfo())) {}

StatementBlock* Conditional::AddAlternate(Expression* condition) {
  // The conditional must not have been previously closed with an unconditional
  // alternate ("else").
  CHECK(alternates_.empty() || alternates_.back().first != nullptr);
  alternates_.push_back(
      {condition, file()->Make<StatementBlock>(SourceInfo())});
  return alternates_.back().second;
}

std::string Conditional::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  std::string result;
  std::string cond = condition_->Emit(line_info);
  std::string conseq = consequent()->Emit(line_info);
  absl::StrAppendFormat(&result, "if (%s) %s", cond, conseq);
  for (auto& alternate : alternates_) {
    absl::StrAppend(&result, " else ");
    if (alternate.first != nullptr) {
      absl::StrAppendFormat(&result, "if (%s) ",
                            alternate.first->Emit(line_info));
    }
    absl::StrAppend(&result, alternate.second->Emit(line_info));
  }
  LineInfoEnd(line_info, this);
  return result;
}

std::string ConditionalDirectiveKindToString(ConditionalDirectiveKind kind) {
  switch (kind) {
    case ConditionalDirectiveKind::kIfdef:
      return "`ifdef";
    case ConditionalDirectiveKind::kIfndef:
      return "`ifndef";
  }
  LOG(ERROR) << "Unexpected ConditionalDirectiveKind with value "
             << static_cast<int>(kind);
  return "";
}

StatementConditionalDirective::StatementConditionalDirective(
    ConditionalDirectiveKind kind, std::string identifier, VerilogFile* file,
    const SourceInfo& loc)
    : Statement(file, loc),
      kind_(kind),
      identifier_(std::move(identifier)),
      consequent_(file->Make<MacroStatementBlock>(loc)) {}

MacroStatementBlock* StatementConditionalDirective::AddAlternate(
    std::string identifier) {
  CHECK(alternates_.empty() || !alternates_.back().first.empty());
  alternates_.push_back(
      {std::move(identifier), file()->Make<MacroStatementBlock>(SourceInfo())});
  return alternates_.back().second;
}

std::string StatementConditionalDirective::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  std::string result;

  absl::StrAppend(&result, ConditionalDirectiveKindToString(kind_), " ",
                  identifier_, "\n", consequent_->Emit(line_info));
  LineInfoIncrease(line_info, 2);

  for (const auto& [identifier, block] : alternates_) {
    if (identifier.empty()) {
      absl::StrAppend(&result, "`else\n", block->Emit(line_info));
    } else {
      absl::StrAppend(&result, "`elsif ", identifier, "\n",
                      block->Emit(line_info));
    }
    LineInfoIncrease(line_info, 2);
  }
  absl::StrAppend(&result, "`endif");
  LineInfoIncrease(line_info, 1);
  LineInfoEnd(line_info, this);

  return result;
}

ModuleConditionalDirective::ModuleConditionalDirective(
    ConditionalDirectiveKind kind, std::string identifier, VerilogFile* file,
    const SourceInfo& loc)
    : VastNode(file, loc),
      kind_(kind),
      identifier_(std::move(identifier)),
      consequent_(file->Make<ModuleSection>(loc)) {}

ModuleSection* ModuleConditionalDirective::AddAlternate(
    std::string identifier) {
  CHECK(alternates_.empty() || !alternates_.back().first.empty());
  alternates_.push_back(
      {std::move(identifier), file()->Make<ModuleSection>(SourceInfo())});
  return alternates_.back().second;
}

std::string ModuleConditionalDirective::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  std::vector<std::string> pieces;
  pieces.push_back(
      absl::StrCat(ConditionalDirectiveKindToString(kind_), " ", identifier_));
  pieces.push_back(consequent_->Emit(line_info));
  LineInfoIncrease(line_info, 2);

  for (const auto& [identifier, block] : alternates_) {
    if (identifier.empty()) {
      pieces.push_back("`else");
      pieces.push_back(block->Emit(line_info));
    } else {
      pieces.push_back(absl::StrCat("`elsif ", identifier));
      pieces.push_back(block->Emit(line_info));
    }
    LineInfoIncrease(line_info, 2);
  }
  pieces.push_back("`endif");
  LineInfoIncrease(line_info, 1);
  LineInfoEnd(line_info, this);

  return absl::StrJoin(pieces, "\n");
}

WhileStatement::WhileStatement(Expression* condition, VerilogFile* file,
                               const SourceInfo& loc)
    : Statement(file, loc),
      condition_(condition),
      statements_(file->Make<StatementBlock>(loc)) {}

std::string WhileStatement::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  std::string condition = condition_->Emit(line_info);
  std::string stmts = statements()->Emit(line_info);
  LineInfoEnd(line_info, this);
  return absl::StrFormat("while (%s) %s", condition, stmts);
}

RepeatStatement::RepeatStatement(Expression* repeat_count, VerilogFile* file,
                                 const SourceInfo& loc)
    : Statement(file, loc),
      repeat_count_(repeat_count),
      statements_(file->Make<StatementBlock>(loc)) {}

std::string RepeatStatement::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  std::string repeat_count = repeat_count_->Emit(line_info);
  std::string stmts = statements()->Emit(line_info);
  LineInfoEnd(line_info, this);
  return absl::StrFormat("repeat (%s) %s", repeat_count, stmts);
}

std::string EventControl::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  std::string result =
      absl::StrFormat("@(%s);", event_expression_->Emit(line_info));
  LineInfoEnd(line_info, this);
  return result;
}

std::string PosEdge::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  std::string result =
      absl::StrFormat("posedge %s", expression_->Emit(line_info));
  LineInfoEnd(line_info, this);
  return result;
}

std::string NegEdge::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  std::string result =
      absl::StrFormat("negedge %s", expression_->Emit(line_info));
  LineInfoEnd(line_info, this);
  return result;
}

std::string DelayStatement::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  std::string delay_str = delay_->precedence() < Expression::kMaxPrecedence
                              ? ParenWrap(delay_->Emit(line_info))
                              : delay_->Emit(line_info);
  if (delayed_statement_ != nullptr) {
    std::string result = absl::StrFormat("#%s %s", delay_str,
                                         delayed_statement_->Emit(line_info));
    LineInfoEnd(line_info, this);
    return result;
  }
  LineInfoEnd(line_info, this);
  return absl::StrFormat("#%s;", delay_str);
}

std::string WaitStatement::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  std::string result = absl::StrFormat("wait(%s);", event_->Emit(line_info));
  LineInfoEnd(line_info, this);
  return result;
}

std::string Forever::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  std::string result = absl::StrCat("forever ", statement_->Emit(line_info));
  LineInfoEnd(line_info, this);
  return result;
}

std::string BlockingAssignment::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  std::string lhs_str = lhs()->Emit(line_info);
  std::string rhs_str = rhs()->Emit(line_info);
  LineInfoEnd(line_info, this);
  return absl::StrFormat("%s = %s;", lhs_str, rhs_str);
}

std::string NonblockingAssignment::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  std::string lhs_str = lhs()->Emit(line_info);
  std::string rhs_str = rhs()->Emit(line_info);
  LineInfoEnd(line_info, this);
  return absl::StrFormat("%s <= %s;", lhs_str, rhs_str);
}

std::string ReturnStatement::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  std::string expr = expr_->Emit(line_info);
  LineInfoEnd(line_info, this);
  return absl::StrFormat("return %s;", expr);
}

StructuredProcedure::StructuredProcedure(VerilogFile* file,
                                         const SourceInfo& loc)
    : VastNode(file, loc), statements_(file->Make<StatementBlock>(loc)) {}

namespace {

std::string EmitSensitivityListElement(LineInfo* line_info,
                                       const SensitivityListElement& element) {
  return absl::visit(
      Visitor{[](ImplicitEventExpression e) -> std::string { return "*"; },
              [=](PosEdge* p) -> std::string { return p->Emit(line_info); },
              [=](NegEdge* n) -> std::string { return n->Emit(line_info); },
              [=](LogicRef* s) -> std::string { return s->Emit(line_info); }},
      element);
}

}  // namespace

std::string AlwaysBase::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  LineInfoIncrease(line_info, NumberOfNewlines(name()));
  std::string sensitivity_list = absl::StrJoin(
      sensitivity_list_, " or ",
      [=](std::string* out, const SensitivityListElement& e) {
        absl::StrAppend(out, EmitSensitivityListElement(line_info, e));
      });
  std::string statements = statements_->Emit(line_info);
  LineInfoEnd(line_info, this);
  return absl::StrFormat("%s @ (%s) %s", name(), sensitivity_list, statements);
}

std::string AlwaysComb::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  LineInfoIncrease(line_info, NumberOfNewlines(name()));
  std::string result =
      absl::StrFormat("%s %s", name(), statements_->Emit(line_info));
  LineInfoEnd(line_info, this);
  return result;
}

std::string Initial::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  std::string result = absl::StrCat("initial ", statements_->Emit(line_info));
  LineInfoEnd(line_info, this);
  return result;
}

AlwaysFlop::AlwaysFlop(LogicRef* clk, Reset rst, VerilogFile* file,
                       const SourceInfo& loc)
    : VastNode(file, loc),
      clk_(clk),
      rst_(rst),
      top_block_(file->Make<StatementBlock>(loc)) {
  // Reset signal specified. Construct conditional which switches the reset
  // signal.
  Expression* rst_condition;
  if (rst_->active_low) {
    rst_condition = file->LogicalNot(rst_->signal, loc);
  } else {
    rst_condition = rst_->signal;
  }
  Conditional* conditional = top_block_->Add<Conditional>(loc, rst_condition);
  reset_block_ = conditional->consequent();
  assignment_block_ = conditional->AddAlternate();
}

AlwaysFlop::AlwaysFlop(LogicRef* clk, VerilogFile* file, const SourceInfo& loc)
    : VastNode(file, loc),
      clk_(clk),
      top_block_(file->Make<StatementBlock>(loc)) {
  // No reset signal specified.
  reset_block_ = nullptr;
  assignment_block_ = top_block_;
}

void AlwaysFlop::AddRegister(LogicRef* reg, Expression* reg_next,
                             const SourceInfo& loc, Expression* reset_value) {
  if (reset_value != nullptr) {
    CHECK(reset_block_ != nullptr);
    reset_block_->Add<NonblockingAssignment>(loc, reg, reset_value);
  }
  assignment_block_->Add<NonblockingAssignment>(loc, reg, reg_next);
}

std::string AlwaysFlop::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  std::string result;
  std::string sensitivity_list =
      absl::StrCat("posedge ", clk_->Emit(line_info));
  if (rst_.has_value() && rst_->asynchronous) {
    absl::StrAppendFormat(&sensitivity_list, " or %s %s",
                          (rst_->active_low ? "negedge" : "posedge"),
                          rst_->signal->Emit(line_info));
  }
  absl::StrAppendFormat(&result, "always @ (%s) %s", sensitivity_list,
                        top_block_->Emit(line_info));
  LineInfoEnd(line_info, this);
  return result;
}

std::string Instantiation::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  std::string result = absl::StrCat(module_name_, " ");
  LineInfoIncrease(line_info, NumberOfNewlines(module_name_));
  auto append_connection = [=](std::string* out, const Connection& parameter) {
    absl::StrAppendFormat(out, ".%s(%s)", parameter.port_name,
                          parameter.expression->Emit(line_info));
    LineInfoIncrease(line_info, 1);
  };
  if (!parameters_.empty()) {
    absl::StrAppend(&result, "#(\n  ");
    LineInfoIncrease(line_info, 1);
    absl::StrAppend(&result,
                    absl::StrJoin(parameters_, ",\n  ", append_connection),
                    "\n) ");
  }
  absl::StrAppend(&result, instance_name_, " (\n  ");
  LineInfoIncrease(line_info, NumberOfNewlines(instance_name_) + 1);
  absl::StrAppend(
      &result, absl::StrJoin(connections_, ",\n  ", append_connection), "\n)");
  absl::StrAppend(&result, ";");
  LineInfoEnd(line_info, this);
  return result;
}

std::string TemplateInstantiation::Emit(LineInfo* line_info) const {
  LineInfoStart(line_info, this);
  std::vector<std::string> replacements;
  absl::StatusOr<CodeTemplate> code_template =
      CodeTemplate::Create(template_text_);
  CHECK(code_template.ok());  // Already verified earlier.
  std::string result = code_template->Substitute([&](std::string_view tmpl) {
    if (tmpl == "fn") {
      return instance_name_;
    }
    auto found =
        std::find_if(connections_.begin(), connections_.end(),
                     [&](const Connection& c) { return c.port_name == tmpl; });
    CHECK(found != connections_.end())  // Should've been verified earlier
        << "ExternInstantiation: can't map: template value '" << tmpl << "'";
    return found->expression->Emit(line_info);
  });
  absl::StrAppend(&result, ";");
  LineInfoIncrease(line_info, NumberOfNewlines(result));
  LineInfoEnd(line_info, this);
  return result;
}
}  // namespace verilog
}  // namespace xls
