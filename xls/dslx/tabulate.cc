// Copyright 2026 The XLS Authors
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

#include "xls/dslx/tabulate.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/attribute_data.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/bytecode/bytecode.h"
#include "xls/dslx/bytecode/bytecode_emitter.h"
#include "xls/dslx/bytecode/bytecode_interpreter.h"
#include "xls/dslx/domain_utils.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/ir/format_preference.h"

namespace xls::dslx {
namespace {

constexpr int64_t kMaxDomainPoints = 2000;
constexpr size_t kMinColumnWidth = 3;

struct ColumnSpec {
  std::string name;
  const Type* type = nullptr;
};

std::optional<Function*> FindFunctionByName(Module* module,
                                            std::string_view fn_name) {
  if (std::optional<Function*> fn = module->GetFunction(fn_name);
      fn.has_value()) {
    return *fn;
  }
  if (std::optional<FuzzTestFunction*> ftf =
          module->GetMember<FuzzTestFunction>(fn_name);
      ftf.has_value()) {
    return &(*ftf)->fn();
  }
  if (std::optional<QuickCheck*> qc = module->GetMember<QuickCheck>(fn_name);
      qc.has_value()) {
    return (*qc)->fn();
  }
  if (std::optional<TestFunction*> tf =
          module->GetMember<TestFunction>(fn_name);
      tf.has_value()) {
    return &(*tf)->fn();
  }
  return std::nullopt;
}

absl::StatusOr<Function*> GetTabulatedFunction(Module* module,
                                               std::string_view fn_name) {
  if (fn_name.empty()) {
    return absl::InvalidArgumentError(
        "Function name must be specified for tabulation");
  }
  std::optional<Function*> fn_opt = FindFunctionByName(module, fn_name);
  if (!fn_opt.has_value()) {
    return absl::NotFoundError(absl::StrFormat(
        "Function `%s` not found in module `%s`", fn_name, module->name()));
  }
  Function* fn = *fn_opt;
  if (fn->IsParametric()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Function `%s` is parametric; #[tabulate] only supports non-parametric "
        "functions",
        fn->identifier()));
  }

  for (const Attribute* attr : fn->attributes()) {
    if (attr->attribute_kind() == AttributeKind::kTabulate) {
      return fn;
    }
  }
  return absl::InvalidArgumentError(absl::StrFormat(
      "Function `%s` does not have a #[tabulate] attribute", fn->identifier()));
}

absl::StatusOr<std::vector<ColumnSpec>> BuildColumns(
    const Function& fn, const FunctionType& fn_type) {
  if (fn.params().empty()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Function `%s` has no parameters; #[tabulate] requires at least one "
        "parameter",
        fn.identifier()));
  }

  const Type& return_type = fn_type.return_type();
  size_t output_cols = 1;
  if (return_type.IsStruct()) {
    output_cols = return_type.AsStruct().size();
  } else if (return_type.IsTuple()) {
    output_cols = return_type.AsTuple().size();
  }
  if (output_cols == 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Function `%s` has an empty return type `%s`; #[tabulate] requires at "
        "least one return value",
        fn.identifier(), return_type.ToString()));
  }

  std::vector<ColumnSpec> columns;
  columns.reserve(fn.params().size() + output_cols);
  for (size_t i = 0; i < fn.params().size(); ++i) {
    columns.push_back(ColumnSpec{
        .name = fn.params()[i]->identifier(),
        .type = fn_type.params()[i].get(),
    });
  }

  if (return_type.IsStruct()) {
    const StructType& struct_type = return_type.AsStruct();
    for (size_t i = 0; i < struct_type.size(); ++i) {
      columns.push_back(ColumnSpec{
          .name = std::string(struct_type.GetMemberName(i)),
          .type = &struct_type.GetMemberType(i),
      });
    }
  } else if (return_type.IsTuple()) {
    const TupleType& tuple_type = return_type.AsTuple();
    for (size_t i = 0; i < tuple_type.size(); ++i) {
      columns.push_back(ColumnSpec{
          .name = absl::StrFormat("result_%d", i),
          .type = &tuple_type.GetMemberType(i),
      });
    }
  } else {
    columns.push_back(ColumnSpec{
        .name = "result",
        .type = &return_type,
    });
  }
  return columns;
}

// Evaluates `fn` on each input row and appends the unpacked return value(s)
// to the row in-place.
absl::Status EvaluateAndAppendOutputs(
    ImportData* import_data, TypeInfo* type_info, const Function& fn,
    const Type& return_type, size_t expected_total_cols,
    std::vector<std::vector<InterpValue>>& rows) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<BytecodeFunction> bf,
                       BytecodeEmitter::Emit(import_data, type_info, fn,
                                             /*caller_bindings=*/std::nullopt));
  for (std::vector<InterpValue>& row : rows) {
    XLS_ASSIGN_OR_RETURN(InterpValue result, BytecodeInterpreter::Interpret(
                                                 import_data, bf.get(), row));

    if (return_type.IsStruct() || return_type.IsTuple()) {
      const std::vector<InterpValue>& outputs = result.GetValuesOrDie();
      row.insert(row.end(), outputs.begin(), outputs.end());
    } else {
      row.push_back(std::move(result));
    }
    XLS_RET_CHECK_EQ(row.size(), expected_total_cols);
  }
  return absl::OkStatus();
}

std::string FormatValue(const InterpValue& val, const Type& type) {
  if (type.IsEnum()) {
    const EnumType& enum_type = type.AsEnum();
    const EnumDef& enum_def = enum_type.nominal_type();
    const std::vector<InterpValue>& members = enum_type.members();
    for (size_t i = 0; i < members.size(); ++i) {
      if (members[i] == val) {
        return enum_def.GetMemberName(i);
      }
    }
    return val.ToString(/*humanize=*/true);
  }
  if (IsBool(type)) {
    return val.IsTrue() ? "true" : "false";
  }
  if (type.IsStruct()) {
    const StructType& struct_type = type.AsStruct();
    std::string_view struct_name = struct_type.nominal_type().identifier();
    if (struct_type.size() == 0) {
      return absl::StrFormat("%s {}", struct_name);
    }
    const std::vector<InterpValue>& fields = val.GetValuesOrDie();
    std::vector<std::string> formatted_fields;
    formatted_fields.reserve(struct_type.size());
    for (size_t i = 0; i < struct_type.size(); ++i) {
      formatted_fields.push_back(absl::StrFormat(
          "%s: %s", struct_type.GetMemberName(i),
          FormatValue(fields[i], struct_type.GetMemberType(i))));
    }
    return absl::StrFormat("%s { %s }", struct_name,
                           absl::StrJoin(formatted_fields, ", "));
  }
  if (type.IsTuple()) {
    const TupleType& tuple_type = type.AsTuple();
    const std::vector<InterpValue>& elems = val.GetValuesOrDie();
    std::vector<std::string> formatted_elems;
    formatted_elems.reserve(tuple_type.size());
    for (size_t i = 0; i < tuple_type.size(); ++i) {
      formatted_elems.push_back(
          FormatValue(elems[i], tuple_type.GetMemberType(i)));
    }
    return absl::StrFormat("(%s)", absl::StrJoin(formatted_elems, ", "));
  }
  if (type.IsArray()) {
    const ArrayType& array_type = type.AsArray();
    const std::vector<InterpValue>& elems = val.GetValuesOrDie();
    std::vector<std::string> formatted_elems;
    formatted_elems.reserve(elems.size());
    for (const InterpValue& elem : elems) {
      formatted_elems.push_back(FormatValue(elem, array_type.element_type()));
    }
    return absl::StrFormat("[%s]", absl::StrJoin(formatted_elems, ", "));
  }
  return val.ToString(/*humanize=*/true, FormatPreference::kDefault);
}

std::string RenderMarkdownTable(
    absl::Span<const ColumnSpec> columns,
    absl::Span<const std::vector<InterpValue>> rows) {
  const size_t total_cols = columns.size();
  std::vector<size_t> widths(total_cols, 0);
  for (size_t c = 0; c < total_cols; ++c) {
    widths[c] = std::max(kMinColumnWidth, columns[c].name.size());
  }

  std::vector<std::vector<std::string>> formatted_rows;
  formatted_rows.reserve(rows.size());
  for (const std::vector<InterpValue>& row : rows) {
    std::vector<std::string> cells;
    cells.reserve(total_cols);
    for (size_t c = 0; c < total_cols; ++c) {
      std::string text = FormatValue(row[c], *columns[c].type);
      widths[c] = std::max(widths[c], text.size());
      cells.push_back(std::move(text));
    }
    formatted_rows.push_back(std::move(cells));
  }

  std::string out;
  absl::StrAppend(&out, "|");
  for (size_t c = 0; c < total_cols; ++c) {
    absl::StrAppendFormat(&out, " %-*s |", static_cast<int>(widths[c]),
                          columns[c].name);
  }
  absl::StrAppend(&out, "\n|");
  for (size_t c = 0; c < total_cols; ++c) {
    absl::StrAppend(&out, " ", std::string(widths[c], '-'), " |");
  }
  absl::StrAppend(&out, "\n");
  for (const std::vector<std::string>& row : formatted_rows) {
    absl::StrAppend(&out, "|");
    for (size_t c = 0; c < total_cols; ++c) {
      absl::StrAppendFormat(&out, " %-*s |", static_cast<int>(widths[c]),
                            row[c]);
    }
    absl::StrAppend(&out, "\n");
  }
  return out;
}

}  // namespace

absl::StatusOr<std::string> TabulateFunction(ImportData* import_data,
                                             const TypecheckedModule& tm,
                                             std::string_view fn_name) {
  XLS_RET_CHECK(import_data != nullptr);
  XLS_RET_CHECK(tm.module != nullptr);
  XLS_RET_CHECK(tm.type_info != nullptr);

  XLS_ASSIGN_OR_RETURN(Function * fn, GetTabulatedFunction(tm.module, fn_name));

  std::optional<Type*> fn_type_opt = tm.type_info->GetItem(fn);
  if (!fn_type_opt.has_value() || !(*fn_type_opt)->IsFunction()) {
    return absl::InternalError(absl::StrFormat(
        "Could not resolve FunctionType for `%s`", fn->identifier()));
  }
  const FunctionType& fn_type = (*fn_type_opt)->AsFunction();

  XLS_ASSIGN_OR_RETURN(std::vector<ColumnSpec> columns,
                       BuildColumns(*fn, fn_type));
  XLS_ASSIGN_OR_RETURN(
      std::vector<std::vector<InterpValue>> rows,
      EnumerateFunctionDomains(import_data, tm.type_info, *fn, fn_type,
                               fn->tabulate_domains(), kMaxDomainPoints));
  XLS_RETURN_IF_ERROR(EvaluateAndAppendOutputs(import_data, tm.type_info, *fn,
                                               fn_type.return_type(),
                                               columns.size(), rows));
  return RenderMarkdownTable(columns, rows);
}

}  // namespace xls::dslx
