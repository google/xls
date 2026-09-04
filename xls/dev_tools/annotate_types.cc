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

#include "xls/dev_tools/annotate_types.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/base/casts.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/default_dslx_stdlib_path.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/type_to_type_annotation.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/ir_convert/convert_options.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/virtualizable_file_system.h"

namespace xls::dslx {

absl::StatusOr<std::string> AnnotateTypes(std::string_view dslx_code,
                                          std::string_view module_name,
                                          std::string_view path,
                                          const AnnotateTypesOptions& options) {
  std::filesystem::path stdlib_path =
      options.dslx_stdlib_path.empty()
          ? std::filesystem::path(xls::kDefaultDslxStdlibPath)
          : std::filesystem::path(options.dslx_stdlib_path);

  ImportData import_data =
      CreateImportData(stdlib_path, options.dslx_paths, options.warnings,
                       std::make_unique<RealFilesystem>());

  ConvertOptions convert_options = {
      .warnings_as_errors = options.warnings_as_errors,
      .warnings = options.warnings,
      .type_inference_v2 = options.type_inference_v2,
  };

  XLS_ASSIGN_OR_RETURN(
      TypecheckedModule tm,
      ParseAndTypecheck(dslx_code, path, module_name, &import_data,
                        /*comments=*/nullptr, convert_options));

  std::vector<int64_t> line_starts;
  line_starts.push_back(0);
  for (int64_t i = 0; i < static_cast<int64_t>(dslx_code.size()); ++i) {
    if (dslx_code[i] == '\n') {
      line_starts.push_back(i + 1);
    }
  }

  auto pos_to_byte_offset = [&](const Pos& pos) -> absl::StatusOr<int64_t> {
    if (pos.lineno() < 0 ||
        pos.lineno() >= static_cast<int64_t>(line_starts.size())) {
      return absl::OutOfRangeError(
          absl::StrFormat("Line number %d out of range (total lines %d)",
                          pos.lineno(), line_starts.size()));
    }
    int64_t line_start = line_starts[pos.lineno()];
    int64_t offset = line_start + pos.colno();
    if (offset < 0 || offset > static_cast<int64_t>(dslx_code.size())) {
      return absl::OutOfRangeError(absl::StrFormat(
          "Offset %d out of range (total size %d)", offset, dslx_code.size()));
    }
    return offset;
  };

  struct Edit {
    int64_t offset;
    std::string text;
  };
  std::vector<Edit> edits;

  const std::vector<const AstNode*> contained =
      tm.module->FindContained(tm.module->span());
  for (const AstNode* node : contained) {
    if (node->kind() != AstNodeKind::kLet) {
      continue;
    }
    const auto* let = absl::down_cast<const Let*>(node);
    if (let->type_annotation() != nullptr) {
      continue;
    }
    const PatternTree& pattern = let->pattern();
    AstNode* pattern_node = ToAstNode(pattern);
    std::optional<Type*> maybe_type = tm.type_info->GetItem(pattern_node);
    if (!maybe_type.has_value()) {
      maybe_type = tm.type_info->GetItem(let->rhs());
    }
    if (!maybe_type.has_value()) {
      VLOG(3) << "No type information available for let: `" << let->ToString()
              << "` at " << let->span().ToString(import_data.file_table());
      continue;
    }
    const Type& type = *maybe_type.value();
    if (type.IsMeta()) {
      continue;
    }

    std::string type_str;
    absl::StatusOr<TypeAnnotation*> ta = CreateTypeAnnotation(
        *tm.module, type, GetPatternSpan(pattern), tm.module, tm.type_info);
    if (ta.ok()) {
      type_str = (*ta)->ToString();
    } else {
      type_str = type.ToInlayHintString();
    }
    if (type_str.empty()) {
      continue;
    }

    const Pos& limit_pos = GetPatternSpan(pattern).limit();
    absl::StatusOr<int64_t> maybe_offset = pos_to_byte_offset(limit_pos);
    if (!maybe_offset.ok()) {
      VLOG(3) << "Could not compute byte offset for limit pos: "
              << limit_pos.ToString(import_data.file_table()) << ": "
              << maybe_offset.status();
      continue;
    }
    edits.push_back(Edit{
        .offset = *maybe_offset,
        .text = absl::StrCat(": ", type_str),
    });
  }

  std::sort(edits.begin(), edits.end(),
            [](const Edit& a, const Edit& b) { return a.offset > b.offset; });

  std::string result(dslx_code);
  int64_t last_offset = -1;
  for (const auto& edit : edits) {
    if (edit.offset == last_offset) {
      continue;
    }
    result.insert(edit.offset, edit.text);
    last_offset = edit.offset;
  }

  return result;
}

}  // namespace xls::dslx
