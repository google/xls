// Copyright 2023 The XLS Authors
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

#include "xls/dslx/lsp/lsp_type_utils.h"

#include <filesystem>  // NOLINT
#include <string_view>

#include "verible/common/lsp/lsp-file-utils.h"
#include "verible/common/lsp/lsp-protocol.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/lsp/lsp_uri.h"

namespace xls::dslx {

verible::lsp::Position ConvertPosToLspPosition(const Pos& pos) {
  return verible::lsp::Position{.line = static_cast<int>(pos.lineno()),
                                .character = static_cast<int>(pos.colno())};
}

verible::lsp::Range ConvertSpanToLspRange(const Span& span) {
  return verible::lsp::Range{.start = ConvertPosToLspPosition(span.start()),
                             .end = ConvertPosToLspPosition(span.limit())};
}

verible::lsp::Location ConvertSpanToLspLocation(const Span& span,
                                                const FileTable& file_table) {
  verible::lsp::Location location{.range = ConvertSpanToLspRange(span)};
  location.uri = verible::lsp::PathToLSPUri(span.GetFilename(file_table));
  return location;
}

Pos ConvertLspPositionToPos(const LspUri& file_uri,
                            const verible::lsp::Position& position,
                            FileTable& file_table) {
  std::filesystem::path path = file_uri.GetFilesystemPath();
  Fileno fileno = file_table.GetOrCreate(path.c_str());
  return Pos(fileno, position.line, position.character);
}

Span ConvertLspRangeToSpan(const LspUri& file_uri,
                           const verible::lsp::Range& range,
                           FileTable& file_table) {
  return Span(ConvertLspPositionToPos(file_uri, range.start, file_table),
              ConvertLspPositionToPos(file_uri, range.end, file_table));
}

}  // namespace xls::dslx
