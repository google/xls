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

#include <string_view>

#include "verible/common/lsp/lsp-protocol.h"
#include "xls/dslx/frontend/pos.h"

namespace xls::dslx {

verible::lsp::Position ConvertPosToLspPosition(const Pos& pos) {
  return verible::lsp::Position{.line = static_cast<int>(pos.lineno()),
                                .character = static_cast<int>(pos.colno())};
}

verible::lsp::Range ConvertSpanToLspRange(const Span& span) {
  return verible::lsp::Range{.start = ConvertPosToLspPosition(span.start()),
                             .end = ConvertPosToLspPosition(span.limit())};
}

verible::lsp::Location ConvertSpanToLspLocation(const Span& span) {
  return verible::lsp::Location{.range = ConvertSpanToLspRange(span)};
}

Pos ConvertLspPositionToPos(std::string_view file_uri,
                            const verible::lsp::Position& position,
                            FileTable& file_table) {
  Fileno fileno = file_table.GetOrCreate(file_uri);
  return Pos(fileno, position.line, position.character);
}

Span ConvertLspRangeToSpan(std::string_view file_uri,
                           const verible::lsp::Range& range,
                           FileTable& file_table) {
  return Span(ConvertLspPositionToPos(file_uri, range.start, file_table),
              ConvertLspPositionToPos(file_uri, range.end, file_table));
}

}  // namespace xls::dslx
