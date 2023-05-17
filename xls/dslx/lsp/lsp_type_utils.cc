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

namespace xls::dslx {

verible::lsp::Range ConvertSpanToLspRange(const Span& span) {
  verible::lsp::Range result = {
      .start = {.line = static_cast<int>(span.start().lineno()),
                .character = static_cast<int>(span.start().colno())},
      .end = {.line = static_cast<int>(span.limit().lineno()),
              .character = static_cast<int>(span.limit().colno())}};
  return result;
}

verible::lsp::Location ConvertSpanToLspLocation(const Span& span) {
  return verible::lsp::Location{.range = ConvertSpanToLspRange(span)};
}

Pos ConvertLspPositionToPos(std::string_view file_uri,
                            const verible::lsp::Position& position) {
  return Pos("", position.line, position.character);
}

}  // namespace xls::dslx
