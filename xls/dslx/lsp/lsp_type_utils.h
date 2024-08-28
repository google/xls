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

// Helpers for converting between DSLX types and Language Server Protocol types.

#ifndef XLS_DSLX_LSP_LSP_TYPE_UTILS_H_
#define XLS_DSLX_LSP_LSP_TYPE_UTILS_H_

#include <string_view>

#include "external/verible/common/lsp/lsp-protocol.h"
#include "xls/dslx/frontend/pos.h"

namespace xls::dslx {

verible::lsp::Position ConvertPosToLspPosition(const Pos& pos);
verible::lsp::Range ConvertSpanToLspRange(const Span& span);

verible::lsp::Location ConvertSpanToLspLocation(const Span& span);

// Note: DSLX positions have filenames included in them, whereas LSP positions
// do not -- we need the LSP adapter to handle filename resolution from URIs to
// handle this in a uniform way, so we assume this will only be used in
// single-file contexts for the moment and use an empty string for the filename.
Pos ConvertLspPositionToPos(std::string_view file_uri,
                            const verible::lsp::Position& position);

Span ConvertLspRangeToSpan(std::string_view file_uri,
                           const verible::lsp::Range& range);

}  // namespace xls::dslx

#endif  // XLS_DSLX_LSP_LSP_TYPE_UTILS_H_
