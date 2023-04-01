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

#include "verible/common/lsp/lsp-protocol.h"
#include "xls/dslx/frontend/pos.h"

namespace xls::dslx {

verible::lsp::Range ConvertSpanToRange(const Span& span);

}  // namespace xls::dslx

#endif  // XLS_DSLX_LSP_LSP_TYPE_UTILS_H_
