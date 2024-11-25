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

#ifndef XLS_DSLX_LSP_DOCUMENT_SYMBOLS_H_
#define XLS_DSLX_LSP_DOCUMENT_SYMBOLS_H_

#include <vector>

#include "verible/common/lsp/lsp-protocol.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"

namespace xls::dslx {

// Traverses the module "m" and gathers up document symbols contained therein.
//
// Note that the DocumentSymbol type is hierarchical, if a top level symbol has
// children they will be contained inside, so the vector returned has direct
// elements that are the "top level" of the symbol tree.
std::vector<verible::lsp::DocumentSymbol> ToDocumentSymbols(const Module& m);

}  // namespace xls::dslx

#endif  // XLS_DSLX_LSP_DOCUMENT_SYMBOLS_H_
