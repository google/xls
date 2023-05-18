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

#include "xls/dslx/lsp/document_symbols.h"

#include "external/verible/common/lsp/lsp-protocol-enums.h"
#include "xls/common/visitor.h"
#include "xls/dslx/lsp/lsp_type_utils.h"

namespace xls::dslx {
namespace {

std::vector<verible::lsp::DocumentSymbol> ToDocumentSymbols(const Function& f) {
  XLS_VLOG(3) << "ToDocumentSymbols; f: " << f.identifier();
  verible::lsp::DocumentSymbol ds = {
      .name = f.identifier(),
      .kind = verible::lsp::SymbolKind::kMethod,
      .range = ConvertSpanToLspRange(f.span()),
      .selectionRange = ConvertSpanToLspRange(f.name_def()->span()),
  };
  return {std::move(ds)};
}

std::vector<verible::lsp::DocumentSymbol> ToDocumentSymbols(
    const StructDef& s) {
  verible::lsp::DocumentSymbol ds = {
      .name = s.identifier(),
      .kind = verible::lsp::SymbolKind::kStruct,
      .range = ConvertSpanToLspRange(s.span()),
      .selectionRange = ConvertSpanToLspRange(s.name_def()->span()),
  };
  return {std::move(ds)};
}

std::vector<verible::lsp::DocumentSymbol> ToDocumentSymbols(const EnumDef& e) {
  verible::lsp::DocumentSymbol ds = {
      .name = e.identifier(),
      .kind = verible::lsp::SymbolKind::kEnum,
      .range = ConvertSpanToLspRange(e.span()),
      .selectionRange = ConvertSpanToLspRange(e.name_def()->span()),
  };
  return {std::move(ds)};
}

std::vector<verible::lsp::DocumentSymbol> ToDocumentSymbols(
    const ConstantDef& c) {
  verible::lsp::DocumentSymbol ds = {
      .name = c.identifier(),
      .kind = verible::lsp::SymbolKind::kConstant,
      .range = ConvertSpanToLspRange(c.span()),
      .selectionRange = ConvertSpanToLspRange(c.name_def()->span()),
  };
  return {std::move(ds)};
}

}  // namespace

std::vector<verible::lsp::DocumentSymbol> ToDocumentSymbols(const Module& m) {
  XLS_VLOG(1) << "ToDocumentSymbols; module: " << m.name() << " has "
              << m.top().size() << " top-level elements";
  std::vector<verible::lsp::DocumentSymbol> result;
  for (const ModuleMember& member : m.top()) {
    std::vector<verible::lsp::DocumentSymbol> symbols =
        absl::visit(Visitor{
                        [](Function* f) { return ToDocumentSymbols(*f); },
                        [](Proc*) {
                          return std::vector<verible::lsp::DocumentSymbol>{};
                        },  // TODO
                        [](TestFunction*) {
                          return std::vector<verible::lsp::DocumentSymbol>{};
                        },  // TODO
                        [](TestProc*) {
                          return std::vector<verible::lsp::DocumentSymbol>{};
                        },  // TODO
                        [](QuickCheck*) {
                          return std::vector<verible::lsp::DocumentSymbol>{};
                        },  // TODO
                        [](TypeAlias*) {
                          return std::vector<verible::lsp::DocumentSymbol>{};
                        },  // TODO
                        [](StructDef* s) { return ToDocumentSymbols(*s); },
                        [](ConstantDef* c) { return ToDocumentSymbols(*c); },
                        [](EnumDef* e) { return ToDocumentSymbols(*e); },
                        [](Import*) {
                          return std::vector<verible::lsp::DocumentSymbol>{};
                        }  // TODO
                    },
                    member);
    for (auto& ds : symbols) {
      result.push_back(std::move(ds));
    }
  }
  return result;
}

}  // namespace xls::dslx
