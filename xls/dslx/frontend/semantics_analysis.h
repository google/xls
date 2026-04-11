// Copyright 2025 The XLS Authors
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

#ifndef XLS_DSLX_FRONTEND_SEMANTICS_ANALYSIS_H_
#define XLS_DSLX_FRONTEND_SEMANTICS_ANALYSIS_H_

#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node_visitor_with_default.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/warning_collector.h"

namespace xls::dslx {

// TokenOrderCheckData is used for storing data necessary to perform
// dependency order check in semantics analysis post type check pass.
struct TokenOrderCheckData {
  std::vector<const AstNode*> data_nodes = {};
  std::vector<const AstNode*> token_nodes = {};
  const AstNode* operation = nullptr;
};

// A visitor which collects dependency levels for each token and data node.
// Each `recv` channel operation defines data dependency level, based on tokens
// it used. For every `send` node it creates an entry for a post type check pass
// to see if sent data can be assembled at given dependency level defined by
// `send` operation tokens.
class DependencyOrderCollector : public AstNodeVisitorWithDefault {
 public:
  DependencyOrderCollector() = default;

  absl::Status HandleLet(const Let* let) override;
  absl::Status HandleInvocation(const Invocation* invocation) override;

  const std::vector<TokenOrderCheckData>& io_ordering_check_items() {
    return io_ordering_check_items_;
  }

  unsigned int GetNodeDependencyLevel(const AstNode* token) {
    return node_dependencies_[token];
  }

 private:
  void AddNodeDependency(const AstNode* node,
                         const absl::flat_hash_set<const AstNode*>& dependencies);
  void AddNodeAlias(const AstNode* alias,
                    const absl::flat_hash_set<const AstNode*>& aliased);

  absl::Status HandleSend(const Invocation* invocation);

  std::vector<TokenOrderCheckData> io_ordering_check_items_;
  absl::flat_hash_map<const AstNode*, unsigned int> node_dependencies_;
  absl::flat_hash_set<const AstNode*> io_nodes_;
};

// This class traverses the AST to generate warnings for improper XLS code that
// is misleading or has no effect. Several kinds of warnings are context free
// and does not require type-checking so they can be emitted at
// RunPreTypeCheckPass, while the others require type-checking and are emitted
// at a later stage. This class also collects and holds necessary information to
// assist the generation of those warnings.
class SemanticsAnalysis {
 public:
  SemanticsAnalysis(bool suppress_warnings = false);

  absl::Status RunPreTypeCheckPass(Module& module,
                                   WarningCollector& warning_collector,
                                   ImportData& import_data);

  absl::Status RunPostTypeCheckPass(WarningCollector& warning_collector);

  void SetNameDefType(const NameDef* def, const Type* type);

  absl::Status TrackIODependency(const AstNode* node);

 private:
  // Used by kUnusedDefinition. We cannot completely determine whether a
  // definition is truly unused at RunPreTypeCheckPass, because (1) tokens are
  // implicitly joined, and (2) if we implement const_if, unused definitions in
  // a never-taken branch should not be warned. They have to be reported after
  // type checking.
  std::vector<std::pair<const Function*, std::vector<const NameDef*>>>
      maybe_unreferenced_defs;
  absl::flat_hash_map<const NameDef*, std::unique_ptr<Type>> def_to_type_;
  bool suppress_warnings_;

  // Used by kIOOrderingMismatch. It checks whether the data arg used in `send`
  // channel operation is possible to assemble from all data available with
  // constraints of given tokens.
  absl::Status RunIOOrderingAnalysis(WarningCollector& warning_collector);

  DependencyOrderCollector dependency_order_collector_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_FRONTEND_SEMANTICS_ANALYSIS_H_
