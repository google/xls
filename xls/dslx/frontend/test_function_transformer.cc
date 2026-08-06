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

#include "xls/dslx/frontend/test_function_transformer.h"

#include <memory>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "xls/common/attribute_data.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_cloner.h"
#include "xls/dslx/frontend/ast_node_visitor_with_default.h"
#include "xls/dslx/frontend/module.h"

namespace xls {
namespace dslx {

namespace {
class SpawnFinder : public AstNodeVisitorWithDefault {
 public:
  // Legacy spawn statement, not the new spawn attribute.
  absl::Status HandleSpawn(const Spawn* node) override {
    has_spawn_ = true;
    return DefaultHandler(node);
  }

  absl::Status HandleInvocation(const Invocation* node) override {
    // Trait-based spawn
    if (const Attr* attr = dynamic_cast<const Attr*>(node->callee());
        attr != nullptr) {
      has_spawn_ = attr->attr() == "spawn";
    }
    return absl::OkStatus();
  }

  absl::Status HandleFunction(const Function* node) override {
    has_spawn_ = false;
    if (GetAttribute(node, AttributeKind::kTest).has_value()) {
      // Skip functions that aren't test functions.
      return absl::OkStatus();
    }
    // Recurse into children to see if there are any spawns.
    return DefaultHandler(node);
  }
  absl::Status DefaultHandler(const AstNode* node) override {
    for (const auto& child : node->GetChildren(false)) {
      XLS_RETURN_IF_ERROR(child->Accept(this));
    }
    return absl::OkStatus();
  }

  bool has_spawn() const { return has_spawn_; }

 private:
  bool has_spawn_ = false;
};

absl::StatusOr<TestProc*> TransmuteToTestProc(const Function* fn,
                                              Module* module) {
  // TODO(davidplass): Implement this.
  return nullptr;
}

}  // namespace

absl::StatusOr<std::unique_ptr<Module>> TransformTestFunctions(
    const Module& module) {
  // Cannot reserve space in the vector because we don't know how many functions
  // will be transformed.
  std::vector<const AstNode*> test_functions_with_spawn;

  // 1. Find all test functions with spawns.
  for (auto* func : module.GetFunctions()) {
    SpawnFinder spawn_finder;
    XLS_RETURN_IF_ERROR(func->Accept(&spawn_finder));
    if (spawn_finder.has_spawn()) {
      test_functions_with_spawn.push_back(func);
    }
  }

  // 2. Clone the module, removing test functions with spawns.
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<Module> cloned_module,
      CloneModuleRemovingMembers(module, test_functions_with_spawn));

  // 3. For each test function with a spawn, transform it into a TestProc and
  // add to the cloned module.
  for (auto* node : test_functions_with_spawn) {
    const Function* func = dynamic_cast<const Function*>(node);
    XLS_ASSIGN_OR_RETURN(TestProc * test_proc,
                         TransmuteToTestProc(func, cloned_module.get()));
    // TODO(davidplass): Add collision error handler using the node's location
    XLS_RETURN_IF_ERROR(
        cloned_module->AddTop(test_proc, /*make_collision_error=*/nullptr));
  }
  return cloned_module;
}
}  // namespace dslx
}  // namespace xls
