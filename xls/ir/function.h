// Copyright 2020 The XLS Authors
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

#ifndef XLS_IR_FUNCTION_H_
#define XLS_IR_FUNCTION_H_

#include <functional>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/ret_check.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"

namespace xls {

class Function : public FunctionBase {
 private:
  using NodeList = std::list<std::unique_ptr<Node>>;

 public:
  Function(std::string_view name, Package* package)
      : FunctionBase(name, package) {}
  ~Function() override = default;

  // Returns the node that serves as the return value of this function.
  Node* return_value() const { return return_value_; }

  // Sets the node that serves as the return value of this function.
  absl::Status set_return_value(Node* n) {
    XLS_RET_CHECK_EQ(n->function_base(), this) << absl::StreamFormat(
        "Return value node %s is not in this function %s (is in function %s)",
        n->GetName(), name(), n->function_base()->name());
    return_value_ = n;
    return absl::OkStatus();
  }

  FunctionType* GetType();

  // DumpIr emits the IR in a parsable, hierarchical text format.
  // Parameter:
  //   'recursive' if true, will dump counted-for body functions as well.
  //   This is only useful when dumping individual functions, and not packages.
  std::string DumpIr() const override;

  // DumpIr emits the IR in a hierarchical text format with the returned
  // annotations after each node definition.
  std::string DumpIrWithAnnotations(
      const std::function<std::optional<std::string>(Node*)>& annotate) const;

  // Creates a clone of the function with the new name 'new_name'. Function is
  // owned by target_package.  call_remapping specifies any function
  // substitutions to be used in the cloned function, e.g. If call_remapping
  // holds {funcA, funcB}, any references to funcA in the function will be
  // references to funcB in the cloned function.
  absl::StatusOr<Function*> Clone(
      std::string_view new_name, Package* target_package = nullptr,
      const absl::flat_hash_map<const Function*, Function*>& call_remapping =
          {}) const;

  // Returns true if analysis indicates that this function always produces the
  // same value as 'other' when run with the same arguments. The analysis is
  // conservative and false may be returned for some "equivalent" functions.
  bool IsDefinitelyEqualTo(const Function* other) const;

  bool HasImplicitUse(Node* node) const final { return node == return_value(); }

 private:
  Node* return_value_ = nullptr;
};

}  // namespace xls

#endif  // XLS_IR_FUNCTION_H_
