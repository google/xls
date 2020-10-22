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

#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/iterator_range.h"
#include "xls/common/status/ret_check.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/function_base.h"
#include "xls/ir/name_uniquer.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/ir/unwrapping_iterator.h"
#include "xls/ir/verifier.h"

namespace xls {

class Function : public FunctionBase {
 private:
  using NodeList = std::list<std::unique_ptr<Node>>;

 public:
  Function(absl::string_view name, Package* package)
      : FunctionBase(name, package) {}
  virtual ~Function() = default;

  // DumpIr emits the IR in a parsable, hierarchical text format.
  // Parameter:
  //   'recursive' if true, will dump counted-for body functions as well.
  //   This is only useful when dumping individual functions, and not packages.
  std::string DumpIr(bool recursive = false) const override;

  // Creates a clone of the function with the new name 'new_name'. Function is
  // owned by targt_package.  call_remapping specifies any function
  // substitutions to be used in the cloned function, e.g. If call_remapping
  // holds {funcA, funcB}, any references to funcA in the function will be
  // references to funcB in the cloned function.
  absl::StatusOr<Function*> Clone(
      absl::string_view new_name, Package* target_package = nullptr,
      const absl::flat_hash_map<const Function*, Function*>& call_remapping =
          {}) const;

  // Returns true if analysis indicates that this function always produces the
  // same value as 'other' when run with the same arguments. The analysis is
  // conservative and false may be returned for some "equivalent" functions.
  bool IsDefinitelyEqualTo(const Function* other) const;
};

}  // namespace xls

#endif  // XLS_IR_FUNCTION_H_
