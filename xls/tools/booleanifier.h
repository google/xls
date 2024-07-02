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

#ifndef XLS_TOOLS_BOOLEANIFIER_H_
#define XLS_TOOLS_BOOLEANIFIER_H_

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/node.h"

namespace xls {

class BitEvaluator;

// Booleanifier converts an IR Function written with high-level constructs into
// a Function with the same behavior but implemented in terms of only And, Or,
// and Not operations.
class Booleanifier {
 public:
  // "Typedefs" to use the same terminology as AbstractEvaluator.
  using Element = Node*;
  using Vector = std::vector<Node*>;

  // Return a booleanified function equivalent to the given function. The
  // function is placed in the same Package as f. boolean_function_name, if
  // given, is the name of the generated boolean function.
  static absl::StatusOr<Function*> Booleanify(
      Function* f, std::string_view boolean_function_name = "");

 private:
  Booleanifier(Function* f, std::string_view boolean_function_name);

  // Driver for doing the actual conversion.
  absl::StatusOr<Function*> Run();

  // The inverse of UnpackParam - overlays structure on top of a flat bit array.
  absl::StatusOr<BValue> PackReturnValue(
      LeafTypeTreeView<std::vector<Node*>> result);

  Function* input_fn_;
  FunctionBuilder builder_;
  std::unique_ptr<BitEvaluator> evaluator_;
  absl::flat_hash_map<std::string, BValue> params_;
};

}  // namespace xls
#endif  // XLS_TOOLS_BOOLEANIFIER_H_
