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
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/node.h"
#include "xls/ir/type.h"

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

  // The resulting function will be named "<f->name()>_boolean", and be placed
  // in the same Package as f.
  static absl::StatusOr<Function*> Booleanify(Function* f);

 private:
  Booleanifier(Function* f);

  // Driver for doing the actual conversion.
  absl::StatusOr<Function*> Run();

  // "Callback" from the AbstractEvaluator for ops that don't make sense there.
  // So far, these are ops that involve data layouts: param handling, tuple
  // construction/access, etc.
  Vector HandleSpecialOps(Node* node);

  // Converts a structured input param into a flat bit array.
  Vector UnpackParam(Type* type, BValue bv_node);

  // The inverse of UnpackParam - overlays structure on top of a flat bit array.
  // We take a span here, instead of a Vector, so we can easily create subspans.
  BValue PackReturnValue(absl::Span<const Element> bits, const Type* type);

  Function* input_fn_;
  FunctionBuilder builder_;
  std::unique_ptr<BitEvaluator> evaluator_;
  absl::flat_hash_map<std::string, BValue> params_;
  absl::flat_hash_map<Node*, Vector> node_map_;
};

}  // namespace xls
#endif  // XLS_TOOLS_BOOLEANIFIER_H_
