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

#ifndef XLS_PASSES_TERNARY_QUERY_ENGINE_TEST_COMMON_H_
#define XLS_PASSES_TERNARY_QUERY_ENGINE_TEST_COMMON_H_

#include "gtest/gtest.h"
#include "cppitertools/zip.hpp"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_test_library.h"
#include "xls/fuzzer/ir_fuzzer/query_engine_helpers.h"
#include "xls/ir/node.h"
#include "xls/ir/ternary.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/query_engine.h"

namespace xls {

template <typename TernaryEngineT>
void CheckTernaryEngineConsistency(const FuzzPackageWithArgs& fuzz) {
  CheckQueryEngineConsistency<TernaryEngineT>(
      fuzz, [](const QueryEngine& qe, Node* n, const Value& v) -> bool {
        auto tern = qe.GetTernary(n);
        if (!tern) {
          // Unconstrained.
          return true;
        }
        auto ltt = ValueToLeafTypeTree(v, n->GetType());
        if (!ltt.ok()) {
          ADD_FAILURE() << "Unable to parse " << v << " into ltt";
          return false;
        }
        for (const auto& [tern_bits, val] :
             iter::zip(tern->elements(), ltt->elements())) {
          if (val.IsBits()) {
            EXPECT_TRUE(ternary_ops::IsCompatible(tern_bits, val.bits()))
                << "Incompatible ternary segment: " << ToString(tern_bits)
                << " incompatible with " << val << " with node " << n;
          } else {
            EXPECT_TRUE(tern_bits.empty());
          }
        }
        return true;
      });
}

}  // namespace xls

#endif  // XLS_PASSES_TERNARY_QUERY_ENGINE_TEST_COMMON_H_
