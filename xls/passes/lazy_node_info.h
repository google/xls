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

#ifndef XLS_PASSES_LAZY_NODE_INFO_H_
#define XLS_PASSES_LAZY_NODE_INFO_H_

#include <cstdint>
#include <optional>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/node.h"
#include "xls/ir/type.h"
#include "xls/passes/lazy_node_data.h"

namespace xls {

// Specialization of LazyNodeData which uses LeafTypeTrees to store
// information about each leaf element of a node's type.
template <typename Info>
class LazyNodeInfo : public LazyNodeData<LeafTypeTree<Info>> {
 public:
  // Users must implement ComputeInfo (inherited) and MergeWithGiven.
  // MergeWithGiven(Info&, const Info&) is called to merge 'given' leaf
  // information into 'info'.
  virtual absl::Status MergeWithGiven(Info& info, const Info& given) const = 0;

  // Shadows LazyNodeData::GetInfo to return the LeafTypeTree in the expected
  // optional<Shared...> format.
  std::optional<SharedLeafTypeTree<Info>> GetInfo(Node* node) const {
    const LeafTypeTree<Info>* info =
        LazyNodeData<LeafTypeTree<Info>>::GetInfo(node);
    if (info == nullptr) {
      return std::nullopt;
    }
    return info->AsView().AsShared();
  }

 private:
  // Implements LazyNodeData::MergeWithGiven by applying MergeLeafs to each
  // element of the LeafTypeTree.
  absl::Status MergeWithGiven(
      LeafTypeTree<Info>& info,
      const LeafTypeTree<Info>& given) const final override {
    return leaf_type_tree::UpdateFrom<Info, Info>(
        info.AsMutableView(), given.AsView(),
        [this](Type* type, Info& i, const Info& g,
               absl::Span<const int64_t> idx) { return MergeWithGiven(i, g); });
  }
};

}  // namespace xls

#endif  // XLS_PASSES_LAZY_NODE_INFO_H_
