// Copyright 2021 The XLS Authors
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

#include "xls/noc/config_ng/network_view.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/memory/memory.h"
#include "xls/common/iterator_range.h"
#include "xls/ir/unwrapping_iterator.h"
#include "xls/noc/config_ng/network_component.h"
#include "xls/noc/config_ng/network_connection.h"

namespace xls::noc {

xabsl::iterator_range<UnwrappingIterator<
    std::vector<std::unique_ptr<NetworkComponent>>::iterator>>
NetworkView::components() {
  return xabsl::make_range(MakeUnwrappingIterator(components_.begin()),
                           MakeUnwrappingIterator(components_.end()));
}
xabsl::iterator_range<UnwrappingIterator<
    std::vector<std::unique_ptr<NetworkComponent>>::const_iterator>>
NetworkView::components() const {
  return xabsl::make_range(MakeUnwrappingIterator(components_.begin()),
                           MakeUnwrappingIterator(components_.end()));
}

int64_t NetworkView::GetComponentCount() const { return components_.size(); }

NetworkConnection& NetworkView::AddConnection() {
  // Using `new` to access a non-public constructor.
  connections_.emplace_back(absl::WrapUnique(new NetworkConnection(this)));
  return *connections_.back();
}

xabsl::iterator_range<UnwrappingIterator<
    std::vector<std::unique_ptr<NetworkConnection>>::iterator>>
NetworkView::connections() {
  return xabsl::make_range(MakeUnwrappingIterator(connections_.begin()),
                           MakeUnwrappingIterator(connections_.end()));
}

xabsl::iterator_range<UnwrappingIterator<
    std::vector<std::unique_ptr<NetworkConnection>>::const_iterator>>
NetworkView::connections() const {
  return xabsl::make_range(MakeUnwrappingIterator(connections_.begin()),
                           MakeUnwrappingIterator(connections_.end()));
}

int64_t NetworkView::GetConnectionCount() const { return connections_.size(); }

}  // namespace xls::noc
