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

#ifndef XLS_NOC_CONFIG_NG_NETWORK_VIEW_H_
#define XLS_NOC_CONFIG_NG_NETWORK_VIEW_H_

#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#include "xls/common/iterator_range.h"
#include "xls/ir/unwrapping_iterator.h"
#include "xls/noc/config_ng/network_component.h"
#include "xls/noc/config_ng/network_connection.h"

namespace xls::noc {

// A network view contains components and connections. The network view owns the
// components and connections that it contains.
// TODO (vmirian) 02-05-21 enable view type
class NetworkView {
 public:
  NetworkView() = default;

  NetworkView(const NetworkView&) = delete;
  NetworkView& operator=(const NetworkView&) = delete;

  // Add a component to the view. The component type must be a network component
  // base class or derived class. Returns a reference to the newly added
  // component. The component is owned by the view, the lifetime of the
  // component is equivalent to the lifetime of the view.
  template <typename Type>
  Type& AddComponent() {
    static_assert(std::is_base_of<NetworkComponent, Type>::value,
                  "Type is not a Network Component subclass");
    components_.emplace_back(std::make_unique<Type>(this));
    return static_cast<Type&>(*components_.back());
  }

  // Counts the number of a components of the type specified by the template.
  // The component type must be a network component base class or derived class.
  template <typename Type>
  int64_t GetCount() const {
    static_assert(std::is_base_of<NetworkComponent, Type>::value,
                  "Type is not a Network Component subclass");
    int64_t count = 0;
    for (const NetworkComponent* component : components()) {
      if (typeid(*component) == typeid(Type)) {
        count++;
      }
    }
    return count;
  }

  // Returns an iterator range for the components. The objects are guaranteed to
  // be non-null. Note that, when using the result of this function, if the view
  // is modified (e.g. a component is added), the returned result may become
  // invalid.
  xabsl::iterator_range<UnwrappingIterator<
      std::vector<std::unique_ptr<NetworkComponent>>::iterator>>
  components();
  xabsl::iterator_range<UnwrappingIterator<
      std::vector<std::unique_ptr<NetworkComponent>>::const_iterator>>
  components() const;

  // Returns the number of components.
  int64_t GetComponentCount() const;

  // Add a connection to the view. The connection is owned by the view, the
  // lifetime of the connection is equivalent to the lifetime of the view.
  NetworkConnection& AddConnection();

  // Returns an iterator range for the connections. The objects are guaranteed
  // to be non-null. Note that, when using the result of this function, if the
  // view is modified (e.g. a connection is added), the returned result may
  // become invalid.
  xabsl::iterator_range<UnwrappingIterator<
      std::vector<std::unique_ptr<NetworkConnection>>::iterator>>
  connections();
  xabsl::iterator_range<UnwrappingIterator<
      std::vector<std::unique_ptr<NetworkConnection>>::const_iterator>>
  connections() const;

  // Returns the number of connections.
  int64_t GetConnectionCount() const;

  virtual ~NetworkView() = default;

 private:
  std::vector<std::unique_ptr<NetworkComponent>> components_;
  std::vector<std::unique_ptr<NetworkConnection>> connections_;
};

}  // namespace xls::noc

#endif  // XLS_NOC_CONFIG_NG_NETWORK_VIEW_H_
