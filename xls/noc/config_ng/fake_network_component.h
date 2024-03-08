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

#ifndef XLS_NOC_CONFIG_NG_FAKE_NETWORK_COMPONENT_H_
#define XLS_NOC_CONFIG_NG_FAKE_NETWORK_COMPONENT_H_

#include "xls/noc/config_ng/network_component.h"

namespace xls::noc {

class NetworkView;

class FakeNetworkComponent : public NetworkComponent {
 public:
  explicit FakeNetworkComponent(NetworkView* network_view)
      : NetworkComponent(network_view) {}

  absl::Status Visit(NetworkComponentVisitor& v) override {
    return absl::Status();
  };
};

}  // namespace xls::noc

#endif  // XLS_NOC_CONFIG_NG_FAKE_NETWORK_COMPONENT_H_
