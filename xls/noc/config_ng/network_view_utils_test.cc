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

#include "xls/noc/config_ng/network_view_utils.h"

#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/noc/config_ng/network_view.h"

namespace xls::noc {
namespace {

// Validates an empty network view (no components and no connections).
TEST(NetworkViewUtilsTest, ValidateNetworkView) {
  NetworkView view;
  XLS_EXPECT_OK(ValidateNetworkView(view));
}

}  // namespace
}  // namespace xls::noc
