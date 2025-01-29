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

#ifndef XLS_DEV_TOOLS_EXTRACT_STATE_ELEMENT_H_
#define XLS_DEV_TOOLS_EXTRACT_STATE_ELEMENT_H_

#include <memory>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/ir/package.h"
#include "xls/ir/state_element.h"
namespace xls {

// Create a new package containing a single proc which is the components of
// 'proc' which are responsible for updating the state elements in
// 'state_elements'.
//
// If 'send_state_values' is true then the value of each state element
// will be sent to a streaming channel on each proc activation.
absl::StatusOr<std::unique_ptr<Package>> ExtractStateElementsInNewPackage(
    Proc* proc, absl::Span<StateElement* const> state_elements,
    bool send_state_values);

}  // namespace xls

#endif  // XLS_DEV_TOOLS_EXTRACT_STATE_ELEMENT_H_
