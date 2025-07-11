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

#ifndef XLS_PASSES_TOOLS_GENERATE_DOCUMENTATION_PROTO_H_
#define XLS_PASSES_TOOLS_GENERATE_DOCUMENTATION_PROTO_H_

#include <string>
#include <string_view>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/tools/pass_documentation.pb.h"
namespace xls {

absl::StatusOr<PassDocumentationProto> GenerateDocumentationProto(
    const OptimizationPassRegistryBase& registry, std::string_view headers,
    absl::Span<std::string const> copts);

}  // namespace xls

#endif  // XLS_PASSES_TOOLS_GENERATE_DOCUMENTATION_PROTO_H_
