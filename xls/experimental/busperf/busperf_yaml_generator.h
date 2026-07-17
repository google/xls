// Copyright 2026 The XLS Authors
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

#ifndef XLS_EXPERIMENTAL_BUSPERF_BUSPERF_YAML_GENERATOR_H_
#define XLS_EXPERIMENTAL_BUSPERF_BUSPERF_YAML_GENERATOR_H_

#include <string>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/codegen/module_signature.pb.h"

namespace xls::busperf {

// Generates a busperf YAML bus description from an XLS ModuleSignatureProto.
// Covers both external interface, and internal channels between a proc
// and any children it spawns.
//
// Args:
//   signature: the top-level block's ModuleSignatureProto (codegen_main
//     --output_signature_path=...). Child block instantiations carry their
//     own signature inline (instantiations().block_instantiation()
//     .block_signature()), so this recurses through the whole design.
//   scope: VCD scope path components leading to the DUT instance, e.g.
//     {"tb_passthrough", "dut"}.
absl::StatusOr<std::string> GenerateBusperfYaml(
    const verilog::ModuleSignatureProto& signature,
    absl::Span<const std::string> scope);

}  // namespace xls::busperf

#endif  // XLS_EXPERIMENTAL_BUSPERF_BUSPERF_YAML_GENERATOR_H_
