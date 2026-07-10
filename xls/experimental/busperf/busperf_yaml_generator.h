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

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/codegen/module_signature.pb.h"

namespace xls::busperf {

// Generates a busperf (https://github.com/antmicro/busperf) YAML bus
// description from an XLS ModuleSignatureProto. Covers both external interface,
// and internal channels between a proc and any children it spawns.
//
// Args:
//   signature: the top-level block's ModuleSignatureProto (codegen_main
//     --output_signature_path=...).
//   scope: VCD scope path components leading to the DUT instance, e.g.
//     {"tb_passthrough", "dut"}.
//   child_signatures: standalone signatures for spawned child procs
//     (codegen'd without --module_name, so each module_name is the
//     mangled block name), keyed by module_name. Matched against the
//     parent's instantiations so their channels get included too, scoped
//     under `scope` plus their instance_name.
absl::StatusOr<std::string> GenerateBusperfYaml(
    const verilog::ModuleSignatureProto& signature,
    absl::Span<const std::string> scope,
    const absl::flat_hash_map<std::string, verilog::ModuleSignatureProto>&
        child_signatures);

}  // namespace xls::busperf

#endif  // XLS_EXPERIMENTAL_BUSPERF_BUSPERF_YAML_GENERATOR_H_
