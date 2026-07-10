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

#include "xls/experimental/busperf/busperf_yaml_generator.h"

#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/codegen/module_signature.pb.h"

namespace xls::busperf {
namespace {

constexpr std::string_view kDefaultClockName = "clk";
constexpr std::string_view kDefaultResetName = "rst";
constexpr std::string_view kClockAlias = "main_clk";

// One resolved busperf `interfaces:` entry.
struct BusInterface {
  std::string name;
  std::vector<std::string> scope;
  std::string ready_port;
  std::string valid_port;
};

// Gets signature's own ready-valid channels; does not descend into
// instantiations.
std::vector<const verilog::ChannelInterfaceProto*> ReadyValidChannelsOf(
    const verilog::ModuleSignatureProto& signature) {
  std::vector<const verilog::ChannelInterfaceProto*> result;
  for (const verilog::ChannelInterfaceProto& channel_interface :
       signature.channel_interfaces()) {
    if (!channel_interface.has_streaming()) {
      continue;  // single-value: no handshake.
    }
    if (channel_interface.streaming().flow_control() !=
        verilog::CHANNEL_FLOW_CONTROL_READY_VALID) {
      continue;
    }
    result.push_back(&channel_interface);
  }
  return result;
}

// Quotes and escapes `s` for use as a YAML scalar.
std::string YamlString(std::string_view s) {
  std::string result;
  result.reserve(s.size() + 2);
  result.push_back('"');
  for (char c : s) {
    if (c == '\\' || c == '"') {
      result.push_back('\\');
    }
    result.push_back(c);
  }
  result.push_back('"');
  return result;
}

// Recursively collects ready-valid channels from `signature` and any
// instantiated child found in `child_signatures` (keyed by module_name)
// into `interfaces`.
void CollectInterfaces(
    const verilog::ModuleSignatureProto& signature,
    const std::vector<std::string>& scope_prefix,
    const std::vector<std::string>& name_prefix,
    const absl::flat_hash_map<std::string, verilog::ModuleSignatureProto>&
        child_signatures,
    std::vector<BusInterface>& interfaces) {
  for (const verilog::ChannelInterfaceProto* channel_interface :
       ReadyValidChannelsOf(signature)) {
    std::vector<std::string> name = name_prefix;
    name.push_back(channel_interface->channel_name());
    interfaces.push_back(BusInterface{
        .name = absl::StrJoin(name, "."),
        .scope = scope_prefix,
        .ready_port = channel_interface->streaming().ready_port_name(),
        .valid_port = channel_interface->streaming().valid_port_name(),
    });
  }

  for (const verilog::InstantiationProto& instantiation :
       signature.instantiations()) {
    if (!instantiation.has_block_instantiation()) {
      continue;  // extern/fifo: not a proc, nothing to recurse into.
    }
    const verilog::BlockInstantiationProto& block_instantiation =
        instantiation.block_instantiation();
    auto it = child_signatures.find(block_instantiation.block_name());
    if (it == child_signatures.end()) {
      LOG(WARNING) << "instance " << block_instantiation.instance_name()
                   << " (block " << block_instantiation.block_name()
                   << ") has no matching child signature; skipping its "
                      "internal channels";
      continue;
    }
    std::vector<std::string> child_scope = scope_prefix;
    child_scope.push_back(block_instantiation.instance_name());
    std::vector<std::string> child_name = name_prefix;
    child_name.push_back(block_instantiation.instance_name());
    CollectInterfaces(it->second, child_scope, child_name, child_signatures,
                       interfaces);
  }
}

void AppendClockResetBlock(const verilog::ModuleSignatureProto& signature,
                           std::vector<std::string>& lines) {
  std::string_view clock_name = signature.clock_name().empty()
                                     ? kDefaultClockName
                                     : signature.clock_name();
  std::string_view reset_name =
      signature.has_reset() ? signature.reset().name() : kDefaultResetName;
  bool reset_active_low =
      signature.has_reset() && signature.reset().active_low();

  if (signature.clock_name().empty() || !signature.has_reset()) {
    LOG(WARNING) << "signature has no clock_name/reset name; each interface "
                    "will need clock/reset filled in by hand";
  }

  lines.push_back("common_clk_rst_ifs:");
  lines.push_back(absl::StrFormat("  %s: &%s", kClockAlias, kClockAlias));
  lines.push_back(absl::StrFormat("    clock: %s", YamlString(clock_name)));
  lines.push_back(absl::StrFormat("    reset: %s", YamlString(reset_name)));
  lines.push_back(absl::StrFormat("    reset_type: %s",
                                  YamlString(reset_active_low ? "low"
                                                              : "high")));
}

void AppendInterfaceEntry(const BusInterface& interface,
                          std::vector<std::string>& lines) {
  lines.push_back(absl::StrFormat("  %s:", YamlString(interface.name)));
  lines.push_back(absl::StrCat(
      "    scope: [",
      absl::StrJoin(interface.scope, ", ",
                    [](std::string* out, const std::string& scope_part) {
                      absl::StrAppend(out, YamlString(scope_part));
                    }),
      "]"));
  lines.push_back(absl::StrFormat("    clk_rst_if: *%s", kClockAlias));
  lines.push_back("");
  lines.push_back("    handshake: \"ReadyValid\"");
  lines.push_back(
      absl::StrFormat("    ready: %s", YamlString(interface.ready_port)));
  lines.push_back(
      absl::StrFormat("    valid: %s", YamlString(interface.valid_port)));
  lines.push_back("");
}

}  // namespace

absl::StatusOr<std::string> GenerateBusperfYaml(
    const verilog::ModuleSignatureProto& signature,
    absl::Span<const std::string> scope,
    const absl::flat_hash_map<std::string, verilog::ModuleSignatureProto>&
        child_signatures) {
  std::vector<std::string> scope_prefix(scope.begin(), scope.end());
  std::vector<BusInterface> interfaces;
  CollectInterfaces(signature, scope_prefix, /*name_prefix=*/{},
                     child_signatures, interfaces);
  if (interfaces.empty()) {
    return absl::InvalidArgumentError(
        "No CHANNEL_FLOW_CONTROL_READY_VALID channel_interfaces found "
        "(top-level or in any child signature); refusing to emit a "
        "busperf YAML with an empty interfaces: block");
  }

  std::vector<std::string> lines;
  lines.push_back("# Auto-generated by xls_sig_to_busperf from an XLS");
  lines.push_back(absl::StrFormat("# ModuleSignatureProto for module %s.",
                                  YamlString(signature.module_name())));
  AppendClockResetBlock(signature, lines);
  lines.push_back("");
  lines.push_back("interfaces:");
  for (const BusInterface& interface : interfaces) {
    AppendInterfaceEntry(interface, lines);
  }

  return absl::StrJoin(lines, "\n");
}

}  // namespace xls::busperf
