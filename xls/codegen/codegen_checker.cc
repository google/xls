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

#include "xls/codegen/codegen_checker.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/node.h"
#include "xls/ir/register.h"
#include "xls/ir/verifier.h"
#include "xls/passes/pass_base.h"

namespace xls::verilog {

namespace {

absl::Status CheckNodeToStageMap(const CodegenContext& context) {
  for (const auto& [block, metadata] : context.metadata()) {
    XLS_RET_CHECK_EQ(
        metadata.streaming_io_and_pipeline.node_to_stage_map.size(),
        absl::c_count_if(
            block->nodes(),
            [&node_to_stage_map = std::as_const(
                 metadata.streaming_io_and_pipeline.node_to_stage_map)](
                Node* n) { return node_to_stage_map.contains(n); }))
        << "Dangling pointers present in node-id-to-stage map\n";
  }
  return absl::OkStatus();
}

// Verify that all nodes, instantiations, and registers in `streaming_io` are
// contained in `block`.
absl::Status CheckStreamingIO(const StreamingIOPipeline& streaming_io,
                              Block* block) {
  absl::flat_hash_set<Node*> nodes(block->nodes().begin(),
                                   block->nodes().end());
  absl::flat_hash_set<::xls::Instantiation*> instantiations(
      block->GetInstantiations().begin(), block->GetInstantiations().end());
  absl::flat_hash_set<Register*> registers(block->GetRegisters().begin(),
                                           block->GetRegisters().end());
  for (absl::Span<StreamingInput const> streaming_inputs :
       streaming_io.inputs) {
    for (const StreamingInput& input : streaming_inputs) {
      XLS_RET_CHECK(!input.GetInstantiation().has_value() ||
                    instantiations.contains(*input.GetInstantiation()));
      if (input.GetDataPort().has_value()) {
        XLS_RET_CHECK(nodes.contains(*input.GetDataPort()))
            << absl::StreamFormat("Port not found for %s",
                                  input.GetChannelName());
      }
      if (input.GetReadyPort().has_value()) {
        XLS_RET_CHECK(nodes.contains(*input.GetReadyPort()))
            << absl::StreamFormat("Ready port not found for %s",
                                  input.GetChannelName());
      }
      if (input.GetValidPort().has_value()) {
        XLS_RET_CHECK(nodes.contains(*input.GetValidPort()))
            << absl::StreamFormat("Valid port not found for %s",
                                  input.GetChannelName());
      }
      if (input.GetSignalData().has_value()) {
        XLS_RET_CHECK(nodes.contains(*input.GetSignalData()))
            << absl::StreamFormat("Signal data not found for %s",
                                  input.GetChannelName());
      }
      if (input.GetSignalValid().has_value()) {
        XLS_RET_CHECK(nodes.contains(*input.GetSignalValid()))
            << absl::StreamFormat("Signal valid not found for %s",
                                  input.GetChannelName());
      }
      if (input.GetPredicate().has_value()) {
        XLS_RET_CHECK(nodes.contains(*input.GetPredicate()))
            << absl::StreamFormat("Predicate not found for %s",
                                  input.GetChannelName());
      }
    }
  }
  for (absl::Span<StreamingOutput const> streaming_outputs :
       streaming_io.outputs) {
    for (const StreamingOutput& output : streaming_outputs) {
      XLS_RET_CHECK(!output.GetInstantiation().has_value() ||
                    instantiations.contains(*output.GetInstantiation()));
      if (output.GetDataPort().has_value()) {
        XLS_RET_CHECK(nodes.contains(*output.GetDataPort()));
      }
      if (output.GetReadyPort().has_value()) {
        XLS_RET_CHECK(nodes.contains(*output.GetReadyPort()));
      }
      if (output.GetValidPort().has_value()) {
        XLS_RET_CHECK(nodes.contains(*output.GetValidPort()));
      }
      if (output.GetPredicate().has_value()) {
        XLS_RET_CHECK(nodes.contains(*output.GetPredicate()))
            << absl::StreamFormat("Predicate not found for %s",
                                  output.GetChannelName());
      }
    }
  }
  for (const SingleValueInput& input : streaming_io.single_value_inputs) {
    if (input.GetDataPort().has_value()) {
      XLS_RET_CHECK(nodes.contains(*input.GetDataPort()));
    }
  }
  for (const SingleValueOutput& output : streaming_io.single_value_outputs) {
    if (output.GetDataPort().has_value()) {
      XLS_RET_CHECK(nodes.contains(*output.GetDataPort()));
    }
  }
  for (const PipelineStageRegisters& stage_registers :
       streaming_io.pipeline_registers) {
    for (const PipelineRegister& stage_register : stage_registers) {
      XLS_RET_CHECK(registers.contains(stage_register.reg));
      XLS_RET_CHECK(nodes.contains(stage_register.reg_read));
      XLS_RET_CHECK(nodes.contains(stage_register.reg_write));
    }
  }
  for (const std::optional<StateRegister>& state_register :
       streaming_io.state_registers) {
    if (state_register.has_value()) {
      if (state_register->reg != nullptr) {
        XLS_RET_CHECK(registers.contains(state_register->reg));
        XLS_RET_CHECK(nodes.contains(state_register->reg_read));
        for (RegisterWrite* reg_write : state_register->reg_writes) {
          XLS_RET_CHECK(nodes.contains(reg_write));
        }
      }
      if (state_register->reg_full.has_value()) {
        XLS_RET_CHECK(registers.contains(state_register->reg_full->reg));
        XLS_RET_CHECK(nodes.contains(state_register->reg_full->read));
        for (RegisterWrite* reg_full_set : state_register->reg_full->sets) {
          XLS_RET_CHECK(nodes.contains(reg_full_set));
        }
      }
    }
  }
  if (streaming_io.idle_port.has_value()) {
    XLS_RET_CHECK(nodes.contains(*streaming_io.idle_port))
        << absl::StreamFormat("Idle port not found for %s", block->name());
  }
  int64_t stage = 0;
  for (const std::optional<Node*>& node : streaming_io.pipeline_valid) {
    XLS_RET_CHECK(!node.has_value() || nodes.contains(*node))
        << absl::StreamFormat("Stage %d pipeline valid not found for %s", stage,
                              block->name());
    stage++;
  }
  for (const std::optional<Node*>& node : streaming_io.stage_valid) {
    XLS_RET_CHECK(!node.has_value() || nodes.contains(*node))
        << absl::StreamFormat("Stage %d valid not found for %s", stage,
                              block->name());
  }
  for (const std::optional<Node*>& node : streaming_io.stage_done) {
    XLS_RET_CHECK(!node.has_value() || nodes.contains(*node))
        << absl::StreamFormat("Stage %d done not found for %s", stage,
                              block->name());
  }
  for (const auto& [node, _] : streaming_io.node_to_stage_map) {
    XLS_RET_CHECK(nodes.contains(node));
  }
  return absl::OkStatus();
}

}  // namespace
absl::Status CodegenChecker::Run(Package* package,
                                 const CodegenPassOptions& options,
                                 PassResults* results,
                                 CodegenContext& context) const {
  XLS_RETURN_IF_ERROR(CheckNodeToStageMap(context)) << package->DumpIr();
  for (const auto& [block, metadata] : context.metadata()) {
    XLS_RETURN_IF_ERROR(
        CheckStreamingIO(metadata.streaming_io_and_pipeline, block));
  }
  return VerifyPackage(package);
}

}  // namespace xls::verilog
