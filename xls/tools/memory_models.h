// Copyright 2024 The XLS Authors
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

#ifndef XLS_TOOLS_MEMORY_MODELS_H_
#define XLS_TOOLS_MEMORY_MODELS_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/ir/ram_rewrite.pb.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls {

namespace memory_model {

// XLS doesn't have X. Fill with all 1s, as this is generally more likely
// to expose logical problems.
Value XsOfType(Type* type);

class ProcMemoryModel {
 public:
  virtual ~ProcMemoryModel() = default;
  virtual absl::Status Tick() = 0;
  virtual absl::Status Reset() = 0;
};

class AbstractProcMemoryModel : public ProcMemoryModel {
 public:
  AbstractProcMemoryModel(const std::string& name, int64_t size,
                          ChannelQueue* read_request_channel,
                          ChannelQueue* read_response_channel,
                          ChannelQueue* write_request_channel,
                          ChannelQueue* write_response_channel);

  absl::Status Tick() override;
  absl::Status Reset() override;

 private:
  std::string name_;

  std::vector<Value> elements_;

  Type* elements_type_;

  ChannelQueue* read_request_channel_;
  ChannelQueue* read_response_channel_;
  ChannelQueue* write_request_channel_;
  ChannelQueue* write_response_channel_;
};

absl::StatusOr<std::unique_ptr<ProcMemoryModel>> CreateAbstractProcMemoryModel(
    const RamRewriteProto& ram_rewrite, ChannelQueueManager& queue_manager);

absl::StatusOr<std::unique_ptr<ProcMemoryModel>> CreateRewrittenProcMemoryModel(
    const RamRewriteProto& ram_rewrite, ChannelQueueManager& queue_manager);

class BlockMemoryModel {
 public:
  BlockMemoryModel(const std::string& name, size_t size,
                   const Value& initial_value, const Value& read_disabled_value,
                   bool show_trace);

  absl::Status Read(int64_t addr);
  Value GetValueReadLastTick() const;
  bool DidReadLastTick() const;
  bool DidWriteLastTick() const;
  absl::Status Write(int64_t addr, const Value& value);
  absl::Status Tick();
  absl::Status Reset();

 private:
  const std::string name_;
  const Value initial_value_;
  const Value read_disabled_value_;
  std::vector<Value> elements_;
  std::optional<std::pair<int64_t, Value>> write_this_tick_;
  bool write_valid_last_tick_ = false;
  std::optional<Value> read_this_tick_;
  std::optional<Value> read_last_tick_;
  const bool show_trace_;
};

}  // namespace memory_model

}  // namespace xls

#endif  // XLS_TOOLS_MEMORY_MODELS_H_
