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

#include "xls/tools/memory_models.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/ir/ram_rewrite.pb.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"

namespace xls {

namespace memory_model {

// XLS doesn't have X. Fill with all 1s, as this is generally more likely
// to expose logical problems.
Value XsOfType(Type* type) { return AllOnesOfType(type); }

// TODO: Implement in XLS using XLS IR (DSLX/C++ source) google/xls#1638
// Possibly replace with ram.x, which also implements different
// simultaneous read/write behaviors.
// This model always does read before write.
AbstractProcMemoryModel::AbstractProcMemoryModel(
    const std::string& name, int64_t size, ChannelQueue* read_request_channel,
    ChannelQueue* read_response_channel, ChannelQueue* write_request_channel,
    ChannelQueue* write_response_channel)
    : name_(name),
      read_request_channel_(read_request_channel),
      read_response_channel_(read_response_channel),
      write_request_channel_(write_request_channel),
      write_response_channel_(write_response_channel) {
  Type* read_response_type = read_response_channel->channel()->type();
  CHECK(read_response_type->IsTuple());
  CHECK_EQ(read_response_type->AsTupleOrDie()->element_types().size(), 1);
  elements_type_ = read_response_type->AsTupleOrDie()->element_type(0);

  elements_.resize(size, XsOfType(elements_type_));
}

absl::Status AbstractProcMemoryModel::Tick() {
  while (!read_request_channel_->IsEmpty()) {
    std::optional<Value> opt_read_req = read_request_channel_->Read();
    XLS_RET_CHECK(opt_read_req.has_value());
    const Value& read_req = opt_read_req.value();
    XLS_RET_CHECK(read_req.IsTuple());
    XLS_RET_CHECK_EQ(read_req.elements().size(), 2);
    const Value& addr = read_req.element(0);
    XLS_ASSIGN_OR_RETURN(int64_t addr_i, addr.bits().ToUint64());
    if (addr_i < 0 || addr_i >= elements_.size()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Read request address %s out of range [0, %li)",
                          addr.ToString().c_str(), elements_.size()));
    }
    VLOG(5) << name_ << " reading from address " << addr_i << " data "
            << elements_.at(addr_i);
    XLS_RETURN_IF_ERROR(
        read_response_channel_->Write(Value::Tuple({elements_.at(addr_i)})));
  }

  while (!write_request_channel_->IsEmpty()) {
    std::optional<Value> opt_write_req = write_request_channel_->Read();
    XLS_RET_CHECK(opt_write_req.has_value());
    const Value& write_req = opt_write_req.value();
    XLS_RET_CHECK(write_req.IsTuple());
    XLS_RET_CHECK_EQ(write_req.elements().size(), 3);
    const Value& addr = write_req.element(0);
    const Value& data = write_req.element(1);
    XLS_ASSIGN_OR_RETURN(int64_t addr_i, addr.bits().ToUint64());
    if (addr_i < 0 || addr_i >= elements_.size()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Write request address %s out of range [0, %li)",
                          addr.ToString().c_str(), elements_.size()));
    }
    VLOG(5) << name_ << " writing to address " << addr_i << " with data "
            << data;
    elements_.at(addr_i) = data;
    XLS_RETURN_IF_ERROR(write_response_channel_->Write(Value::Tuple({})));
  }

  return absl::OkStatus();
}

absl::Status AbstractProcMemoryModel::Reset() {
  elements_.assign(elements_.size(), XsOfType(elements_type_));
  return absl::OkStatus();
}

// TODO: Implement in XLS using XLS IR (DSLX/C++ source) google/xls#1638
// Possibly replace with ram.x, which also implements different
// simultaneous read/write behaviors.
// This model always does read before write.
class RewrittenProcMemoryModel : public ProcMemoryModel {
 public:
  RewrittenProcMemoryModel(const std::string& name, int64_t size,
                           ChannelQueue* request_channel,
                           ChannelQueue* response_channel,
                           ChannelQueue* write_completion_channel)
      : name_(name),
        request_channel_(request_channel),
        response_channel_(response_channel),
        write_completion_channel_(write_completion_channel) {
    Type* read_response_type = response_channel_->channel()->type();
    CHECK(read_response_type->IsTuple());
    CHECK_EQ(read_response_type->AsTupleOrDie()->element_types().size(), 1);
    elements_type_ = read_response_type->AsTupleOrDie()->element_type(0);

    elements_.resize(size, AllOnesOfType(elements_type_));
  }

  absl::Status Tick() final {
    struct Request {
      bool read = false;
      bool write = false;
      int64_t addr = -1;
      Value data;
    };
    std::vector<Request> requests;
    while (!request_channel_->IsEmpty()) {
      std::optional<Value> opt_req = request_channel_->Read();
      XLS_RET_CHECK(opt_req.has_value());
      const Value& req = opt_req.value();
      XLS_RET_CHECK(req.IsTuple());
      XLS_RET_CHECK_EQ(req.elements().size(), 6);
      const Value& addr = req.element(0);
      const Value& read = req.element(5);
      const Value& write = req.element(4);

      Request request;
      XLS_ASSIGN_OR_RETURN(request.addr, addr.bits().ToUint64());
      request.data = req.element(1);
      request.read = read.IsAllOnes();
      request.write = write.IsAllOnes();
      requests.push_back(request);

      if (request.read || request.write) {
        if (request.addr < 0 || request.addr >= elements_.size()) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Request address %s to memory %s out of range [0, %li)",
              addr.ToString(), name_, elements_.size()));
        }
      }

      XLS_RET_CHECK(!(request.read && request.write));
    }
    // Reads first
    for (const Request& request : requests) {
      if (!request.read) {
        continue;
      }
      XLS_RETURN_IF_ERROR(
          response_channel_->Write(Value::Tuple({elements_.at(request.addr)})));
    }
    // Then writes
    for (const Request& request : requests) {
      if (!request.write) {
        continue;
      }
      elements_.at(request.addr) = request.data;
      XLS_RETURN_IF_ERROR(write_completion_channel_->Write(Value::Tuple({})));
    }
    return absl::OkStatus();
  }

  absl::Status Reset() final {
    elements_.assign(elements_.size(), XsOfType(elements_type_));
    return absl::OkStatus();
  }

 private:
  std::string name_;

  std::vector<Value> elements_;

  Type* elements_type_;

  ChannelQueue* request_channel_;
  ChannelQueue* response_channel_;
  ChannelQueue* write_completion_channel_;
};

absl::StatusOr<std::unique_ptr<ProcMemoryModel>> CreateAbstractProcMemoryModel(
    const RamRewriteProto& ram_rewrite, ChannelQueueManager& queue_manager) {
  ChannelQueue* read_request_queue = nullptr;
  ChannelQueue* read_response_queue = nullptr;
  ChannelQueue* write_request_queue = nullptr;
  ChannelQueue* write_response_queue = nullptr;

  for (const auto& [logical_name, physical_name] :
       ram_rewrite.from_channels_logical_to_physical()) {
    if (logical_name == "abstract_read_req") {
      XLS_ASSIGN_OR_RETURN(read_request_queue,
                           queue_manager.GetQueueByName(physical_name));
    } else if (logical_name == "abstract_read_resp") {
      XLS_ASSIGN_OR_RETURN(read_response_queue,
                           queue_manager.GetQueueByName(physical_name));
    } else if (logical_name == "abstract_write_req") {
      XLS_ASSIGN_OR_RETURN(write_request_queue,
                           queue_manager.GetQueueByName(physical_name));
    } else if (logical_name == "write_completion") {
      XLS_ASSIGN_OR_RETURN(write_response_queue,
                           queue_manager.GetQueueByName(physical_name));
    } else {
      return absl::UnimplementedError(
          absl::StrFormat("Unsupported logical name in RAM rewrite %s: %s",
                          ram_rewrite.to_name_prefix(), logical_name));
    }
  }

  if (read_request_queue == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("No read request channel found for RAM rewrite %s",
                        ram_rewrite.to_name_prefix()));
  }
  if (read_response_queue == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("No read response channel found for RAM rewrite %s",
                        ram_rewrite.to_name_prefix()));
  }
  if (write_request_queue == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("No write request channel found for RAM rewrite %s",
                        ram_rewrite.to_name_prefix()));
  }
  if (write_response_queue == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("No write response channel found for RAM rewrite %s",
                        ram_rewrite.to_name_prefix()));
  }

  auto memory_model = std::make_unique<AbstractProcMemoryModel>(
      ram_rewrite.to_name_prefix(),
      /*size=*/ram_rewrite.from_config().depth(), read_request_queue,
      read_response_queue, write_request_queue, write_response_queue);

  return std::move(memory_model);
}

absl::StatusOr<std::unique_ptr<ProcMemoryModel>> CreateRewrittenProcMemoryModel(
    const RamRewriteProto& ram_rewrite, ChannelQueueManager& queue_manager) {
  XLS_ASSIGN_OR_RETURN(
      ChannelQueue * request_queue,
      queue_manager.GetQueueByName(ram_rewrite.to_name_prefix() + "_req"));
  XLS_ASSIGN_OR_RETURN(
      ChannelQueue * response_queue,
      queue_manager.GetQueueByName(ram_rewrite.to_name_prefix() + "_resp"));
  XLS_ASSIGN_OR_RETURN(ChannelQueue * write_completion_queue,
                       queue_manager.GetQueueByName(
                           ram_rewrite.to_name_prefix() + "_write_completion"));

  if (request_queue == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("No request channel found for RAM rewrite %s",
                        ram_rewrite.to_name_prefix()));
  }
  if (response_queue == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("No response channel found for RAM rewrite %s",
                        ram_rewrite.to_name_prefix()));
  }
  if (write_completion_queue == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("No write completion channel found for RAM rewrite %s",
                        ram_rewrite.to_name_prefix()));
  }

  auto memory_model = std::make_unique<RewrittenProcMemoryModel>(
      ram_rewrite.to_name_prefix(),
      /*size=*/ram_rewrite.from_config().depth(), request_queue, response_queue,
      write_completion_queue);

  return std::move(memory_model);
}

// TODO: Implement in XLS using XLS IR (DSLX/C++ source) google/xls#1638
// Possibly replace with ram.x, which also implements different
// simultaneous read/write behaviors.
BlockMemoryModel::BlockMemoryModel(const std::string& name, size_t size,
                                   const Value& initial_value,
                                   const Value& read_disabled_value,
                                   bool show_trace)
    : name_(name),
      initial_value_(initial_value),
      read_disabled_value_(read_disabled_value),
      show_trace_(show_trace) {
  elements_.resize(size, initial_value);
}
absl::Status BlockMemoryModel::Read(int64_t addr) {
  if (addr < 0 || addr >= elements_.size()) {
    return absl::OutOfRangeError(
        absl::StrFormat("Memory %s read out of range at %i", name_, addr));
  }
  if (read_this_tick_.has_value()) {
    return absl::FailedPreconditionError(
        absl::StrFormat("Memory %s double read in tick at %i", name_, addr));
  }
  read_this_tick_ = elements_[addr];
  if (show_trace_) {
    LOG(INFO) << "Memory Model: Initiated read " << name_ << "[" << addr
              << "] = " << read_this_tick_.value();
  }
  return absl::OkStatus();
}
Value BlockMemoryModel::GetValueReadLastTick() const {
  if (show_trace_) {
    if (read_last_tick_.has_value()) {
      LOG(INFO) << "Memory Model: Got read last value " << name_ << " = "
                << read_last_tick_.value();
    } else {
      LOG(INFO) << "Memory Model: Got read last default " << name_ << " = "
                << read_disabled_value_;
    }
  }
  return read_last_tick_.has_value() ? read_last_tick_.value()
                                     : read_disabled_value_;
}
bool BlockMemoryModel::DidReadLastTick() const {
  return read_last_tick_.has_value();
}
bool BlockMemoryModel::DidWriteLastTick() const {
  return write_valid_last_tick_;
}
absl::Status BlockMemoryModel::Write(int64_t addr, const Value& value) {
  if (addr < 0 || addr >= elements_.size()) {
    return absl::OutOfRangeError(
        absl::StrFormat("Memory %s write out of range at %i", name_, addr));
  }
  if (write_this_tick_.has_value()) {
    return absl::FailedPreconditionError(
        absl::StrFormat("Memory %s double write in tick at %i", name_, addr));
  }
  if (value.GetFlatBitCount() != elements_[0].GetFlatBitCount()) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "Memory %s write value at %i with wrong bit count %i, expected %i",
        name_, addr, value.GetFlatBitCount(), elements_[0].GetFlatBitCount()));
  }
  if (show_trace_) {
    LOG(INFO) << "Memory Model: Initiated write " << name_ << "[" << addr
              << "] = " << value;
  }
  write_this_tick_ = std::make_pair(addr, value);
  return absl::OkStatus();
}
absl::Status BlockMemoryModel::Tick() {
  if (write_this_tick_.has_value()) {
    if (show_trace_) {
      LOG(INFO) << "Memory Model: Committed write " << name_ << "["
                << write_this_tick_->first
                << "] = " << write_this_tick_->second;
    }
    elements_[write_this_tick_->first] = write_this_tick_->second;
    write_valid_last_tick_ = true;
    write_this_tick_.reset();
  } else {
    write_valid_last_tick_ = false;
  }
  read_last_tick_ = read_this_tick_;
  read_this_tick_.reset();
  return absl::OkStatus();
}

absl::Status BlockMemoryModel::Reset() {
  elements_.assign(elements_.size(), initial_value_);
  write_this_tick_.reset();
  read_this_tick_.reset();
  return absl::OkStatus();
}

}  // namespace memory_model

}  // namespace xls
