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

#include "xls/tools/eval_proc.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/random/distributions.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/interpreter/interpreter_proc_runtime.h"
#include "xls/interpreter/random_value.h"
#include "xls/interpreter/serial_proc_runtime.h"
#include "xls/ir/channel.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/proc_elaboration.h"
#include "xls/ir/value.h"
#include "xls/jit/jit_evaluator_options.h"
#include "xls/jit/jit_proc_runtime.h"

namespace xls {
namespace {

class CyclicValueGenerator {
 public:
  explicit CyclicValueGenerator(std::vector<Value> values)
      : values_(std::move(values)) {}

  std::optional<Value> operator()() {
    if (values_.empty()) {
      return std::nullopt;
    }
    const Value& value = values_[index_];
    index_ = (index_ + 1) % static_cast<int64_t>(values_.size());
    return value;
  }

 private:
  std::vector<Value> values_;
  int64_t index_ = 0;
};

std::vector<Value> GenerateRandomValuesForChannel(Channel* channel,
                                                  int64_t total_ticks,
                                                  absl::BitGenRef rng) {
  std::vector<Value> values;
  if (total_ticks <= 0) {
    return values;
  }
  int64_t count = absl::Uniform<int64_t>(rng, 0, total_ticks);
  values.reserve(count);
  for (int64_t i = 0; i < count; ++i) {
    values.push_back(RandomValue(channel->type(), rng));
  }
  return values;
}

}  // namespace

absl::StatusOr<std::unique_ptr<SerialProcRuntime>> CreateSerialProcRuntime(
    Package* package, const SerialProcRuntimeOptions& options) {
  XLS_RET_CHECK_NE(package, nullptr);
  if (options.use_jit) {
    JitEvaluatorOptions jit_options;
    jit_options.set_opt_level(options.llvm_opt_level);
    return CreateJitSerialProcRuntime(package, options.evaluator_options,
                                      jit_options);
  }
  return CreateInterpreterSerialProcRuntime(package, options.evaluator_options);
}

absl::StatusOr<std::unique_ptr<SerialProcRuntime>> CreateSerialProcRuntime(
    Proc* top, const SerialProcRuntimeOptions& options) {
  XLS_RET_CHECK_NE(top, nullptr);
  if (!top->is_new_style_proc()) {
    return CreateSerialProcRuntime(top->package(), options);
  }
  if (options.use_jit) {
    JitEvaluatorOptions jit_options;
    jit_options.set_opt_level(options.llvm_opt_level);
    return CreateJitSerialProcRuntime(top, options.evaluator_options,
                                      jit_options);
  }
  return CreateInterpreterSerialProcRuntime(top, options.evaluator_options);
}

absl::StatusOr<absl::btree_map<std::string, std::vector<Value>>>
GenerateRandomInputsForPackage(Package* package, int64_t total_ticks,
                               absl::BitGenRef rng) {
  XLS_RET_CHECK_NE(package, nullptr);
  std::vector<Channel*> channels;
  if (package->ChannelsAreProcScoped()) {
    XLS_ASSIGN_OR_RETURN(Proc * top, package->GetTopAsProc());
    for (Channel* channel : top->channels()) {
      if (channel->CanReceive() && !channel->CanSend()) {
        channels.push_back(channel);
      }
    }
  } else {
    for (Channel* channel : package->channels()) {
      if (channel->CanReceive() && !channel->CanSend()) {
        channels.push_back(channel);
      }
    }
  }
  absl::btree_map<std::string, std::vector<Value>> inputs;
  for (Channel* channel : channels) {
    inputs[std::string(channel->name())] =
        GenerateRandomValuesForChannel(channel, total_ticks, rng);
  }
  return inputs;
}

absl::Status AttachRandomInputGenerators(SerialProcRuntime& runtime,
                                         absl::BitGenRef rng,
                                         int64_t pool_size) {
  XLS_RET_CHECK_GT(pool_size, 0);
  for (ChannelInstance* channel_instance :
       runtime.elaboration().channel_instances()) {
    Channel* channel = channel_instance->channel;
    if (!channel->CanReceive() || channel->CanSend()) {
      continue;
    }
    ChannelQueue& queue = runtime.queue_manager().GetQueue(channel_instance);
    if (channel->kind() == ChannelKind::kSingleValue) {
      XLS_RETURN_IF_ERROR(queue.Write(RandomValue(channel->type(), rng)));
      continue;
    }
    std::vector<Value> pool;
    pool.reserve(pool_size);
    for (int64_t i = 0; i < pool_size; ++i) {
      pool.push_back(RandomValue(channel->type(), rng));
    }
    XLS_RETURN_IF_ERROR(
        queue.AttachGenerator(CyclicValueGenerator(std::move(pool))));
  }
  return absl::OkStatus();
}

void DrainOutputChannels(SerialProcRuntime& runtime) {
  for (ChannelInstance* channel_instance :
       runtime.elaboration().channel_instances()) {
    Channel* channel = channel_instance->channel;
    if (channel->CanSend() && !channel->CanReceive() &&
        channel->kind() == ChannelKind::kStreaming) {
      ChannelQueue& queue = runtime.queue_manager().GetQueue(channel_instance);
      while (queue.Read().has_value()) {
      }
    }
  }
}

}  // namespace xls
