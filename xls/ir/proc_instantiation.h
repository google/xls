// Copyright 2023 The XLS Authors
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

#ifndef XLS_IR_PROC_INSTANTIATION_H_
#define XLS_IR_PROC_INSTANTIATION_H_

#include <string>
#include <string_view>
#include <vector>

#include "absl/types/span.h"
#include "xls/ir/channel.h"

namespace xls {

class Proc;

// Abstraction representing an instantiation of a proc. Supported for new style
// procs with proc-scoped channels.
class ProcInstantiation {
 public:
  ProcInstantiation(std::string_view name,
                    absl::Span<ChannelInterface* const> channel_args,
                    Proc* proc)
      : name_(name),
        channel_args_(channel_args.begin(), channel_args.end()),
        proc_(proc) {}

  std::string_view name() const { return name_; }

  // The channel arguments to the instantiated proc. These channel interfaces
  // match the type and direction of the interface of the instantiated proc.
  absl::Span<ChannelInterface* const> channel_args() const {
    return channel_args_;
  }

  // Returns the instantiated proc.
  Proc* proc() const { return proc_; }

  std::string ToString() const;

 private:
  std::string name_;
  std::vector<ChannelInterface*> channel_args_;
  Proc* proc_;
};

}  // namespace xls

#endif  // XLS_IR_PROC_INSTANTIATION_H_
