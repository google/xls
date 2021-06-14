// Copyright 2020 The XLS Authors
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

#ifndef XLS_CODEGEN_PROC_GENERATOR_H_
#define XLS_CODEGEN_PROC_GENERATOR_H_

#include "absl/status/statusor.h"
#include "xls/codegen/codegen_options.h"
#include "xls/ir/proc.h"

namespace xls {
namespace verilog {

// Generates and returns a (System)Verilog module implementing the given proc
// with the specified options. The proc must have no explicit state. That is,
// the state type must be an empty tuple. Typically, the state should have
// already been converted to a channel. Nodes in the proc (send/receive) must
// only communicate over RegisterChannels and PortChannels.
// TODO(https://github.com/google/xls/issues/410): 2021/05/19 Remove this and
// replace with block generator.
absl::StatusOr<std::string> GenerateVerilog(const CodegenOptions& options,
                                            Proc* proc);

}  // namespace verilog
}  // namespace xls

#endif  // XLS_CODEGEN_PROC_GENERATOR_H_
