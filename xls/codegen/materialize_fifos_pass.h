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

#ifndef XLS_CODEGEN_MATERIALIZE_FIFOS_PASS_H_
#define XLS_CODEGEN_MATERIALIZE_FIFOS_PASS_H_

#include "absl/status/statusor.h"
#include "xls/codegen/codegen_pass.h"

namespace xls::verilog {

// Materialize FIFO instantiations into blocks. The performance
// characteristics/DV Support may not be ideal for all use cases.
class MaterializeFifosPass : public CodegenPass {
 public:
  MaterializeFifosPass()
      : CodegenPass("materialize_fifos",
                    "Materialize FIFO instantiations into block-instantiations "
                    "that match the same API.") {}

 protected:
  absl::StatusOr<bool> RunInternal(CodegenPassUnit* unit,
                                   const CodegenPassOptions& options,
                                   CodegenPassResults* results) const final;
};

// Helper pass to materialize fifos in cases where there is no configured fifo
// template.
class MaybeMaterializeInternalFifoPass : public CodegenPass {
 public:
  MaybeMaterializeInternalFifoPass()
      : CodegenPass("fallback_fifos",
                    "Materialize FIFO instantiations into block-instantiation "
                    "if no FIFO template is specified.") {}

 protected:
  absl::StatusOr<bool> RunInternal(CodegenPassUnit* unit,
                                   const CodegenPassOptions& options,
                                   CodegenPassResults* results) const final;

 private:
  MaterializeFifosPass materialize_;
};

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_MATERIALIZE_FIFOS_PASS_H_
