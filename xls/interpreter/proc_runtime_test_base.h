// Copyright 2022 The XLS Authors
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

#ifndef XLS_INTERPRETER_PROC_RUNTIME_TEST_BASE_H_
#define XLS_INTERPRETER_PROC_RUNTIME_TEST_BASE_H_

#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/interpreter/proc_runtime.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"

namespace xls {

class ProcRuntimeTestParam {
 public:
  ProcRuntimeTestParam(
      std::string_view name,
      std::function<std::unique_ptr<ProcRuntime>(Package*)> runtime_factory)
      : name_(name), runtime_factory_(runtime_factory) {}
  ProcRuntimeTestParam() = default;

  std::unique_ptr<ProcRuntime> CreateRuntime(Package* package) const {
    return runtime_factory_(package);
  }

  std::string name() const { return name_; }

 private:
  std::string name_;
  std::function<std::unique_ptr<ProcRuntime>(Package*)> runtime_factory_;
};

template <typename TestT>
std::string ParameterizedTestName(
    const testing::TestParamInfo<typename TestT::ParamType>& info) {
  return info.param.name();
}

// A suite of test which can be run against arbitrary ProcRuntime
// implementations. Users should instantiate with an INSTANTIATE_TEST_SUITE_P
// macro.
class ProcRuntimeTestBase
    : public IrTestBase,
      public testing::WithParamInterface<ProcRuntimeTestParam> {};

}  // namespace xls

#endif  // XLS_INTERPRETER_PROC_RUNTIME_TEST_BASE_H_
