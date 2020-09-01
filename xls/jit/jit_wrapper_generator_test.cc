// Copyright 2020 Google LLC
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
#include "xls/jit/jit_wrapper_generator.h"

#include <filesystem>
#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"

namespace xls {
namespace {

TEST(JitWrapperGeneratorTest, GeneratesHeaderGuards) {
  constexpr const char kClassName[] = "MyClass";
  const std::filesystem::path kHeaderPath =
      "some/silly/genfiles/path/this_is_myclass.h";

  const std::string program = R"(package p
fn foo(x: bits[4]) -> bits[4] {
  ret identity.2: bits[4] = identity(x)
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("foo"));
  GeneratedJitWrapper generated = GenerateJitWrapper(
      *f, kClassName, kHeaderPath, "some/silly/genfiles/path");
  int64 pos = generated.header.find("THIS_IS_MYCLASS_H_");
  EXPECT_NE(pos, std::string::npos);

  generated.header = generated.header.substr(pos);
  pos = generated.header.find("THIS_IS_MYCLASS_H_");
  EXPECT_NE(pos, std::string::npos);

  generated.header = generated.header.substr(pos);
  pos = generated.header.find("THIS_IS_MYCLASS_H_");
  EXPECT_NE(pos, std::string::npos);
}

}  // namespace
}  // namespace xls
