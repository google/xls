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
#include "xls/jit/jit_wrapper_generator.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"

namespace xls {
namespace {

using ::testing::HasSubstr;

TEST(JitWrapperGeneratorTest, GeneratesHeaderGuards) {
  constexpr const char kClassName[] = "MyClass";
  const std::filesystem::path kHeaderPath =
      "some/silly/genfiles/path/this_is_myclass.h";
  constexpr const char kNamespace[] = "my_namespace";

  const std::string program = R"(package p
fn foo(x: bits[4]) -> bits[4] {
  ret identity.2: bits[4] = identity(x)
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("foo"));
  GeneratedJitWrapper generated = GenerateJitWrapper(
      *f, kClassName, kNamespace, kHeaderPath, "some/silly/genfiles/path");
  int64_t pos = generated.header.find("THIS_IS_MYCLASS_H_");
  EXPECT_NE(pos, std::string::npos);

  generated.header = generated.header.substr(pos);
  pos = generated.header.find("THIS_IS_MYCLASS_H_");
  EXPECT_NE(pos, std::string::npos);

  generated.header = generated.header.substr(pos);
  pos = generated.header.find("THIS_IS_MYCLASS_H_");
  EXPECT_NE(pos, std::string::npos);
}

TEST(JitWrapperGeneratorTest, GeneratesNativeInts) {
  constexpr const char kClassName[] = "MyClass";
  const std::filesystem::path kHeaderPath =
      "some/silly/genfiles/path/this_is_myclass.h";
  constexpr const char kNamespace[] = "my_namespace";

  const std::string program = R"(package p
fn foo8(x: bits[8]) -> bits[8] {
  ret identity.2: bits[8] = identity(x)
}

fn foo16(x: bits[16]) -> bits[16] {
  ret identity.4: bits[16] = identity(x)
}

fn foo32(x: bits[32]) -> bits[32] {
  ret identity.6: bits[32] = identity(x)
}

fn foo64(x: bits[64]) -> bits[64] {
  ret identity.8: bits[64] = identity(x)
})";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("foo8"));
  GeneratedJitWrapper generated = GenerateJitWrapper(
      *f, kClassName, kNamespace, kHeaderPath, "some/silly/genfiles/path");
  int64_t pos =
      generated.header.find("absl::StatusOr<uint8_t> Run(uint8_t x);");
  EXPECT_NE(pos, std::string::npos);

  XLS_ASSERT_OK_AND_ASSIGN(f, p->GetFunction("foo16"));
  generated = GenerateJitWrapper(*f, kClassName, kNamespace, kHeaderPath,
                                 "some/silly/genfiles/path");
  pos = generated.header.find("absl::StatusOr<uint16_t> Run(uint16_t x);");
  EXPECT_NE(pos, std::string::npos);

  XLS_ASSERT_OK_AND_ASSIGN(f, p->GetFunction("foo32"));
  generated = GenerateJitWrapper(*f, kClassName, kNamespace, kHeaderPath,
                                 "some/silly/genfiles/path");
  pos = generated.header.find("absl::StatusOr<uint32_t> Run(uint32_t x);");
  EXPECT_NE(pos, std::string::npos);

  XLS_ASSERT_OK_AND_ASSIGN(f, p->GetFunction("foo64"));
  generated = GenerateJitWrapper(*f, kClassName, kNamespace, kHeaderPath,
                                 "some/silly/genfiles/path");
  pos = generated.header.find("absl::StatusOr<uint64_t> Run(uint64_t x);");
  EXPECT_NE(pos, std::string::npos);
}

TEST(JitWrapperGeneratorTest, GeneratesNonnativeInts) {
  constexpr const char kClassName[] = "MyClass";
  const std::filesystem::path kHeaderPath =
      "some/silly/genfiles/path/this_is_myclass.h";
  constexpr const char kNamespace[] = "my_namespace";

  const std::string program = R"(package p

fn foo5(x: bits[5]) -> bits[5] {
  ret identity.2: bits[5] = identity(x)
}

fn foo13(x: bits[13]) -> bits[13] {
  ret identity.4: bits[13] = identity(x)
}

fn foo22(x: bits[22]) -> bits[22] {
  ret identity.6: bits[22] = identity(x)
}

fn foo63(x: bits[63]) -> bits[63] {
  ret identity.8: bits[63] = identity(x)
}

fn foo65(x: bits[65]) -> bits[65] {
  ret identity.16: bits[65] = identity(x)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("foo5"));
  GeneratedJitWrapper generated = GenerateJitWrapper(
      *f, kClassName, kNamespace, kHeaderPath, "some/silly/genfiles/path");
  int64_t pos =
      generated.header.find("absl::StatusOr<uint8_t> Run(uint8_t x);");
  EXPECT_NE(pos, std::string::npos);

  XLS_ASSERT_OK_AND_ASSIGN(f, p->GetFunction("foo13"));
  generated = GenerateJitWrapper(*f, kClassName, kNamespace, kHeaderPath,
                                 "some/silly/genfiles/path");
  pos = generated.header.find("absl::StatusOr<uint16_t> Run(uint16_t x);");
  EXPECT_NE(pos, std::string::npos);

  XLS_ASSERT_OK_AND_ASSIGN(f, p->GetFunction("foo22"));
  generated = GenerateJitWrapper(*f, kClassName, kNamespace, kHeaderPath,
                                 "some/silly/genfiles/path");
  pos = generated.header.find("absl::StatusOr<uint32_t> Run(uint32_t x);");
  EXPECT_NE(pos, std::string::npos);

  XLS_ASSERT_OK_AND_ASSIGN(f, p->GetFunction("foo63"));
  generated = GenerateJitWrapper(*f, kClassName, kNamespace, kHeaderPath,
                                 "some/silly/genfiles/path");
  pos = generated.header.find("absl::StatusOr<uint64_t> Run(uint64_t x);");
  EXPECT_NE(pos, std::string::npos);

  // Verify we stop at 64 bits (we'll handle 128b integers once they're part of
  // a standard).
  XLS_ASSERT_OK_AND_ASSIGN(f, p->GetFunction("foo65"));
  generated = GenerateJitWrapper(*f, kClassName, kNamespace, kHeaderPath,
                                 "some/silly/genfiles/path");
  pos = generated.header.find("absl::StatusOr<uint64_t> Run(uint64_t x);");
  EXPECT_EQ(pos, std::string::npos);
}

TEST(JitWrapperGeneratorTest, GeneratesTokenActivatedFunction) {
  constexpr const char kClassName[] = "MyClass";
  const std::filesystem::path kHeaderPath =
      "some/silly/genfiles/path/this_is_myclass.h";
  constexpr const char kNamespace[] = "my_namespace";

  const std::string program = R"(package p

fn main(t: token, activated: bits[1], x: bits[32]) -> (token, bits[32]) {
  ret r: (token, bits[32]) = tuple(t, x)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(program));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction("main"));
  GeneratedJitWrapper generated = GenerateJitWrapper(
      *f, kClassName, kNamespace, kHeaderPath, "some/silly/genfiles/path");
  EXPECT_THAT(
      generated.header,
      HasSubstr(
          "Run(xls::PackedBitsView<32> x, xls::PackedBitsView<32> result)"));
  EXPECT_THAT(generated.header,
              HasSubstr("absl::StatusOr<xls::Value> Run(xls::Value x)"));
}

}  // namespace
}  // namespace xls
