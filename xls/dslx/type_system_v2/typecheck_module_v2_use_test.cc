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

#include <filesystem>
#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system/typecheck_test_utils.h"
#include "xls/dslx/type_system_v2/matchers.h"
#include "xls/dslx/virtualizable_file_system.h"

namespace xls::dslx {
namespace {

TEST(TypecheckV2UseTest, UseConstantDirectly) {
  absl::flat_hash_map<std::filesystem::path, std::string> files = {
      {std::filesystem::path("/foo.x"), "pub const D = u32:5;"},
  };
  auto vfs = std::make_unique<FakeFilesystem>(
      files, /*cwd=*/std::filesystem::path("/"));
  ImportData import_data = CreateImportDataForTest(std::move(vfs));
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result,
                           TypecheckV2(R"(#![feature(use_syntax)]
use foo::D;
const X = D;
)",
                                       "fake", &import_data));
  EXPECT_THAT(result, HasTypeInfo(HasNodeWithType("X", "uN[32]")));
}

TEST(TypecheckV2UseTest, UseFunctionDirectly) {
  absl::flat_hash_map<std::filesystem::path, std::string> files = {
      {std::filesystem::path("/foo.x"),
       "pub fn bar(x: u32) -> u32 { x + u32:1 }"},
  };
  auto vfs = std::make_unique<FakeFilesystem>(
      files, /*cwd=*/std::filesystem::path("/"));
  ImportData import_data = CreateImportDataForTest(std::move(vfs));
  XLS_ASSERT_OK(TypecheckV2(R"(#![feature(use_syntax)]
use foo::bar;
fn main() -> u32 { bar(u32:1) }
)",
                            "fake", &import_data));
}

TEST(TypecheckV2UseTest, UseStructDirectly) {
  absl::flat_hash_map<std::filesystem::path, std::string> files = {
      {std::filesystem::path("/foo.x"), "pub struct Point { x: u32, y: u32 }"},
  };
  auto vfs = std::make_unique<FakeFilesystem>(
      files, /*cwd=*/std::filesystem::path("/"));
  ImportData import_data = CreateImportDataForTest(std::move(vfs));
  XLS_ASSERT_OK(TypecheckV2(R"(#![feature(use_syntax)]
use foo::Point;
fn main(p: Point) -> Point { p }
)",
                            "fake", &import_data));
}

TEST(TypecheckV2UseTest, UseTypeAliasDirectly) {
  absl::flat_hash_map<std::filesystem::path, std::string> files = {
      {std::filesystem::path("/foo.x"),
       "pub struct Point { x: u32, y: u32 }\npub type PointAlias = Point;"},
  };
  auto vfs = std::make_unique<FakeFilesystem>(
      files, /*cwd=*/std::filesystem::path("/"));
  ImportData import_data = CreateImportDataForTest(std::move(vfs));
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result,
                           TypecheckV2(R"(#![feature(use_syntax)]
use foo::{Point, PointAlias};
fn make(x: u32, y: u32) -> PointAlias { Point { x, y } }
)",
                                       "fake", &import_data));
  EXPECT_THAT(result, HasTypeInfo(HasNodeWithType(
                          "make", "(uN[32], uN[32]) -> Point")));
}

TEST(TypecheckV2UseTest, UseModuleThenColonRefToConstant) {
  absl::flat_hash_map<std::filesystem::path, std::string> files = {
      {std::filesystem::path("/foo.x"), "pub const D = u32:5;"},
  };
  auto vfs = std::make_unique<FakeFilesystem>(
      files, /*cwd=*/std::filesystem::path("/"));
  ImportData import_data = CreateImportDataForTest(std::move(vfs));
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result,
                           TypecheckV2(R"(#![feature(use_syntax)]
use foo;
const X = foo::D;
)",
                                       "fake", &import_data));
  EXPECT_THAT(result, HasTypeInfo(HasNodeWithType("X", "uN[32]")));
}

TEST(TypecheckV2UseTest, UseModuleThenColonRefToFunction) {
  absl::flat_hash_map<std::filesystem::path, std::string> files = {
      {std::filesystem::path("/foo.x"),
       "pub fn bar(x: u32) -> u32 { x + u32:1 }"},
  };
  auto vfs = std::make_unique<FakeFilesystem>(
      files, /*cwd=*/std::filesystem::path("/"));
  ImportData import_data = CreateImportDataForTest(std::move(vfs));
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result,
                           TypecheckV2(R"(#![feature(use_syntax)]
use foo;
fn main() -> u32 { foo::bar(u32:1) }
)",
                                       "fake", &import_data));
  EXPECT_THAT(result, HasTypeInfo(HasNodeWithType("main", "() -> uN[32]")));
}

TEST(TypecheckV2UseTest, UseModuleThenColonRefToStruct) {
  absl::flat_hash_map<std::filesystem::path, std::string> files = {
      {std::filesystem::path("/foo.x"), "pub struct Point { x: u32, y: u32 }"},
  };
  auto vfs = std::make_unique<FakeFilesystem>(
      files, /*cwd=*/std::filesystem::path("/"));
  ImportData import_data = CreateImportDataForTest(std::move(vfs));
  XLS_ASSERT_OK(TypecheckV2(R"(#![feature(use_syntax)]
use foo;
fn main(p: foo::Point) -> foo::Point { p }
)",
                            "fake", &import_data));
}

TEST(TypecheckV2UseTest, UseNestedModuleThenColonRef) {
  absl::flat_hash_map<std::filesystem::path, std::string> files = {
      {std::filesystem::path("/foo/sub.x"), "pub const D = u32:5;"},
  };
  auto vfs = std::make_unique<FakeFilesystem>(
      files, /*cwd=*/std::filesystem::path("/"));
  ImportData import_data = CreateImportDataForTest(std::move(vfs));
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result,
                           TypecheckV2(R"(#![feature(use_syntax)]
use foo::sub;
const X = sub::D;
)",
                                       "fake", &import_data));
  EXPECT_THAT(result, HasTypeInfo(HasNodeWithType("X", "uN[32]")));
}

TEST(TypecheckV2UseTest, UseNestedConstantDirectly) {
  absl::flat_hash_map<std::filesystem::path, std::string> files = {
      {std::filesystem::path("/foo/sub.x"), "pub const D = u32:5;"},
  };
  auto vfs = std::make_unique<FakeFilesystem>(
      files, /*cwd=*/std::filesystem::path("/"));
  ImportData import_data = CreateImportDataForTest(std::move(vfs));
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result,
                           TypecheckV2(R"(#![feature(use_syntax)]
use foo::sub::D;
const X = D;
)",
                                       "fake", &import_data));
  EXPECT_THAT(result, HasTypeInfo(HasNodeWithType("X", "uN[32]")));
}

TEST(TypecheckV2UseTest, UseNestedGroupWithinGroup) {
  absl::flat_hash_map<std::filesystem::path, std::string> files = {
      {std::filesystem::path("/foo.x"), "pub const A = u32:1;"},
      {std::filesystem::path("/foo/sub.x"),
       "pub const B = u32:2;\npub const C = u32:3;"},
  };
  auto vfs = std::make_unique<FakeFilesystem>(
      files, /*cwd=*/std::filesystem::path("/"));
  ImportData import_data = CreateImportDataForTest(std::move(vfs));
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result,
                           TypecheckV2(R"(#![feature(use_syntax)]
use foo::{A, sub::{B, C}};
const X = A + B + C;
)",
                                       "fake", &import_data));
  EXPECT_THAT(result, HasTypeInfo(HasNodeWithType("X", "uN[32]")));
}

TEST(TypecheckV2UseTest, UseStructLiteralFromSpeculativeTypeGuess) {
  absl::flat_hash_map<std::filesystem::path, std::string> files = {
      {std::filesystem::path("/foo.x"), "pub struct Point { x: u32, y: u32 }"},
  };
  auto vfs = std::make_unique<FakeFilesystem>(
      files, /*cwd=*/std::filesystem::path("/"));
  ImportData import_data = CreateImportDataForTest(std::move(vfs));
  XLS_ASSERT_OK_AND_ASSIGN(TypecheckResult result,
                           TypecheckV2(R"(#![feature(use_syntax)]
use foo::Point;
fn main() -> u32 { (Point { x: u32:1, y: u32:2 }).x }
)",
                                       "fake", &import_data));
  EXPECT_THAT(result, HasTypeInfo(HasNodeWithType("main", "() -> uN[32]")));
}

TEST(TypecheckV2UseTest, UseParametricProcSpawnedWithUseConstant) {
  absl::flat_hash_map<std::filesystem::path, std::string> files = {
      {std::filesystem::path("/foo.x"), R"(
pub const WIDTH = u32:16;
pub proc Counter<N: u32> {
    in_r: chan<uN[N]> in;
    out_s: chan<uN[N]> out;
    config(in_r: chan<uN[N]> in, out_s: chan<uN[N]> out) { (in_r, out_s) }
    init { uN[N]:0 }
    next(state: uN[N]) { state }
}
)"},
  };
  auto vfs = std::make_unique<FakeFilesystem>(
      files, /*cwd=*/std::filesystem::path("/"));
  ImportData import_data = CreateImportDataForTest(std::move(vfs));
  XLS_EXPECT_OK(TypecheckV2(R"(#![feature(use_syntax)]
use foo::{Counter, WIDTH};

proc Main {
    init { }
    config() {
        let (in_s, in_r) = chan<u16>("in");
        let (out_s, out_r) = chan<u16>("out");
        spawn Counter<WIDTH>(in_r, out_s);
    }
    next(state: ()) { () }
})",
                            "fake", &import_data)
                    .status());
}

}  // namespace
}  // namespace xls::dslx
