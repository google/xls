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

#include "xls/codegen/vast/translate_vast_to_dslx.h"

#include <cstdint>
#include <filesystem>  // NOLINT
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xls/codegen/vast/vast.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/temp_directory.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/default_dslx_stdlib_path.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/fileno.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/number_parser.h"
#include "xls/ir/source_location.h"

namespace xls {
namespace verilog {
namespace {

// Translates the given `std::unique_ptr<VerilogFile>` objects to one combined
// DSLX file using the function under test.
template <typename... T>
absl::StatusOr<std::string> Translate(T... files) {
  std::vector<std::string> warnings;
  std::vector<std::filesystem::path> paths;
  absl::flat_hash_map<std::filesystem::path,
                      std::unique_ptr<verilog::VerilogFile>>
      paths_to_files;
  int i = 0;
  auto add_file = [&](std::unique_ptr<VerilogFile> file) {
    // The paths are fictitious and not actually used on disk.
    std::filesystem::path path = absl::StrCat("/tmp/file", i++);
    paths.push_back(path);
    paths_to_files.emplace(path, std::move(file));
  };
  (add_file(std::move(files)), ...);
  return TranslateVastToCombinedDslx(
      /*dslx_stdlib_path=*/kDefaultDslxStdlibPath, paths, paths_to_files);
}

// Translates the given `std::unique_ptr<VerilogFile>` objects to multiple DSLX
// files using the function under test.
//
// Note that `out_dir_path` should be a real filesystem directory path where
// the translated files can be written.
template <typename... T>
absl::StatusOr<std::vector<std::string>> TranslateToMultiDslx(
    const std::filesystem::path& out_dir_path, T... files) {
  std::vector<std::string> warnings;
  std::vector<std::filesystem::path> paths;
  std::vector<std::filesystem::path> out_paths;
  absl::flat_hash_map<std::filesystem::path,
                      std::unique_ptr<verilog::VerilogFile>>
      paths_to_files;
  int i = 0;
  auto add_file = [&](std::unique_ptr<VerilogFile> file) {
    // The output paths are real.
    for (auto& member : file->members()) {
      if (std::holds_alternative<verilog::Module*>(member)) {
        out_paths.push_back(
            out_dir_path /
            absl::StrCat(std::get<verilog::Module*>(member)->name(), ".x"));
      }
    }
    // The input paths are fictitious and not actually used on disk.
    std::filesystem::path path = absl::StrCat("/tmp/file", i++);
    paths.push_back(path);
    paths_to_files.emplace(path, std::move(file));
  };
  (add_file(std::move(files)), ...);
  XLS_RETURN_IF_ERROR(TranslateVastToDslx(out_dir_path, kDefaultDslxStdlibPath,
                                          paths, paths_to_files));
  std::vector<std::string> results(out_paths.size());
  for (i = 0; i < results.size(); i++) {
    XLS_ASSIGN_OR_RETURN(results[i], GetFileContents(out_paths[i]));
  }
  return results;
}

// This is a workaround for the fact that all "ok and holds" shortcuts output
// the strings in an unhelpful single-line format on failure. The diff output
// that we want requires a direct "eq" test. The parameters are a
// `VerilogFileHelper` and the expected DSLX that its file translates to.
#define XLS_EXPECT_VAST_TRANSLATION(f, expected)                        \
  XLS_ASSERT_OK_AND_ASSIGN(std::string actual, Translate(f.release())); \
  EXPECT_EQ(actual, expected);

// A wrapper for a VAST `VerilogFile` object that adds useful functions for
// testing.
class VerilogFileHelper {
 public:
  explicit VerilogFileHelper(int file_no)
      : file_no_(file_no),
        file_(std::make_unique<VerilogFile>(FileType::kSystemVerilog)) {}

  // Note that type inference machinery requires VAST nodes to have a distinct
  // `SourceInfo`, so we can't just use a default-constructed `SourceInfo`
  // everywhere, but there doesn't need to be any semblance of accuracy.
  SourceInfo NextLoc() {
    return SourceInfo(
        SourceLocation(Fileno(file_no_), Lineno(next_lineno_++), Colno(0)));
  }

  VerilogFile& file() { return *file_; }

  // Creates a `Literal` object the way it would be created for a decorated
  // literal of any base or signedness in Verilog source, such as "32'shbeef".
  // Note that a decorated literal without a bit count is implicitly 32-bit
  // according to Verilog rules, and to represent that, the 32 must be passed as
  // the `bit_count` here.
  Literal* LiteralWithBitCount(
      int bit_count, int64_t value,
      FormatPreference format_preference = FormatPreference::kUnsignedDecimal,
      bool declared_as_signed = false) {
    absl::StatusOr<Bits> bits = ParseNumber(std::to_string(value));
    QCHECK_OK(bits);
    return file_->Make<Literal>(NextLoc(), *bits, format_preference, bit_count,
                                /*emit_bit_count=*/true, declared_as_signed);
  }

  // Creates a `Literal` object the way it would be created for a plain decimal
  // integer literal encountered in Verilog source. Such literals are implicitly
  // 32-bit signed ints according to Verilog rules.
  Literal* BareLiteral(int32_t value) {
    absl::StatusOr<Bits> bits = ParseNumber(std::to_string(value));
    CHECK_OK(bits);
    return file_->Make<Literal>(NextLoc(),
                                value < 0 ? bits_ops::SignExtend(*bits, 32)
                                          : bits_ops::ZeroExtend(*bits, 32),
                                FormatPreference::kDefault,
                                /*declared_bit_count=*/32,
                                /*emit_bit_count=*/true,
                                /*declared_as_signed=*/true);
  }

  VerilogFile* operator->() { return file_.get(); }

  std::unique_ptr<VerilogFile> release() { return std::move(file_); }

 private:
  const int file_no_;
  int next_lineno_ = 1;
  std::unique_ptr<VerilogFile> file_;
};

class TranslateVastToDslxTest : public ::testing::Test {
 public:
  TranslateVastToDslxTest() {
    absl::SetVLogLevel("translate_vast_to_dslx", 10);
    absl::SetVLogLevel("dslx_builder", 10);
  }

  VerilogFileHelper CreateFile() { return VerilogFileHelper(next_fileno_++); }

  int next_fileno_ = 0;
};

// This test verifies that enums from a 2-chain of SV files can correctly be
// converted to DSLX.
TEST_F(TranslateVastToDslxTest, EnumSmoke) {
  // package a;
  //   typedef enum logic {
  //     kElem0 = 1'b0,
  //     kElem1 = 1'b1
  //   } enum_t;
  //
  //   function automatic integer compute_a(enum_t in_val);
  //     return in_val == kElem0 ? 3 : 4;
  //   endfunction
  // endpackage
  VerilogFileHelper file_a = CreateFile();
  Module* a = file_a->AddModule("a", file_a.NextLoc());
  Enum* a_enum = file_a->Make<Enum>(file_a.NextLoc(), DataKind::kLogic,
                                    file_a->ScalarType(file_a.NextLoc()));
  EnumMemberRef* a_elem_0 = a_enum->AddMember(
      "kElem0", file_a.LiteralWithBitCount(1, 0, FormatPreference::kBinary),
      file_a.NextLoc());
  a_enum->AddMember("kElem1",
                    file_a.LiteralWithBitCount(1, 1, FormatPreference::kBinary),
                    file_a.NextLoc());
  Typedef* a_enum_t = file_a->Make<Typedef>(
      file_a.NextLoc(),
      file_a->Make<Def>(file_a.NextLoc(), "enum_t", DataKind::kUser, a_enum));
  a->AddModuleMember(a_enum_t);
  VerilogFunction* compute_a = file_a->Make<VerilogFunction>(
      file_a.NextLoc(), "compute_a", file_a->IntegerType(file_a.NextLoc()));
  LogicRef* in_val = compute_a->AddArgument(
      file_a->Make<Def>(file_a.NextLoc(), "in_val", DataKind::kUser,
                        file_a->Make<TypedefType>(file_a.NextLoc(), a_enum_t)),
      file_a.NextLoc());
  compute_a->AddStatement<ReturnStatement>(
      file_a.NextLoc(),
      file_a->Ternary(file_a->Equals(in_val, a_elem_0, file_a.NextLoc()),
                      file_a.BareLiteral(3), file_a.BareLiteral(4),
                      file_a.NextLoc()));
  a->AddModuleMember(compute_a);

  // package b;
  //   typedef enum logic[1:0] {
  //     kElem0 = 1'd0,
  //     kElem1 = 2'd1,
  //     kElem2 = 2'd2,
  //     kElem3 = 2'd3
  //   } enum_t;
  //
  //   function automatic integer compute_b(enum_t in_val1, a::enum_t in_val2);
  //     return in_val1 == in_val2 ? 5 : 6;
  //   endfunction
  // endpackage
  VerilogFileHelper file_b = CreateFile();
  Module* b = file_b->AddModule("b", file_b.NextLoc());
  Enum* b_enum = file_b->Make<Enum>(file_b.NextLoc(), DataKind::kLogic,
                                    file_b->BitVectorType(2, file_b.NextLoc()));
  b_enum->AddMember("kElem0", file_b.LiteralWithBitCount(1, 0),
                    file_b.NextLoc());
  b_enum->AddMember("kElem1", file_b.LiteralWithBitCount(2, 1),
                    file_b.NextLoc());
  b_enum->AddMember("kElem2", file_b.LiteralWithBitCount(2, 2),
                    file_b.NextLoc());
  b_enum->AddMember("kElem3", file_b.LiteralWithBitCount(2, 3),
                    file_b.NextLoc());
  Typedef* b_enum_t = file_a->Make<Typedef>(
      file_b.NextLoc(),
      file_b->Make<Def>(file_b.NextLoc(), "enum_t", DataKind::kUser, b_enum));
  b->AddModuleMember(b_enum_t);
  VerilogFunction* compute_b = file_b->Make<VerilogFunction>(
      file_b.NextLoc(), "compute_b", file_b->IntegerType(file_b.NextLoc()));
  LogicRef* in_val1 = compute_b->AddArgument(
      file_b->Make<Def>(file_b.NextLoc(), "in_val1", DataKind::kUser,
                        file_b->Make<TypedefType>(file_b.NextLoc(), b_enum_t)),
      file_b.NextLoc());
  LogicRef* in_val2 = compute_b->AddArgument(
      file_b->Make<Def>(file_b.NextLoc(), "in_val2", DataKind::kUser,
                        file_b->Make<TypedefType>(file_b.NextLoc(), a_enum_t)),
      file_b.NextLoc());
  compute_b->AddStatement<ReturnStatement>(
      file_b.NextLoc(),
      file_b->Ternary(file_b->Equals(in_val1, in_val2, file_b.NextLoc()),
                      file_b.BareLiteral(5), file_b.BareLiteral(6),
                      file_b.NextLoc()));
  b->AddModuleMember(compute_b);

  const std::string kExpected = R"(#[sv_type("a::enum_t")]
pub enum a_enum_t : bits[1] {
    kElem0 = 0b0,
    kElem1 = 0b1,
}

pub fn a_compute_a(in_val: a_enum_t) -> s32 {
    if in_val == a_enum_t::kElem0 { s32:3 } else { s32:4 }
}

#[sv_type("b::enum_t")]
pub enum enum_t : bits[2] {
    kElem0 = 0,
    kElem1 = 1,
    kElem2 = 2,
    kElem3 = 3,
}

pub fn compute_b(in_val1: enum_t, in_val2: a_enum_t) -> s32 {
    if in_val1 as u2 == in_val2 as u2 { s32:5 } else { s32:6 }
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::string actual,
                           Translate(file_a.release(), file_b.release()));
  EXPECT_EQ(actual, kExpected);
}

// Verifies that ranges are interpreted as unsigned.
TEST_F(TranslateVastToDslxTest, RangesUnsigned) {
  // package a;
  //
  // typedef enum logic [4'd15:0] {
  //   kElemA0 = 4'b0,
  //   kElemA1 = 4'b1
  // } a_enum_t;
  //
  // endpackage
  VerilogFileHelper f = CreateFile();
  Module* a = f->AddModule("a", f.NextLoc());
  Enum* a_enum = f->Make<Enum>(
      f.NextLoc(), DataKind::kLogic,
      f->Make<BitVectorType>(f.NextLoc(), f.BareLiteral(15),
                             /*is_signed=*/false, /*size_expr_is_max=*/true));
  a_enum->AddMember("kElemA0",
                    f.LiteralWithBitCount(4, 0, FormatPreference::kBinary),
                    f.NextLoc());
  a_enum->AddMember("kElemA1",
                    f.LiteralWithBitCount(4, 1, FormatPreference::kBinary),
                    f.NextLoc());
  a->AddModuleMember(f->Make<Typedef>(
      f.NextLoc(),
      f->Make<Def>(f.NextLoc(), "a_enum_t", DataKind::kUser, a_enum)));

  const std::string kExpected = R"(#[sv_type("a::a_enum_t")]
pub enum a_enum_t : bits[16] {
    kElemA0 = 0b0,
    kElemA1 = 0b1,
}
)";

  XLS_EXPECT_VAST_TRANSLATION(f, kExpected);
}

// Tests that a simple (non-TypeRef) struct can be translated.
TEST_F(TranslateVastToDslxTest, BasicStruct) {
  // package p;
  //
  // typedef struct packed {
  //   logic [3:0] elem_0;
  //   logic [17:0] elem_1;
  //   logic [5:0] elem_2;
  // } my_struct_t;
  // endpackage
  VerilogFileHelper f = CreateFile();
  Module* p = f->AddModule("p", f.NextLoc());
  std::vector<Def*> members{
      f->Make<Def>(f.NextLoc(), "elem_0", DataKind::kLogic,
                   f->Make<BitVectorType>(f.NextLoc(), f.BareLiteral(3),
                                          /*is_signed=*/false,
                                          /*size_expr_is_max=*/true)),
      f->Make<Def>(f.NextLoc(), "elem_1", DataKind::kLogic,
                   f->Make<BitVectorType>(f.NextLoc(), f.BareLiteral(17),
                                          /*is_signed=*/false,
                                          /*size_expr_is_max=*/true)),
      f->Make<Def>(f.NextLoc(), "elem_2", DataKind::kLogic,
                   f->Make<BitVectorType>(f.NextLoc(), f.BareLiteral(5),
                                          /*is_signed=*/false,
                                          /*size_expr_is_max=*/true))};
  p->AddTypedef(f->Make<Def>(f.NextLoc(), "my_struct_t", DataKind::kUser,
                             f->Make<Struct>(f.NextLoc(), members)),
                f.NextLoc());

  const std::string kExpected = R"(#[sv_type("p::my_struct_t")]
pub struct my_struct_t {
    // 28 bits
    elem_0: bits[4],
    elem_1: bits[18],
    elem_2: bits[6],
}
)";

  XLS_EXPECT_VAST_TRANSLATION(f, kExpected);
}

// Tests that structs containing typerefs can be translated correctly.
TEST_F(TranslateVastToDslxTest, TypeRefStruct) {
  // package p;
  //
  // typedef logic [25:0] my_typedef_t;
  //
  // typedef enum {
  //   kElem0 = 1'b0,
  //   kElem1 = 1'b1
  // } my_enum_t;
  //
  // typedef struct packed {
  //   logic [3:0] elem_0;
  //   my_typedef_t elem_1;
  //   my_enum_t elem_2;
  // } my_struct_t;
  // endpackage
  VerilogFileHelper f = CreateFile();
  Module* p = f->AddModule("p", f.NextLoc());
  Typedef* type_def = p->AddTypedef(
      f->Make<Def>(f.NextLoc(), "my_typedef_t", DataKind::kLogic,
                   f->Make<BitVectorType>(f.NextLoc(), f.BareLiteral(25),
                                          /*is_signed=*/false,
                                          /*size_expr_is_max=*/true)),
      f.NextLoc());
  Enum* enum_def =
      f->Make<Enum>(f.NextLoc(), DataKind::kLogic, f->IntegerType(f.NextLoc()));
  enum_def->AddMember("kElem0",
                      f.LiteralWithBitCount(1, 0, FormatPreference::kBinary),
                      f.NextLoc());
  enum_def->AddMember("kElem1",
                      f.LiteralWithBitCount(1, 1, FormatPreference::kBinary),
                      f.NextLoc());
  Typedef* my_enum_t = p->AddTypedef(
      f->Make<Def>(f.NextLoc(), "my_enum_t", DataKind::kUser, enum_def),
      f.NextLoc());
  std::vector<Def*> members{
      f->Make<Def>(f.NextLoc(), "elem_0", DataKind::kLogic,
                   f->Make<BitVectorType>(f.NextLoc(), f.BareLiteral(3),
                                          /*is_signed=*/false,
                                          /*size_expr_is_max=*/true)),
      f->Make<Def>(f.NextLoc(), "elem_1", DataKind::kUser,
                   f->Make<TypedefType>(f.NextLoc(), type_def)),
      f->Make<Def>(f.NextLoc(), "elem_2", DataKind::kUser,
                   f->Make<TypedefType>(f.NextLoc(), my_enum_t))};
  p->AddTypedef(f->Make<Def>(f.NextLoc(), "my_struct_t", DataKind::kUser,
                             f->Make<Struct>(f.NextLoc(), members)),
                f.NextLoc());

  const std::string kExpected = R"(#[sv_type("p::my_typedef_t")]
pub type my_typedef_t = bits[26];

#[sv_type("p::my_enum_t")]
pub enum my_enum_t : s32 {
    kElem0 = 0b0,
    kElem1 = 0b1,
}

#[sv_type("p::my_struct_t")]
pub struct my_struct_t {
    // 62 bits
    elem_0: bits[4],
    elem_1: my_typedef_t,
    elem_2: my_enum_t,
}
)";

  XLS_EXPECT_VAST_TRANSLATION(f, kExpected);
}

// Verifies that we can convert a simple param into a DSLX constant.
TEST_F(TranslateVastToDslxTest, SimpleParams) {
  // package p;
  //
  // parameter int kParam1 = 2;
  // parameter int kParam2 = 4;
  // parameter integer kParam3 = 4'hf;
  // parameter kParam4 = 4'hf;
  // parameter kParam5 = 33'sh1ffffffff;
  // parameter logic[32:0] kParam6 = 33'sh1ffffffff;
  // parameter logic[99:0] kParam7 = 100'sh1ffffffff;
  // parameter kParam8 = 33'sh1ffffffff;
  // parameter kParam9 = 100'sh1ffffffff;
  // parameter kParam10 = 100'h1ffffffff;
  //
  // endpackage
  VerilogFileHelper f = CreateFile();
  Module* p = f->AddModule("p", f.NextLoc());
  p->AddParameter(f->Make<Def>(f.NextLoc(), "kParam1", DataKind::kInteger,
                               f->IntegerType(f.NextLoc())),
                  f.BareLiteral(2), f.NextLoc());
  p->AddParameter(f->Make<Def>(f.NextLoc(), "kParam2", DataKind::kInteger,
                               f->IntegerType(f.NextLoc())),
                  f.BareLiteral(4), f.NextLoc());
  p->AddParameter(f->Make<Def>(f.NextLoc(), "kParam3", DataKind::kInteger,
                               f->IntegerType(f.NextLoc())),
                  f.LiteralWithBitCount(4, 0xf, FormatPreference::kHex),
                  f.NextLoc());
  p->AddParameter("kParam4",
                  f.LiteralWithBitCount(4, 0xf, FormatPreference::kHex),
                  f.NextLoc());
  p->AddParameter("kParam5",
                  f.LiteralWithBitCount(33, 0x1ffffffff, FormatPreference::kHex,
                                        /*declared_as_signed=*/true),
                  f.NextLoc());
  p->AddParameter(
      f->Make<Def>(f.NextLoc(), "kParam6", DataKind::kLogic,
                   f->Make<BitVectorType>(f.NextLoc(), f.BareLiteral(32),
                                          /*is_signed=*/false,
                                          /*size_expr_is_max=*/true)),
      f.LiteralWithBitCount(33, 0x1ffffffff, FormatPreference::kHex,
                            /*declared_as_signed=*/true),
      f.NextLoc());
  p->AddParameter(
      f->Make<Def>(f.NextLoc(), "kParam7", DataKind::kLogic,
                   f->Make<BitVectorType>(f.NextLoc(), f.BareLiteral(99),
                                          /*is_signed=*/false,
                                          /*size_expr_is_max=*/true)),
      f.LiteralWithBitCount(100, 0x1ffffffff, FormatPreference::kHex,
                            /*declared_as_signed=*/true),
      f.NextLoc());
  p->AddParameter("kParam8",
                  f.LiteralWithBitCount(33, 0x1ffffffff, FormatPreference::kHex,
                                        /*declared_as_signed=*/true),
                  f.NextLoc());
  p->AddParameter(
      "kParam9",
      f.LiteralWithBitCount(100, 0x1ffffffff, FormatPreference::kHex,
                            /*declared_as_signed=*/true),
      f.NextLoc());
  p->AddParameter(
      "kParam10",
      f.LiteralWithBitCount(100, 0x1ffffffff, FormatPreference::kHex),
      f.NextLoc());

  const std::string kExpected = R"(#![allow(nonstandard_constant_naming)]
#![allow(nonstandard_member_naming)]

pub const kParam1 = s32:2;
pub const kParam2 = s32:4;
pub const kParam3 = s32:0xf;
pub const kParam4 = u4:0xf;
pub const kParam5 = s33:0x1_ffff_ffff;
pub const kParam6 = u33:0x1_ffff_ffff;
pub const kParam7 = uN[100]:0x1_ffff_ffff;
pub const kParam8 = s33:0x1_ffff_ffff;
pub const kParam9 = sN[100]:0x1_ffff_ffff;
pub const kParam10 = uN[100]:0x1_ffff_ffff;
)";

  XLS_EXPECT_VAST_TRANSLATION(f, kExpected);
}

TEST_F(TranslateVastToDslxTest, BinOps) {
  // package p;
  //
  // parameter logic a = 16'h32 == 16'h42;
  // parameter logic b = 16'h32 != 16'h42;
  //
  // endpackage
  VerilogFileHelper f = CreateFile();
  Module* p = f->AddModule("p", f.NextLoc());
  p->AddParameter(
      f->Make<Def>(f.NextLoc(), "a", DataKind::kLogic,
                   f->ScalarType(f.NextLoc())),
      f->Equals(f.LiteralWithBitCount(16, 0x32, FormatPreference::kHex),
                f.LiteralWithBitCount(16, 0x42, FormatPreference::kHex),
                f.NextLoc()),
      f.NextLoc());
  p->AddParameter(
      f->Make<Def>(f.NextLoc(), "b", DataKind::kLogic,
                   f->ScalarType(f.NextLoc())),
      f->NotEquals(f.LiteralWithBitCount(16, 0x32, FormatPreference::kHex),
                   f.LiteralWithBitCount(16, 0x42, FormatPreference::kHex),
                   f.NextLoc()),
      f.NextLoc());

  const std::string kExpected = R"(#![allow(nonstandard_constant_naming)]
#![allow(nonstandard_member_naming)]

pub const a = u16:0x32 == u16:0x42;  // bool:0
pub const b = u16:0x32 != u16:0x42;  // bool:1
)";

  XLS_EXPECT_VAST_TRANSLATION(f, kExpected);
}

// Verifies that we can reference typerefs in constant defs.
TEST_F(TranslateVastToDslxTest, ConstantsWithTyperefs) {
  // package p;
  //
  // typedef logic[15:0] foo_t;
  //
  // parameter foo_t kParam1 = 2;
  // parameter foo_t kParam2 = 8'd5;
  // parameter foo_t kParam3 = 16'd5;
  // parameter foo_t kParam4 = kParam3;
  //
  // endpackage
  VerilogFileHelper f = CreateFile();
  Module* p = f->AddModule("p", f.NextLoc());
  Typedef* type_def = p->AddTypedef(
      f->Make<Def>(f.NextLoc(), "foo_t", DataKind::kLogic,
                   f->Make<BitVectorType>(f.NextLoc(), f.BareLiteral(15),
                                          /*is_signed=*/false,
                                          /*size_expr_is_max=*/true)),
      f.NextLoc());
  TypedefType* type_def_type = f->Make<TypedefType>(f.NextLoc(), type_def);
  p->AddParameter(
      f->Make<Def>(f.NextLoc(), "kParam1", DataKind::kUser, type_def_type),
      f.BareLiteral(2), f.NextLoc());
  p->AddParameter(
      f->Make<Def>(f.NextLoc(), "kParam2", DataKind::kUser, type_def_type),
      f.LiteralWithBitCount(8, 5), f.NextLoc());
  ParameterRef* param3 = p->AddParameter(
      f->Make<Def>(f.NextLoc(), "kParam3", DataKind::kUser, type_def_type),
      f.LiteralWithBitCount(16, 5), f.NextLoc());
  p->AddParameter(
      f->Make<Def>(f.NextLoc(), "kParam4", DataKind::kUser, type_def_type),
      param3, f.NextLoc());

  const std::string kExpected = R"(#![allow(nonstandard_constant_naming)]
#![allow(nonstandard_member_naming)]

#[sv_type("p::foo_t")]
pub type foo_t = bits[16];

pub const kParam1 = foo_t:2;
pub const kParam2 = foo_t:5;
pub const kParam3 = foo_t:5;
pub const kParam4 = kParam3;  // u16:5
)";

  XLS_EXPECT_VAST_TRANSLATION(f, kExpected);
}

TEST_F(TranslateVastToDslxTest, AddsPseudoNamespaces) {
  // package a;
  //   typedef logic[15:0] typedef_t;
  // endpackage
  VerilogFileHelper file_a = CreateFile();
  Module* a = file_a->AddModule("a", file_a.NextLoc());
  Typedef* type_def = a->AddTypedef(
      file_a->Make<Def>(file_a.NextLoc(), "typedef_t", DataKind::kLogic,
                        file_a->BitVectorType(16, file_a.NextLoc())),
      file_a.NextLoc());

  // package b;
  //   parameter a::typedef_t my_param = 16'hbeef;
  // endpackage
  VerilogFileHelper file_b = CreateFile();
  Module* b = file_b->AddModule("b", file_b.NextLoc());
  b->AddParameter(
      file_b->Make<Def>(file_b.NextLoc(), "my_param", DataKind::kUser,
                        file_b->Make<TypedefType>(file_b.NextLoc(), type_def)),
      file_b.LiteralWithBitCount(16, 0xbeef, FormatPreference::kHex),
      file_b.NextLoc());

  const std::string kExpected = R"(#![allow(nonstandard_constant_naming)]
#![allow(nonstandard_member_naming)]

#[sv_type("a::typedef_t")]
pub type a_typedef_t = bits[16];

pub const my_param = a_typedef_t:0xbeef;
)";

  XLS_ASSERT_OK_AND_ASSIGN(std::string actual,
                           Translate(file_a.release(), file_b.release()));
  ASSERT_EQ(actual, kExpected);
}

TEST_F(TranslateVastToDslxTest, NoPseudoNamespacesWithMultiFile) {
  // package a;
  //   typedef logic[15:0] typedef_t;
  //   parameter foo = 3;
  // endpackage
  VerilogFileHelper file_a = CreateFile();
  Module* a = file_a->AddModule("a", file_a.NextLoc());
  Typedef* type_def = a->AddTypedef(
      file_a->Make<Def>(file_a.NextLoc(), "typedef_t", DataKind::kLogic,
                        file_a->BitVectorType(16, file_a.NextLoc())),
      file_a.NextLoc());
  ParameterRef* foo =
      a->AddParameter("foo", file_a.BareLiteral(3), file_a.NextLoc());

  // package b;
  //   parameter a::typedef_t my_param = 16'hbeef;
  //   parameter logic[47:0] my_param2 = a::foo + 4;
  // endpackage
  VerilogFileHelper file_b = CreateFile();
  Module* b = file_b->AddModule("b", file_b.NextLoc());
  b->AddParameter(
      file_b->Make<Def>(file_b.NextLoc(), "my_param", DataKind::kUser,
                        file_b->Make<TypedefType>(file_b.NextLoc(), type_def)),
      file_b.LiteralWithBitCount(16, 0xbeef, FormatPreference::kHex),
      file_b.NextLoc());
  b->AddParameter(
      file_b->Make<Def>(
          file_b.NextLoc(), "my_param2", DataKind::kLogic,
          file_b->Make<BitVectorType>(file_b.NextLoc(), file_b.BareLiteral(47),
                                      /*is_signed=*/false,
                                      /*size_expr_is_max=*/true)),
      file_b->Add(foo, file_b.BareLiteral(4), file_b.NextLoc()),
      file_b.NextLoc());

  XLS_ASSERT_OK_AND_ASSIGN(TempDirectory temp_dir, TempDirectory::Create());

  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<std::string> results,
      TranslateToMultiDslx(temp_dir.path(), file_a.release(),
                           file_b.release()));

  const std::string kExpectedA = R"(#![allow(nonstandard_constant_naming)]
#![allow(nonstandard_member_naming)]

#[sv_type("a::typedef_t")]
pub type typedef_t = bits[16];

pub const foo = s32:3;
)";

  const std::string kExpectedB = R"(#![allow(nonstandard_constant_naming)]
#![allow(nonstandard_member_naming)]

import a;

pub const my_param = a::typedef_t:0xbeef;
pub const my_param2 = a::foo as u48 + u48:4;  // u48:7
)";

  EXPECT_THAT(results, testing::ElementsAre(kExpectedA, kExpectedB));
}

// Verifies that _extremely_ simple (single-statement) functions can be
// translated to DSLX.
TEST_F(TranslateVastToDslxTest, HandlesSimpleFunctions) {
  // package p;
  //
  // function automatic logic[31:0] my_called_function(
  //     logic[31:0] a,
  //     logic[31:0] b,
  //     logic[31:0] c);
  //   return (a > b) ? a : ((b < c) ? b : c);
  // endfunction
  //
  // function automatic logic[31:0] my_calling_function();
  //   return my_called_function(32'hdead, 32'hbeef, 32'hf00d);
  // endfunction
  //
  // endpackage
  VerilogFileHelper f = CreateFile();
  Module* p = f->AddModule("p", f.NextLoc());
  VerilogFunction* my_called_function = f->Make<VerilogFunction>(
      f.NextLoc(), "my_called_function",
      f->Make<BitVectorType>(f.NextLoc(), f.BareLiteral(31),
                             /*is_signed=*/false, /*size_expr_is_max=*/true));
  LogicRef* a = my_called_function->AddArgument(
      f->Make<Def>(f.NextLoc(), "a", DataKind::kLogic,
                   f->Make<BitVectorType>(f.NextLoc(), f.BareLiteral(31),
                                          /*is_signed=*/false,
                                          /*size_expr_is_max=*/true)),
      f.NextLoc());
  LogicRef* b = my_called_function->AddArgument(
      f->Make<Def>(f.NextLoc(), "b", DataKind::kLogic,
                   f->Make<BitVectorType>(f.NextLoc(), f.BareLiteral(31),
                                          /*is_signed=*/false,
                                          /*size_expr_is_max=*/true)),
      f.NextLoc());
  LogicRef* c = my_called_function->AddArgument(
      f->Make<Def>(f.NextLoc(), "c", DataKind::kLogic,
                   f->Make<BitVectorType>(f.NextLoc(), f.BareLiteral(31),
                                          /*is_signed=*/false,
                                          /*size_expr_is_max=*/true)),
      f.NextLoc());
  my_called_function->AddStatement<ReturnStatement>(
      f.NextLoc(),
      f->Ternary(f->GreaterThan(a, b, f.NextLoc()), a,
                 f->Ternary(f->LessThan(b, c, f.NextLoc()), b, c, f.NextLoc()),
                 f.NextLoc()));
  p->AddModuleMember(my_called_function);
  VerilogFunction* my_calling_function = f->Make<VerilogFunction>(
      f.NextLoc(), "my_calling_function",
      f->Make<BitVectorType>(f.NextLoc(), f.BareLiteral(31),
                             /*is_signed=*/false, /*size_expr_is_max=*/true));
  std::vector<Expression*> args{
      f.LiteralWithBitCount(32, 0xdead, FormatPreference::kHex),
      f.LiteralWithBitCount(32, 0xbeef, FormatPreference::kHex),
      f.LiteralWithBitCount(32, 0xf00d, FormatPreference::kHex)};
  my_calling_function->AddStatement<ReturnStatement>(
      f.NextLoc(),
      f->Make<VerilogFunctionCall>(f.NextLoc(), my_called_function, args));
  p->AddModuleMember(my_calling_function);

  const std::string kExpected =
      R"(pub fn my_called_function(a: bits[32], b: bits[32], c: bits[32]) -> bits[32] {
    if a > b { a } else { if b < c { b } else { c } }
}

pub fn my_calling_function() -> bits[32] { my_called_function(u32:0xdead, u32:0xbeef, u32:0xf00d) }
)";
  XLS_EXPECT_VAST_TRANSLATION(f, kExpected);
}

// Verifies that the SV system function $clog2 translates correctly.
TEST_F(TranslateVastToDslxTest, Clog2) {
  // package p;
  //
  // parameter logic[$clog2(32767):0] var_1 = 32'hdead;
  // parameter logic[$clog2(32767):0] var_2 = 32'hbeef;
  // parameter integer var_3 = $clog2(1024);
  // parameter integer var_4 = $clog2(1025);
  // parameter var_5 = $clog2(2048);
  //
  // parameter logic[$clog2(32767):0] var_6 = 32'shbeef;
  // parameter logic[$clog2(32767):0] var_7 = -32'shbeef;
  //
  // endpackage
  VerilogFileHelper f = CreateFile();
  Module* p = f->AddModule("p", f.NextLoc());
  p->AddParameter(
      f->Make<Def>(f.NextLoc(), "var_1", DataKind::kLogic,
                   f->Make<BitVectorType>(
                       f.NextLoc(),
                       f->Make<SystemFunctionCall>(
                           f.NextLoc(), "clog2",
                           std::vector<Expression*>{f.BareLiteral(32767)}),
                       /*is_signed=*/false, /*size_expr_is_max=*/true)),
      f.LiteralWithBitCount(32, 0xdead, FormatPreference::kHex), f.NextLoc());
  p->AddParameter(
      f->Make<Def>(f.NextLoc(), "var_2", DataKind::kLogic,
                   f->Make<BitVectorType>(
                       f.NextLoc(),
                       f->Make<SystemFunctionCall>(
                           f.NextLoc(), "clog2",
                           std::vector<Expression*>{f.BareLiteral(32767)}),
                       /*is_signed=*/false, /*size_expr_is_max=*/true)),
      f.LiteralWithBitCount(32, 0xbeef, FormatPreference::kHex), f.NextLoc());
  p->AddParameter(
      f->Make<Def>(f.NextLoc(), "var_3", DataKind::kLogic,
                   f->IntegerType(f.NextLoc())),
      f->Make<SystemFunctionCall>(
          f.NextLoc(), "clog2", std::vector<Expression*>{f.BareLiteral(1024)}),
      f.NextLoc());
  p->AddParameter(
      f->Make<Def>(f.NextLoc(), "var_4", DataKind::kLogic,
                   f->IntegerType(f.NextLoc())),
      f->Make<SystemFunctionCall>(
          f.NextLoc(), "clog2", std::vector<Expression*>{f.BareLiteral(1025)}),
      f.NextLoc());
  p->AddParameter(
      "var_5",
      f->Make<SystemFunctionCall>(
          f.NextLoc(), "clog2", std::vector<Expression*>{f.BareLiteral(2048)}),
      f.NextLoc());
  p->AddParameter(
      f->Make<Def>(f.NextLoc(), "var_6", DataKind::kLogic,
                   f->Make<BitVectorType>(
                       f.NextLoc(),
                       f->Make<SystemFunctionCall>(
                           f.NextLoc(), "clog2",
                           std::vector<Expression*>{f.BareLiteral(32767)}),
                       /*is_signed=*/false, /*size_expr_is_max=*/true)),
      f.LiteralWithBitCount(32, 0xbeef, FormatPreference::kHex,
                            /*declared_as_signed=*/true),
      f.NextLoc());
  p->AddParameter(
      f->Make<Def>(f.NextLoc(), "var_7", DataKind::kLogic,
                   f->Make<BitVectorType>(
                       f.NextLoc(),
                       f->Make<SystemFunctionCall>(
                           f.NextLoc(), "clog2",
                           std::vector<Expression*>{f.BareLiteral(32767)}),
                       /*is_signed=*/false, /*size_expr_is_max=*/true)),
      f->Negate(f.LiteralWithBitCount(32, 0xbeef, FormatPreference::kHex,
                                      /*declared_as_signed=*/true),
                f.NextLoc()),
      f.NextLoc());

  const std::string kExpected = R"(#![allow(nonstandard_constant_naming)]
#![allow(nonstandard_member_naming)]

pub const var_1 = u16:0xdead;
pub const var_2 = u16:0xbeef;

import std;

pub const var_3 = std::clog2(s32:1024 as uN[32]) as s32;  // s32:10
pub const var_4 = std::clog2(s32:1025 as uN[32]) as s32;  // s32:11
pub const var_5 = std::clog2(s32:2048 as uN[32]) as s32;  // s32:11
pub const var_6 = u16:0xbeef;
pub const var_7 = -s32:0xbeef as u16;
)";

  XLS_EXPECT_VAST_TRANSLATION(f, kExpected);
}

// Tests conversion of integer exponentiation.
TEST_F(TranslateVastToDslxTest, UPowSpow) {
  // package p;
  //
  // parameter upow_lhs = 16'h8003;
  // parameter spow_lhs = 16'sh8013;
  // parameter power = 16'h3;
  //
  // parameter logic[15:0] upow_result = upow_lhs ** power;
  // parameter logic[15:0] spow_result = spow_lhs ** power;
  // endpackage

  VerilogFileHelper f = CreateFile();
  Module* p = f->AddModule("p", f.NextLoc());
  ParameterRef* upow_lhs = p->AddParameter(
      "upow_lhs", f.LiteralWithBitCount(16, 0x8003, FormatPreference::kHex),
      f.NextLoc());
  ParameterRef* spow_lhs =
      p->AddParameter("spow_lhs",
                      f.LiteralWithBitCount(16, 0x8013, FormatPreference::kHex,
                                            /*declared_as_signed=*/true),
                      f.NextLoc());
  ParameterRef* power = p->AddParameter(
      "power", f.LiteralWithBitCount(16, 3, FormatPreference::kHex),
      f.NextLoc());
  p->AddParameter(
      f->Make<Def>(f.NextLoc(), "upow_result", DataKind::kLogic,
                   f->Make<BitVectorType>(f.NextLoc(), f.BareLiteral(15),
                                          /*is_signed=*/false,
                                          /*size_expr_is_max=*/true)),
      f->Power(upow_lhs, power, f.NextLoc()), f.NextLoc());
  p->AddParameter(
      f->Make<Def>(f.NextLoc(), "spow_result", DataKind::kLogic,
                   f->Make<BitVectorType>(f.NextLoc(), f.BareLiteral(15),
                                          /*is_signed=*/false,
                                          /*size_expr_is_max=*/true)),
      f->Power(spow_lhs, power, f.NextLoc()), f.NextLoc());

  const std::string kExpected = R"(#![allow(nonstandard_constant_naming)]
#![allow(nonstandard_member_naming)]

pub const upow_lhs = u16:0x8003;
pub const spow_lhs = s16:0x8013;
pub const power = u16:0x3;

import std;

pub const upow_result = std::upow(upow_lhs, power as uN[16]);
pub const spow_result = std::spow(spow_lhs, power as uN[16]) as u16;
)";

  XLS_EXPECT_VAST_TRANSLATION(f, kExpected);
}

TEST_F(TranslateVastToDslxTest, NonBitsEnum) {
  // package a;
  //   typedef enum int {
  //     kElem0 = 32'd0,
  //     kElem1 = 32'd1,
  //     kElem2 = 32'd2,
  //     kElem3 = 32'd3
  //   } enum_t;
  // endpackage
  VerilogFileHelper f = CreateFile();
  Module* a = f->AddModule("a", f.NextLoc());
  Enum* enum_def = f->Make<Enum>(f.NextLoc(), DataKind::kInteger,
                                 f->IntegerType(f.NextLoc()));
  enum_def->AddMember("kElem0", f.LiteralWithBitCount(32, 0), f.NextLoc());
  enum_def->AddMember("kElem1", f.LiteralWithBitCount(32, 1), f.NextLoc());
  enum_def->AddMember("kElem2", f.LiteralWithBitCount(32, 2), f.NextLoc());
  enum_def->AddMember("kElem3", f.LiteralWithBitCount(32, 3), f.NextLoc());
  a->AddTypedef(f->Make<Def>(f.NextLoc(), "enum_t", DataKind::kUser, enum_def),
                f.NextLoc());
  constexpr std::string_view kExpected = R"(#[sv_type("a::enum_t")]
pub enum enum_t : s32 {
    kElem0 = 0,
    kElem1 = 1,
    kElem2 = 2,
    kElem3 = 3,
}
)";

  XLS_EXPECT_VAST_TRANSLATION(f, kExpected);
}

TEST_F(TranslateVastToDslxTest, Concat) {
  // package a;
  // typedef enum logic [13:0] {
  //   kValue0   = {6'd0, 8'd0},
  //   kValue1   = {6'd0, 8'd1},
  //   kValue2   = {6'd0, 8'd2},
  //   kValue3   = {6'd0, 8'd3},
  //   kValue256 = {6'd1, 8'd0},
  //   kValue257 = {6'd1, 8'd1},
  //   kValue512 = {6'd2, 8'd0},
  //   kValue513 = {6'd2, 8'd1},
  //   kValueIdk = {2'd2, 3'd1, 4'd12, 5'd15}
  // } my_enum_t;
  // endpackage
  VerilogFileHelper f = CreateFile();
  Module* a = f->AddModule("a", f.NextLoc());
  Enum* enum_def = f->Make<Enum>(
      f.NextLoc(), DataKind::kLogic,
      f->Make<BitVectorType>(f.NextLoc(), f.BareLiteral(13),
                             /*is_signed=*/false, /*size_expr_is_max=*/true));
  enum_def->AddMember(
      "kValue0",
      f->Concat(std::vector<Expression*>{f.LiteralWithBitCount(6, 0),
                                         f.LiteralWithBitCount(8, 0)},
                f.NextLoc()),
      f.NextLoc());
  enum_def->AddMember(
      "kValue1",
      f->Concat(std::vector<Expression*>{f.LiteralWithBitCount(6, 0),
                                         f.LiteralWithBitCount(8, 1)},
                f.NextLoc()),
      f.NextLoc());
  enum_def->AddMember(
      "kValue2",
      f->Concat(std::vector<Expression*>{f.LiteralWithBitCount(6, 0),
                                         f.LiteralWithBitCount(8, 2)},
                f.NextLoc()),
      f.NextLoc());
  enum_def->AddMember(
      "kValue3",
      f->Concat(std::vector<Expression*>{f.LiteralWithBitCount(6, 0),
                                         f.LiteralWithBitCount(8, 3)},
                f.NextLoc()),
      f.NextLoc());
  enum_def->AddMember(
      "kValue256",
      f->Concat(std::vector<Expression*>{f.LiteralWithBitCount(6, 1),
                                         f.LiteralWithBitCount(8, 0)},
                f.NextLoc()),
      f.NextLoc());
  enum_def->AddMember(
      "kValue257",
      f->Concat(std::vector<Expression*>{f.LiteralWithBitCount(6, 1),
                                         f.LiteralWithBitCount(8, 1)},
                f.NextLoc()),
      f.NextLoc());
  enum_def->AddMember(
      "kValue512",
      f->Concat(std::vector<Expression*>{f.LiteralWithBitCount(6, 2),
                                         f.LiteralWithBitCount(8, 0)},
                f.NextLoc()),
      f.NextLoc());
  enum_def->AddMember(
      "kValue513",
      f->Concat(std::vector<Expression*>{f.LiteralWithBitCount(6, 2),
                                         f.LiteralWithBitCount(8, 1)},
                f.NextLoc()),
      f.NextLoc());
  enum_def->AddMember(
      "kValueIdk",
      f->Concat(std::vector<Expression*>{f.LiteralWithBitCount(2, 2),
                                         f.LiteralWithBitCount(3, 1),
                                         f.LiteralWithBitCount(4, 12),
                                         f.LiteralWithBitCount(5, 15)},
                f.NextLoc()),
      f.NextLoc());
  a->AddTypedef(
      f->Make<Def>(f.NextLoc(), "my_enum_t", DataKind::kUser, enum_def),
      f.NextLoc());

  constexpr std::string_view kExpected = R"(#[sv_type("a::my_enum_t")]
pub enum my_enum_t : bits[14] {
    kValue0 = u6:0 ++ u8:0,
    kValue1 = u6:0 ++ u8:1,
    kValue2 = u6:0 ++ u8:2,
    kValue3 = u6:0 ++ u8:3,
    kValue256 = u6:1 ++ u8:0,
    kValue257 = u6:1 ++ u8:1,
    kValue512 = u6:2 ++ u8:0,
    kValue513 = u6:2 ++ u8:1,
    kValueIdk = u2:2 ++ u3:1 ++ u4:12 ++ u5:15,
}
)";

  XLS_EXPECT_VAST_TRANSLATION(f, kExpected);
}

TEST_F(TranslateVastToDslxTest, ConcatSigned) {
  // package a;
  // typedef enum int {
  //   kValue0   = {-16'd1, 16'd0},
  //   kValue1   = {16'd0, -16'd1},
  //   kValue2   = {-16'd0, 16'd2},
  //   kValue3   = {16'd0, -16'd3},
  //   kValue256 = {-16'd1, 16'd0},
  //   kValue257 = {16'd1, -16'd1},
  //   kValue512 = {-16'd2, 16'd0},
  //   kValue513 = {16'd2, -16'd1}
  // } my_enum_t;
  // endpackage
  VerilogFileHelper f = CreateFile();
  Module* a = f->AddModule("a", f.NextLoc());
  Enum* enum_def = f->Make<Enum>(f.NextLoc(), DataKind::kInteger,
                                 f->IntegerType(f.NextLoc()));
  enum_def->AddMember(
      "kValue0",
      f->Concat(std::vector<Expression*>{f->Negate(f.LiteralWithBitCount(16, 1),
                                                   f.NextLoc()),
                                         f.LiteralWithBitCount(16, 0)},
                f.NextLoc()),
      f.NextLoc());
  enum_def->AddMember(
      "kValue1",
      f->Concat(std::vector<Expression*>{f.LiteralWithBitCount(16, 0),
                                         f->Negate(f.LiteralWithBitCount(16, 1),
                                                   f.NextLoc())},
                f.NextLoc()),
      f.NextLoc());
  enum_def->AddMember(
      "kValue2",
      f->Concat(std::vector<Expression*>{f->Negate(f.LiteralWithBitCount(16, 0),
                                                   f.NextLoc()),
                                         f.LiteralWithBitCount(16, 2)},
                f.NextLoc()),
      f.NextLoc());
  enum_def->AddMember(
      "kValue3",
      f->Concat(std::vector<Expression*>{f.LiteralWithBitCount(16, 0),
                                         f->Negate(f.LiteralWithBitCount(16, 3),
                                                   f.NextLoc())},
                f.NextLoc()),
      f.NextLoc());
  enum_def->AddMember(
      "kValue256",
      f->Concat(std::vector<Expression*>{f->Negate(f.LiteralWithBitCount(16, 1),
                                                   f.NextLoc()),
                                         f.LiteralWithBitCount(16, 0)},
                f.NextLoc()),
      f.NextLoc());
  enum_def->AddMember(
      "kValue257",
      f->Concat(std::vector<Expression*>{f.LiteralWithBitCount(16, 1),
                                         f->Negate(f.LiteralWithBitCount(16, 1),
                                                   f.NextLoc())},
                f.NextLoc()),
      f.NextLoc());
  enum_def->AddMember(
      "kValue512",
      f->Concat(std::vector<Expression*>{f->Negate(f.LiteralWithBitCount(16, 2),
                                                   f.NextLoc()),
                                         f.LiteralWithBitCount(16, 0)},
                f.NextLoc()),
      f.NextLoc());
  enum_def->AddMember(
      "kValue513",
      f->Concat(std::vector<Expression*>{f.LiteralWithBitCount(16, 2),
                                         f->Negate(f.LiteralWithBitCount(16, 1),
                                                   f.NextLoc())},
                f.NextLoc()),
      f.NextLoc());
  a->AddTypedef(
      f->Make<Def>(f.NextLoc(), "my_enum_t", DataKind::kUser, enum_def),
      f.NextLoc());

  constexpr std::string_view kExpected = R"(#[sv_type("a::my_enum_t")]
pub enum my_enum_t : s32 {
    kValue0 = (-u16:1 ++ u16:0) as s32,
    kValue1 = (u16:0 ++ -u16:1) as s32,
    kValue2 = (-u16:0 ++ u16:2) as s32,
    kValue3 = (u16:0 ++ -u16:3) as s32,
    kValue256 = (-u16:1 ++ u16:0) as s32,
    kValue257 = (u16:1 ++ -u16:1) as s32,
    kValue512 = (-u16:2 ++ u16:0) as s32,
    kValue513 = (u16:2 ++ -u16:1) as s32,
}
)";

  XLS_EXPECT_VAST_TRANSLATION(f, kExpected);
}

TEST_F(TranslateVastToDslxTest, ArrayOfStructs) {
  // package a;
  // typedef struct packed {
  //   logic [3:0] a;
  //   logic [7:0] b;
  // } foo_t;
  //
  // typedef struct packed {
  //   foo_t [1:0] c;
  // } bar_t;
  // endpackage

  VerilogFileHelper f = CreateFile();
  Module* a = f->AddModule("a", f.NextLoc());
  Typedef* foo_t = a->AddTypedef(
      f->Make<Def>(
          f.NextLoc(), "foo_t", DataKind::kUser,
          f->Make<Struct>(
              f.NextLoc(),
              std::vector<Def*>{f->Make<Def>(f.NextLoc(), "a", DataKind::kLogic,
                                             f->Make<BitVectorType>(
                                                 f.NextLoc(), f.BareLiteral(3),
                                                 /*is_signed=*/false,
                                                 /*size_expr_is_max=*/true)),
                                f->Make<Def>(f.NextLoc(), "b", DataKind::kLogic,
                                             f->Make<BitVectorType>(
                                                 f.NextLoc(), f.BareLiteral(7),
                                                 /*is_signed=*/false,
                                                 /*size_expr_is_max=*/true))})),
      f.NextLoc());
  a->AddTypedef(
      f->Make<Def>(
          f.NextLoc(), "bar_t", DataKind::kUser,
          f->Make<Struct>(
              f.NextLoc(),
              std::vector<Def*>{f->Make<Def>(
                  f.NextLoc(), "c", DataKind::kUser,
                  f->Make<UnpackedArrayType>(
                      f.NextLoc(), f->Make<TypedefType>(f.NextLoc(), foo_t),
                      std::vector<Expression*>{f.BareLiteral(2)}))})),
      f.NextLoc());

  constexpr std::string_view kExpected = R"(#[sv_type("a::foo_t")]
pub struct foo_t {
    // 12 bits
    a: bits[4],
    b: bits[8],
}

#[sv_type("a::bar_t")]
pub struct bar_t { c: foo_t[2] }
)";

  XLS_EXPECT_VAST_TRANSLATION(f, kExpected);
}

TEST_F(TranslateVastToDslxTest, PackedMultiDimArray) {
  // package a;
  // typedef logic [25:0] my_typedef_t;
  //
  // typedef struct packed {
  //   logic [8:0] [3:0] a;
  //   my_typedef_t [1:0] [7:0] b;
  // } foo_t;
  // endpackage
  VerilogFileHelper f = CreateFile();
  Module* a = f->AddModule("a", f.NextLoc());
  Typedef* my_typedef_t = a->AddTypedef(
      f->Make<Def>(f.NextLoc(), "my_typedef_t", DataKind::kLogic,
                   f->Make<BitVectorType>(f.NextLoc(), f.BareLiteral(25),
                                          /*is_signed=*/false,
                                          /*size_expr_is_max=*/true)),
      f.NextLoc());
  a->AddTypedef(
      f->Make<Def>(
          f.NextLoc(), "foo_t", DataKind::kUser,
          f->Make<Struct>(
              f.NextLoc(),
              std::vector<Def*>{
                  f->Make<Def>(
                      f.NextLoc(), "a", DataKind::kLogic,
                      f->Make<PackedArrayType>(
                          f.NextLoc(),
                          f->Make<BitVectorType>(
                              f.NextLoc(), f.BareLiteral(8),
                              /*is_signed=*/false, /*size_expr_is_max=*/true),
                          std::vector<Expression*>{f.BareLiteral(3)},
                          /*dims_are_max=*/true)),
                  f->Make<Def>(
                      f.NextLoc(), "b", DataKind::kUser,
                      f->Make<PackedArrayType>(
                          f.NextLoc(),
                          f->Make<TypedefType>(f.NextLoc(), my_typedef_t),
                          std::vector<Expression*>{f.BareLiteral(1),
                                                   f.BareLiteral(7)},
                          /*dims_are_max=*/true))})),
      f.NextLoc());

  constexpr std::string_view kExpected = R"(#[sv_type("a::my_typedef_t")]
pub type my_typedef_t = bits[26];

#[sv_type("a::foo_t")]
pub struct foo_t {
    // 452 bits
    a: bits[4][9],
    b: my_typedef_t[8][2],
}
)";

  XLS_EXPECT_VAST_TRANSLATION(f, kExpected);
}

TEST_F(TranslateVastToDslxTest, ConcatOfEnums) {
  // package a;
  //   typedef enum logic[1:0] {
  //     kElem0 = 2'd0,
  //     kElem1 = 2'd1
  //   } enum_t;
  //
  //   parameter logic[4:0] p = {kElem0, kElem1};
  // endpackage
  VerilogFileHelper f = CreateFile();
  Module* a = f->AddModule("a", f.NextLoc());
  Enum* enum_def = f->Make<Enum>(
      f.NextLoc(), DataKind::kLogic,
      f->Make<BitVectorType>(f.NextLoc(), f.BareLiteral(1), /*is_signed=*/false,
                             /*size_expr_is_max=*/true));
  EnumMemberRef* elem0 =
      enum_def->AddMember("kElem0", f.LiteralWithBitCount(2, 0), f.NextLoc());
  EnumMemberRef* elem1 =
      enum_def->AddMember("kElem1", f.LiteralWithBitCount(2, 1), f.NextLoc());
  a->AddTypedef(f->Make<Def>(f.NextLoc(), "enum_t", DataKind::kUser, enum_def),
                f.NextLoc());
  a->AddParameter(
      f->Make<Def>(f.NextLoc(), "p", DataKind::kLogic,
                   f->Make<BitVectorType>(f.NextLoc(), f.BareLiteral(4),
                                          /*is_signed=*/false,
                                          /*size_expr_is_max=*/true)),
      f->Concat(std::vector<Expression*>{elem0, elem1}, f.NextLoc()),
      f.NextLoc());

  constexpr std::string_view kExpected =
      R"(#![allow(nonstandard_constant_naming)]
#![allow(nonstandard_member_naming)]

#[sv_type("a::enum_t")]
pub enum enum_t : bits[2] {
    kElem0 = 0,
    kElem1 = 1,
}

pub const p = enum_t::kElem0 as u2 ++ enum_t::kElem1 as u2;
)";

  XLS_EXPECT_VAST_TRANSLATION(f, kExpected);
}

TEST_F(TranslateVastToDslxTest, ConcatOfTypedefs) {
  // package a;
  //   typedef logic[1:0] foo_t;
  //
  //   function automatic logic[3:0] concatenate(foo_t a, foo_t b);
  //     return {a, b};
  //   endfunction : concatenate
  // endpackage
  VerilogFileHelper f = CreateFile();
  Module* a = f->AddModule("a", f.NextLoc());
  Typedef* foo_t = a->AddTypedef(
      f->Make<Def>(f.NextLoc(), "foo_t", DataKind::kLogic,
                   f->Make<BitVectorType>(f.NextLoc(), f.BareLiteral(1),
                                          /*is_signed=*/false,
                                          /*size_expr_is_max=*/true)),
      f.NextLoc());
  VerilogFunction* fn = f->Make<VerilogFunction>(
      f.NextLoc(), "concatenate",
      f->Make<BitVectorType>(f.NextLoc(), f.BareLiteral(3), /*is_signed=*/false,
                             /*size_expr_is_max=*/true));
  LogicRef* arg_a =
      fn->AddArgument(f->Make<Def>(f.NextLoc(), "a", DataKind::kUser,
                                   f->Make<TypedefType>(f.NextLoc(), foo_t)),
                      f.NextLoc());
  LogicRef* arg_b =
      fn->AddArgument(f->Make<Def>(f.NextLoc(), "b", DataKind::kUser,
                                   f->Make<TypedefType>(f.NextLoc(), foo_t)),
                      f.NextLoc());
  fn->AddStatement<ReturnStatement>(
      f.NextLoc(),
      f->Concat(std::vector<Expression*>{arg_a, arg_b}, f.NextLoc()));
  a->AddModuleMember(fn);

  constexpr std::string_view kExpected =
      R"(#[sv_type("a::foo_t")]
pub type foo_t = bits[2];

pub fn concatenate(a: foo_t, b: foo_t) -> bits[4] { a ++ b }
)";

  XLS_EXPECT_VAST_TRANSLATION(f, kExpected);
}

TEST_F(TranslateVastToDslxTest, BigConstants) {
  // package foo;
  //   parameter int unsigned GiB = 1024 * 1024 * 1024;
  //   parameter logic [63:0] hundredGiB = 100 * GiB;
  //   parameter logic [63:0] other_hundredGiB = 64'h1900000000;
  // endpackage : foo
  VerilogFileHelper f = CreateFile();
  Module* foo = f->AddModule("foo", f.NextLoc());
  ParameterRef* gib = foo->AddParameter(
      f->Make<Def>(f.NextLoc(), "GiB", DataKind::kInteger,
                   f->Make<IntegerType>(f.NextLoc(), /*is_signed=*/false)),
      f->Mul(f.BareLiteral(1024),
             f->Mul(f.BareLiteral(1024), f.BareLiteral(1024), f.NextLoc()),
             f.NextLoc()),
      f.NextLoc());
  foo->AddParameter(
      f->Make<Def>(f.NextLoc(), "hundredGiB", DataKind::kLogic,
                   f->Make<BitVectorType>(f.NextLoc(), f.BareLiteral(63),
                                          /*is_signed=*/false,
                                          /*size_expr_is_max=*/true)),
      f->Mul(f.BareLiteral(100), gib, f.NextLoc()), f.NextLoc());
  foo->AddParameter(
      f->Make<Def>(f.NextLoc(), "other_hundredGiB", DataKind::kLogic,
                   f->Make<BitVectorType>(f.NextLoc(), f.BareLiteral(63),
                                          /*is_signed=*/false,
                                          /*size_expr_is_max=*/true)),
      f.LiteralWithBitCount(64, 0x1900000000, FormatPreference::kHex),
      f.NextLoc());

  constexpr std::string_view kExpected =
      R"(#![allow(nonstandard_constant_naming)]
#![allow(nonstandard_member_naming)]

pub const GiB = (s32:1024 * s32:1024 * s32:1024) as u32;  // u32:0x40000000
pub const hundredGiB = u64:100 * GiB as u64;  // u64:0x1900000000
pub const other_hundredGiB = u64:0x19_0000_0000;
)";

  XLS_EXPECT_VAST_TRANSLATION(f, kExpected);
}

TEST_F(TranslateVastToDslxTest, SignedConstants) {
  // package foo;
  //   parameter int unsigned A = 100;
  //   parameter int unsigned B = -1;
  //   parameter int unsigned C = 103;
  //   parameter int signed D = 104;
  //   parameter int signed E = 'sd105;
  //   parameter int unsigned F = 'sd106;
  //   parameter int signed G = 'd107;
  // endpackage : foo
  VerilogFileHelper f = CreateFile();
  Module* foo = f->AddModule("foo", f.NextLoc());
  foo->AddParameter(
      f->Make<Def>(f.NextLoc(), "A", DataKind::kInteger,
                   f->Make<IntegerType>(f.NextLoc(), /*is_signed=*/false)),
      f.BareLiteral(100), f.NextLoc());
  foo->AddParameter(
      f->Make<Def>(f.NextLoc(), "B", DataKind::kInteger,
                   f->Make<IntegerType>(f.NextLoc(), /*is_signed=*/false)),
      f->Negate(f.BareLiteral(1), f.NextLoc()), f.NextLoc());
  foo->AddParameter(
      f->Make<Def>(f.NextLoc(), "C", DataKind::kInteger,
                   f->Make<IntegerType>(f.NextLoc(), /*is_signed=*/false)),
      f.BareLiteral(103), f.NextLoc());
  foo->AddParameter(
      f->Make<Def>(f.NextLoc(), "D", DataKind::kInteger,
                   f->Make<IntegerType>(f.NextLoc(), /*is_signed=*/true)),
      f.BareLiteral(104), f.NextLoc());
  foo->AddParameter(
      f->Make<Def>(f.NextLoc(), "E", DataKind::kInteger,
                   f->Make<IntegerType>(f.NextLoc(), /*is_signed=*/true)),
      f.LiteralWithBitCount(32, 105, FormatPreference::kUnsignedDecimal,
                            /*declared_as_signed=*/true),
      f.NextLoc());
  foo->AddParameter(
      f->Make<Def>(f.NextLoc(), "F", DataKind::kInteger,
                   f->Make<IntegerType>(f.NextLoc(), /*is_signed=*/false)),
      f.LiteralWithBitCount(32, 106, FormatPreference::kUnsignedDecimal,
                            /*declared_as_signed=*/true),
      f.NextLoc());
  foo->AddParameter(
      f->Make<Def>(f.NextLoc(), "G", DataKind::kInteger,
                   f->Make<IntegerType>(f.NextLoc(), /*is_signed=*/true)),
      f.LiteralWithBitCount(32, 107, FormatPreference::kUnsignedDecimal,
                            /*declared_as_signed=*/false),
      f.NextLoc());

  constexpr std::string_view kExpected =
      R"(#![allow(nonstandard_constant_naming)]
#![allow(nonstandard_member_naming)]

pub const A = u32:100;
pub const B = -s32:1 as u32;
pub const C = u32:103;
pub const D = s32:104;
pub const E = s32:105;
pub const F = u32:106;
pub const G = s32:107;
)";

  XLS_EXPECT_VAST_TRANSLATION(f, kExpected);
}

TEST_F(TranslateVastToDslxTest, TypedefReturnType) {
  // package foo;
  //   parameter width = 24;
  //   typedef logic[width - 1:0] word_t;
  //   function automatic word_t func(word_t a);
  //     return a + 3'd0;
  //   endfunction : func
  // endpackage : foo
  VerilogFileHelper f = CreateFile();
  Module* foo = f->AddModule("foo", f.NextLoc());
  ParameterRef* width =
      foo->AddParameter("width", f.BareLiteral(24), f.NextLoc());
  Typedef* word_t = foo->AddTypedef(
      f->Make<Def>(
          f.NextLoc(), "word_t", DataKind::kLogic,
          f->Make<BitVectorType>(
              f.NextLoc(), f->Sub(width, f.BareLiteral(1), f.NextLoc()),
              /*is_signed=*/false, /*size_expr_is_max=*/true)),
      f.NextLoc());
  VerilogFunction* fn = f->Make<VerilogFunction>(
      f.NextLoc(), "func", f->Make<TypedefType>(f.NextLoc(), word_t));
  LogicRef* a =
      fn->AddArgument(f->Make<Def>(f.NextLoc(), "a", DataKind::kUser,
                                   f->Make<TypedefType>(f.NextLoc(), word_t)),
                      f.NextLoc());
  fn->AddStatement<ReturnStatement>(
      f.NextLoc(), f->Add(a, f.LiteralWithBitCount(3, 0), f.NextLoc()));
  foo->AddModuleMember(fn);

  constexpr std::string_view kExpected =
      R"(#![allow(nonstandard_constant_naming)]
#![allow(nonstandard_member_naming)]

pub const width = s32:24;

#[sv_type("foo::word_t")]
pub type word_t = bits[width as u32];  // u24

pub fn func(a: word_t) -> word_t { a + word_t:0 }
)";

  XLS_EXPECT_VAST_TRANSLATION(f, kExpected);
}

TEST_F(TranslateVastToDslxTest, EnumAndEquivalentBuiltInTypeOperands) {
  // package foo;
  //   typedef enum logic[23:0] {
  //     kFoo = 0,
  //     kBar = 1
  //   } foo_t;
  //
  //   function automatic logic[23:0] func(logic[15:0] a);
  //     return a + kFoo;
  //   endfunction : func
  // endpackage : foo
  VerilogFileHelper f = CreateFile();
  Module* foo = f->AddModule("foo", f.NextLoc());

  Enum* enum_def = f->Make<Enum>(
      f.NextLoc(), DataKind::kLogic,
      f->Make<BitVectorType>(f.NextLoc(), f.BareLiteral(23),
                             /*is_signed=*/false, /*size_expr_is_max=*/true));
  EnumMemberRef* enum_foo =
      enum_def->AddMember("kFoo", f.BareLiteral(0), f.NextLoc());
  enum_def->AddMember("kBar", f.BareLiteral(1), f.NextLoc());
  foo->AddTypedef(f->Make<Def>(f.NextLoc(), "foo_t", DataKind::kUser, enum_def),
                  f.NextLoc());
  VerilogFunction* fn = f->Make<VerilogFunction>(
      f.NextLoc(), "func",
      f->Make<BitVectorType>(f.NextLoc(), f.BareLiteral(23),
                             /*is_signed=*/false, /*size_expr_is_max=*/true));
  LogicRef* arg_a = fn->AddArgument(
      f->Make<Def>(f.NextLoc(), "a", DataKind::kLogic,
                   f->Make<BitVectorType>(f.NextLoc(), f.BareLiteral(15),
                                          /*is_signed=*/false,
                                          /*size_expr_is_max=*/true)),
      f.NextLoc());
  fn->AddStatement<ReturnStatement>(f.NextLoc(),
                                    f->Add(arg_a, enum_foo, f.NextLoc()));
  foo->AddModuleMember(fn);

  constexpr std::string_view kExpected =
      R"(#[sv_type("foo::foo_t")]
pub enum foo_t : bits[24] {
    kFoo = 0,
    kBar = 1,
}

pub fn func(a: bits[16]) -> bits[24] { a as u24 + foo_t::kFoo as u24 }
)";

  XLS_EXPECT_VAST_TRANSLATION(f, kExpected);
}

TEST_F(TranslateVastToDslxTest, PreserveNamedTypes) {
  // package foo;
  //   typedef logic[41:0] foo_t;
  //   parameter int unsigned MiB = 2**20;
  //   parameter foo_t Value1 = 64 * MiB;
  //   parameter foo_t Value2 = 24'h0_0000_0000_0000;
  // endpackage : foo
  VerilogFileHelper f = CreateFile();
  Module* foo = f->AddModule("foo", f.NextLoc());
  Typedef* foo_t = foo->AddTypedef(
      f->Make<Def>(f.NextLoc(), "foo_t", DataKind::kLogic,
                   f->Make<BitVectorType>(f.NextLoc(), f.BareLiteral(41),
                                          /*is_signed=*/false,
                                          /*size_expr_is_max=*/true)),
      f.NextLoc());
  ParameterRef* mib = foo->AddParameter(
      f->Make<Def>(f.NextLoc(), "MiB", DataKind::kInteger,
                   f->Make<IntegerType>(f.NextLoc(), /*is_signed=*/false)),
      f->Power(f.BareLiteral(2), f.BareLiteral(20), f.NextLoc()), f.NextLoc());
  foo->AddParameter(f->Make<Def>(f.NextLoc(), "Value1", DataKind::kUser,
                                 f->Make<TypedefType>(f.NextLoc(), foo_t)),
                    f->Mul(f.BareLiteral(64), mib, f.NextLoc()), f.NextLoc());
  foo->AddParameter(f->Make<Def>(f.NextLoc(), "Value2", DataKind::kUser,
                                 f->Make<TypedefType>(f.NextLoc(), foo_t)),
                    f.LiteralWithBitCount(24, 0, FormatPreference::kHex),
                    f.NextLoc());

  constexpr std::string_view kExpected =
      R"(#![allow(nonstandard_constant_naming)]
#![allow(nonstandard_member_naming)]

#[sv_type("foo::foo_t")]
pub type foo_t = bits[42];

import std;

pub const MiB = std::spow(s32:2, s32:20 as uN[32]) as u32;
pub const Value1 = foo_t:64 * MiB as foo_t;
pub const Value2 = foo_t:0x0;
)";

  XLS_EXPECT_VAST_TRANSLATION(f, kExpected);
}

}  // namespace
}  // namespace verilog
}  // namespace xls
