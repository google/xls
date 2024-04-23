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

#include "xls/examples/sample_packages.h"

#include <cstdint>
#include <filesystem>  // NOLINT
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/file/path.h"
#include "xls/common/status/status_macros.h"
#include "xls/examples/sample_packages.inc.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/verifier.h"

namespace xls {
namespace sample_packages {

std::pair<std::unique_ptr<Package>, Function*> BuildRrot32() {
  auto p = std::make_unique<Package>("Rrot32");
  FunctionBuilder b("rrot", p.get());
  Type* bits_32 = p->GetBitsType(32);
  auto x = b.Param("x", bits_32);
  auto y = b.Param("y", bits_32);
  auto imm_32 = b.Literal(UBits(32, 32));
  auto lhs = (x >> y);
  auto rhs_shamt = (imm_32 - y);
  auto rhs = (x << rhs_shamt);
  auto result = lhs | rhs;
  absl::StatusOr<Function*> f = b.BuildWithReturnValue(result);
  CHECK_OK(f.status());
  CHECK_OK(VerifyPackage(p.get()));
  return {std::move(p), *f};
}

std::pair<std::unique_ptr<Package>, Function*> BuildRrot8Fixed() {
  auto p = std::make_unique<Package>("Rrot8fixed");
  std::string input = R"(
     fn rrot3(x: bits[8]) -> bits[8] {
        three: bits[8] = literal(value=3)
        eight: bits[8] = literal(value=8)
        expr1: bits[8] = shrl(x, three)
        shift: bits[8] = sub(eight, three)
        expr2: bits[8] = shll(x, shift)
        ret orval: bits[8] = or(expr1, expr2)
     }
})";
  Function* function = Parser::ParseFunction(input, p.get()).value();
  QCHECK_OK(p->SetTop(function));
  return {std::move(p), function};
}

std::pair<std::unique_ptr<Package>, Function*> BuildAbs32() {
  auto p = std::make_unique<Package>("Abs32");
  FunctionBuilder b("abs", p.get());
  Type* bits_32 = p->GetBitsType(32);
  auto x = b.Param("x", bits_32);
  auto imm_31 = b.Literal(UBits(31, /*bit_count=*/32));
  auto is_neg = b.Eq(x >> imm_31, b.Literal(UBits(1, /*bit_count=*/32)));
  auto result = b.Select(is_neg, -x, x);
  absl::StatusOr<Function*> f = b.BuildWithReturnValue(result);
  CHECK_OK(f.status());
  CHECK_OK(VerifyPackage(p.get()));
  return {std::move(p), *f};
}

std::pair<std::unique_ptr<Package>, Function*> BuildConcatWith1() {
  auto p = std::make_unique<Package>("ConcatWith1");
  FunctionBuilder b("concat_with_1", p.get());
  Type* bits_31 = p->GetBitsType(31);
  auto x = b.Param("x", bits_31);
  auto imm_1 = b.Literal(UBits(1, /*bit_count=*/1));
  b.Concat({x, imm_1});
  absl::StatusOr<Function*> f = b.Build();
  CHECK_OK(f.status());
  CHECK_OK(VerifyPackage(p.get()));
  return {std::move(p), *f};
}

std::pair<std::unique_ptr<Package>, Function*> BuildSignExtendTo32() {
  auto p = std::make_unique<Package>("SignExtendTo32");
  FunctionBuilder b("sign_extend_to_32", p.get());
  Type* bits_2 = p->GetBitsType(2);
  auto x = b.Param("x", bits_2);
  b.SignExtend(x, 32);
  absl::StatusOr<Function*> f = b.Build();
  CHECK_OK(f.status());
  CHECK_OK(VerifyPackage(p.get()));
  return {std::move(p), *f};
}

std::pair<std::unique_ptr<Package>, Function*> BuildZeroExtendTo32() {
  auto p = std::make_unique<Package>("ZeroExtendTo32");
  FunctionBuilder b("zero_extend_to_32", p.get());
  Type* bits_2 = p->GetBitsType(2);
  auto x = b.Param("x", bits_2);
  b.ZeroExtend(x, 32);
  absl::StatusOr<Function*> f = b.Build();
  CHECK_OK(f.status());
  CHECK_OK(VerifyPackage(p.get()));
  return {std::move(p), *f};
}

std::pair<std::unique_ptr<Package>, Function*> BuildAccumulateIvar(
    int64_t trip_count, int64_t bit_count) {
  auto p = std::make_unique<Package>("AccumulateIvar");
  Function* body = nullptr;
  {
    FunctionBuilder fb("body", p.get());
    auto x = fb.Param("x", p->GetBitsType(bit_count));
    auto y = fb.Param("y", p->GetBitsType(bit_count));
    absl::StatusOr<Function*> f = fb.BuildWithReturnValue(x + y);
    CHECK_OK(f.status());
    body = *f;
  }

  FunctionBuilder fb("main", p.get());
  BValue zero = fb.Literal(UBits(0, bit_count));
  BValue result = fb.CountedFor(/*init_value=*/zero, /*trip_count=*/trip_count,
                                /*stride=*/1, body);
  absl::StatusOr<Function*> f = fb.BuildWithReturnValue(result);
  CHECK_OK(f.status());
  CHECK_OK(VerifyPackage(p.get()));
  return {std::move(p), *f};
}

std::pair<std::unique_ptr<Package>, Function*> BuildTwoLoops(
    bool same_trip_count, bool dependent_loops) {
  int64_t bit_count = 16;
  int64_t trip_cnt1 = 8;
  int64_t trip_cnt2 = same_trip_count ? trip_cnt1 : trip_cnt1 * 2;

  auto p = std::make_unique<Package>("TwoLoops");
  FunctionBuilder fb("main", p.get());
  auto param = fb.Param("param", p->GetBitsType(bit_count));

  Function* inner_body1;
  {
    FunctionBuilder fb("innerbody1", p.get());
    auto x = fb.Param("x", p->GetBitsType(bit_count));
    auto y = fb.Param("y", p->GetBitsType(bit_count));
    absl::StatusOr<Function*> f = fb.BuildWithReturnValue(x + y);
    CHECK_OK(f.status());
    inner_body1 = *f;
  }
  Function* inner_body2;
  {
    FunctionBuilder fb("innerbody2", p.get());
    auto x = fb.Param("x", p->GetBitsType(bit_count));
    auto y = fb.Param("y", p->GetBitsType(bit_count));
    absl::StatusOr<Function*> f = fb.BuildWithReturnValue(x - y);
    CHECK_OK(f.status());
    inner_body2 = *f;
  }
  BValue zero1 = fb.Literal(UBits(0, bit_count));
  BValue zero2 = fb.Literal(UBits(0, bit_count));
  BValue loop1 = fb.CountedFor(/*init_value=*/zero1, /*trip_count=*/trip_cnt1,
                               /*stride=*/1, inner_body1);
  BValue loop2 = fb.CountedFor(/*init_value=*/zero2, /*trip_count=*/trip_cnt2,
                               /*stride=*/1, inner_body2);

  BValue result2 = dependent_loops ? param + loop1 + loop2 : loop2;
  absl::StatusOr<Function*> f = fb.BuildWithReturnValue(result2);
  CHECK_OK(f.status());
  CHECK_OK(VerifyPackage(p.get()));
  return {std::move(p), *f};
}

std::pair<std::unique_ptr<Package>, Function*> BuildSimpleMap(
    int element_count) {
  auto p = std::make_unique<Package>("SimpleMap");
  BitsType* element_type = p->GetBitsType(42);
  ArrayType* array_type = p->GetArrayType(element_count, element_type);
  Function* to_apply;
  {
    FunctionBuilder b("to_apply", p.get());
    b.ULt(b.Param("element", element_type),
          b.Literal(UBits(10, element_type->bit_count())));
    absl::StatusOr<Function*> f = b.Build();
    CHECK_OK(f.status());
    to_apply = *f;
  }
  Function* top;
  {
    FunctionBuilder b("top", p.get());
    b.Map(b.Param("input", array_type), to_apply);
    absl::StatusOr<Function*> f = b.Build();
    CHECK_OK(f.status());
    top = *f;
  }
  CHECK_OK(VerifyPackage(p.get()));
  return {std::move(p), top};
}

absl::StatusOr<std::unique_ptr<Package>> GetBenchmark(std::string_view name,
                                                      bool optimized) {
  std::filesystem::path filename =
      optimized ? absl::StrCat(name, ".opt.ir") : absl::StrCat(name, ".ir");
  XLS_ASSIGN_OR_RETURN(std::filesystem::path runfile_path,
                       GetXlsRunfilePath("xls" / filename));
  absl::StatusOr<std::string> ir_status = GetFileContents(runfile_path);
  if (!ir_status.ok()) {
    return absl::Status(ir_status.status().code(),
                        absl::StrFormat("GetBenchmark %s failed: %s", name,
                                        ir_status.status().message()));
  }
  return Parser::ParsePackage(ir_status.value());
}

absl::StatusOr<std::vector<std::string>> GetBenchmarkNames() {
  XLS_ASSIGN_OR_RETURN(std::vector<std::string> example_paths,
                       GetExamplePaths());
  XLS_ASSIGN_OR_RETURN(
      std::filesystem::path example_file_list_path,
      GetXlsRunfilePath("xls/examples/ir_example_file_list.txt"));

  std::vector<std::string> names;
  for (auto& example_path : example_paths) {
    XLS_ASSIGN_OR_RETURN(std::string absolute_example_path,
                         GetXlsRunfilePath(example_path));
    std::string_view example_path_view = absolute_example_path;
    // Check if the file name ends with ".opt.ir". If so, strip the suffix and
    // add the result to names.
    if (absl::ConsumeSuffix(&example_path_view, ".opt.ir")) {
      // Find the path to the example relative to the example file list.
      XLS_ASSIGN_OR_RETURN(
          std::filesystem::path path_relative_to_example_directory,
          RelativizePath(example_path_view,
                         example_file_list_path.parent_path().parent_path()));

      names.push_back(path_relative_to_example_directory);
    }
  }
  return names;
}

}  // namespace sample_packages
}  // namespace xls
