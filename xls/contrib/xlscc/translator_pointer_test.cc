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

#include <cstdio>
#include <memory>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_format.h"
#include "xls/codegen/block_conversion.h"
#include "xls/codegen/block_generator.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/combinational_generator.h"
#include "xls/common/file/temp_file.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/xlscc/translator.h"
#include "xls/contrib/xlscc/unit_test.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/value.h"

namespace xlscc {
namespace {

class TranslatorTest : public XlsccTestBase {
 public:
};

TEST_F(TranslatorTest, PointerToArraySubscript) {
  const std::string content = R"(
    int my_package() {
	    int arr[10] = {1,2,3,4,5,6,7,8,9,10};
      int*a = &arr[10];
      return a[0];
    })";
  Run({}, 10, content);
}

TEST_F(TranslatorTest, AssignToPointer) {
  const std::string content = R"(
    int my_package() {
	    int arr[10] = {1,2,3,4,5,6,7,8,9,10};
      int*a;
      a = &arr[10];
      return a[0];
    })";
  Run({}, 10, content);
}

TEST_F(TranslatorTest, AssignToOriginalAndReadPointer) {
  const std::string content = R"(
    int my_package() {
	    int arr[10] = {1,2,3,4,5,6,7,8,9,10};
      int*a = &arr[2];
      arr[2] = 150;
      return a[0];
    })";
  Run({}, 150, content);
}

TEST_F(TranslatorTest, PointerArraySlice) {
  const std::string content = R"(
    int sum(const int v[2]) {
      int ret = 0;
      #pragma hls_unroll yes
      for(int i=0;i<2;++i) {
        ret += v[i];
      }
      return ret;
    }

    int my_package() {
      int arr[10] = {1,2,3,4,5,6,7,8,9,10};
      int* p = &arr[4];
      return sum(p);
    })";

  Run({}, 5 + 6, content);
}

TEST_F(TranslatorTest, PointerArraySliceAssign) {
  const std::string content = R"(
    void addto(int v[2]) {
      #pragma hls_unroll yes
      for(int i=0;i<2;++i) {
        v[i] += 3;
      }
    }

    int my_package() {
      int arr[4] = {1,2,3,4};
      int* p = &arr[2];
      addto(p);
      return arr[3];
    })";

  Run({}, 4 + 3, content);
}

TEST_F(TranslatorTest, PointerArraySliceAssign2) {
  const std::string content = R"(
    void addto(int v[2]) {
      #pragma hls_unroll yes
      for(int i=0;i<2;++i) {
        v[i] += 3;
      }
    }

    int my_package() {
      int arr[4] = {1,2,3,4};
      int* p;
      p = &arr[2];
      addto(p);
      return arr[3];
    })";

  Run({}, 4 + 3, content);
}

TEST_F(TranslatorTest, PointerArraySliceAssignDirect) {
  const std::string content = R"(
    void addto(int v[2]) {
      #pragma hls_unroll yes
      for(int i=0;i<2;++i) {
        v[i] += 3;
      }
    }

    int my_package() {
      int arr[4] = {1,2,3,4};
      addto(&arr[2]);
      return arr[3];
    })";

  Run({}, 4 + 3, content);
}

TEST_F(TranslatorTest, PointerArraySliceAssignDirect2) {
  const std::string content = R"(
    void addto(int v[2]) {
      #pragma hls_unroll yes
      for(int i=0;i<2;++i) {
        v[i] += 3;
      }
    }

    int my_package() {
      int arr[4] = {1,2,3,4};
      addto(&arr[0]);
      return arr[0];
    })";

  Run({}, 1 + 3, content);
}

TEST_F(TranslatorTest, PointerArraySliceAssignDirect3) {
  const std::string content = R"(
    void addto(int v[2]) {
      #pragma hls_unroll yes
      for(int i=0;i<2;++i) {
        v[i] += 3;
      }
    }

    int my_package() {
      int arr[4] = {1,2,3,4};
      addto(&arr[1]);
      return arr[2];
    })";

  Run({}, 3 + 3, content);
}

TEST_F(TranslatorTest, PointerArraySliceOutOfBounds) {
  const std::string content = R"(
    void addto(int v[2]) {
      #pragma hls_unroll yes
      for(int i=0;i<2;++i) {
        v[i] += 3;
      }
    }

    int my_package() {
      int arr[4] = {1,2,3,4};
      addto(&arr[3]);
      return arr[3];
    })";

  ASSERT_THAT(SourceToIr(content).status(),
              xls::status_testing::StatusIs(absl::StatusCode::kOutOfRange,
                                            testing::HasSubstr("slice")));
}

TEST_F(TranslatorTest, PointerArraySliceAssignTrueCondition) {
  const std::string content = R"(
    void addto(int v[2]) {
      #pragma hls_unroll yes
      for(int i=0;i<2;++i) {
        v[i] += 3;
      }
    }

    int my_package() {
      const bool configured = true;
      int arr[4] = {1,2,3,4};
      int* p;
      if(configured) {
        p = &arr[2];
      }
      addto(p);
      return arr[3];
    })";

  Run({}, 4 + 3, content);
}

TEST_F(TranslatorTest, PointerArraySliceAssignFalseCondition) {
  const std::string content = R"(
    void addto(int v[2]) {
      #pragma hls_unroll yes
      for(int i=0;i<2;++i) {
        v[i] += 3;
      }
    }

    int my_package() {
      const bool configured = false;
      int arr[4] = {1,2,3,4};
      int* p = &arr[0];
      if(configured) {
        p = &arr[2];
      }
      addto(p);
      return arr[3];
    })";

  Run({}, 4, content);
}

TEST_F(TranslatorTest, PointerArraySliceAssignRuntimeCondition) {
  const std::string content = R"(
    void addto(int v[2]) {
      #pragma hls_unroll yes
      for(int i=0;i<2;++i) {
        v[i] += 3;
      }
    }

    int my_package(int configured) {
      int arr[4] = {1,2,3,4};
      int* p = &arr[0];
      if(configured) {
        p = &arr[2];
      }
      addto(p);
      return arr[3];
    })";

  ASSERT_THAT(SourceToIr(content).status(),
              xls::status_testing::StatusIs(
                  absl::StatusCode::kUnimplemented,
                  testing::HasSubstr("compile-time constant")));
}

TEST_F(TranslatorTest, PointerArraySliceAssignTernary) {
  const std::string content = R"(
    void addto(int v[2]) {
      #pragma hls_unroll yes
      for(int i=0;i<2;++i) {
        v[i] += 3;
      }
    }

    int my_package() {
      const bool configured = true;
      int arr[4] = {1,2,3,4};
      int* p = configured ? &arr[2] : &arr[0];
      addto(p);
      return arr[3];
    })";

  ASSERT_THAT(SourceToIr(content).status(),
              xls::status_testing::StatusIs(absl::StatusCode::kUnimplemented,
                                            testing::HasSubstr("no lvalue")));
}

TEST_F(TranslatorTest, PointerArraySliceAssignTernary2) {
  const std::string content = R"(
    void addto(int v[2]) {
      #pragma hls_unroll yes
      for(int i=0;i<2;++i) {
        v[i] += 3;
      }
    }

    int my_package() {
      const bool configured = true;
      int arr[4] = {1,2,3,4};
      int* p;
      p = configured ? &arr[2] : &arr[0];
      addto(p);
      return arr[3];
    })";

  ASSERT_THAT(SourceToIr(content).status(),
              xls::status_testing::StatusIs(absl::StatusCode::kUnimplemented,
                                            testing::HasSubstr("no lvalue")));
}

TEST_F(TranslatorTest, PointerArraySliceAssignTernaryDirect) {
  const std::string content = R"(
    void addto(int v[2]) {
      #pragma hls_unroll yes
      for(int i=0;i<2;++i) {
        v[i] += 3;
      }
    }

    int my_package() {
      const bool configured = true;
      int arr[4] = {1,2,3,4};
      addto(configured ? &arr[2] : &arr[0]);
      return arr[3];
    })";

  ASSERT_THAT(SourceToIr(content).status(),
              xls::status_testing::StatusIs(absl::StatusCode::kUnimplemented,
                                            testing::HasSubstr("no lvalue")));
}

TEST_F(TranslatorTest, PointerArraySliceReadTernaryDirect) {
  const std::string content = R"(
    int get3(const int v[4]) {
      return v[3];
    }

    int my_package() {
      const bool configured = true;
      int arr[4] = {1,2,3,4};
      int* p = configured ? &arr[2] : &arr[0];
      return get3(p);
    })";

  ASSERT_THAT(SourceToIr(content).status(),
              xls::status_testing::StatusIs(absl::StatusCode::kUnimplemented,
                                            testing::HasSubstr("no lvalue")));
}

TEST_F(TranslatorTest, UninitializedPointer) {
  const std::string content = R"(
    int my_package() {
	    int arr[10] = {1,2,3,4,5,6,7,8,9,10};
      int*a;
      return a[0];
    })";

  ASSERT_THAT(
      SourceToIr(content).status(),
      xls::status_testing::StatusIs(absl::StatusCode::kFailedPrecondition,
                                    testing::HasSubstr("Unable to parse")));
}

TEST_F(TranslatorTest, PointerAliasing) {
  const std::string content = R"(
    void addto(int lhs[2], int rhs[1]) {
      lhs[0] += 2;
      lhs[1] += 2;
      rhs[0] += 5;
    }

    int my_package() {
      int arr[2] = {1, 2};
      addto(&arr[0], &arr[1]);
      return arr[0] + arr[1];
    })";

  // All rvalues are prepared to be passed into the invoke() first,
  // then all lvalues are updated. Therefore, the &arr[1] overwrites the &arr[0]
  // This does not match the behavior of clang. This could be solved using
  // multiple invokes, as is done for IO. However, I think it would be best to
  // just error-out via the unsequenced assignment detection mechanism.
  // https://github.com/google/xls/issues/572
  Run({}, 1 + 2 + 2 + 5, content);
}

}  // namespace

}  // namespace xlscc
