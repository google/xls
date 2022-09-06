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
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/xlscc/translator.h"
#include "xls/contrib/xlscc/unit_test.h"

namespace xlscc {
namespace {

class TranslatorPointerTest : public XlsccTestBase {
 public:
};

TEST_F(TranslatorPointerTest, ToArraySubscript) {
  const std::string content = R"(
    int my_package() {
	    int arr[10] = {1,2,3,4,5,6,7,8,9,10};
      int*a = &arr[10];
      return a[0];
    })";
  Run({}, 10, content);
}

TEST_F(TranslatorPointerTest, AssignToPointer) {
  const std::string content = R"(
    int my_package() {
	    int arr[10] = {1,2,3,4,5,6,7,8,9,10};
      int*a;
      a = &arr[10];
      return a[0];
    })";
  Run({}, 10, content);
}

TEST_F(TranslatorPointerTest, AssignToOriginalAndReadPointer) {
  const std::string content = R"(
    int my_package() {
	    int arr[10] = {1,2,3,4,5,6,7,8,9,10};
      int*a = &arr[2];
      arr[2] = 150;
      return a[0];
    })";
  Run({}, 150, content);
}

TEST_F(TranslatorPointerTest, ArraySlice) {
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

TEST_F(TranslatorPointerTest, ArraySliceDynamic) {
  const std::string content = R"(
    int sum(const int v[2]) {
      int ret = 0;
      #pragma hls_unroll yes
      for(int i=0;i<2;++i) {
        ret += v[i];
      }
      return ret;
    }

    int my_package(long long i) {
      int arr[10] = {1,2,3,4,5,6,7,8,9,10};
      int* p = &arr[i];
      return sum(p);
    })";

  Run({{"i", 0}}, 1 + 2, content);
  Run({{"i", 4}}, 5 + 6, content);
}

TEST_F(TranslatorPointerTest, ArraySliceAssign) {
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

TEST_F(TranslatorPointerTest, ArraySliceAssign2) {
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

TEST_F(TranslatorPointerTest, AssignArrayElementByPointer) {
  const std::string content = R"(
    int my_package() {
      int arr[4] = {1,2,3,4};
      int* p = &arr[2];
      p[1] = 10;
      return arr[3];
    })";

  ASSERT_THAT(
      SourceToIr(content).status(),
      xls::status_testing::StatusIs(absl::StatusCode::kUnimplemented,
                                    testing::HasSubstr("not pointers")));
}

TEST_F(TranslatorPointerTest, AddrOfPointer) {
  const std::string content = R"(
    void addto(int v[2]) {
      #pragma hls_unroll yes
      for(int i=0;i<2;++i) {
        v[i] += 3;
      }
    }

    int my_package() {
      int arr[4] = {1,2,3,4};
      int* p = &arr[0];
      addto(&p[2]);
      return arr[3];
    })";

  ASSERT_THAT(SourceToIr(content).status(),
              xls::status_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr("only supported on array")));
}

TEST_F(TranslatorPointerTest, ArraySliceAssignDirect) {
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

TEST_F(TranslatorPointerTest, ArraySliceAssignDirect2) {
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

TEST_F(TranslatorPointerTest, ArraySliceAssignDirect3) {
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

TEST_F(TranslatorPointerTest, ArraySliceAssignDynamic) {
  const std::string content = R"(
    void addto(int v[2]) {
      #pragma hls_unroll yes
      for(int i=0;i<2;++i) {
        v[i] += 3;
      }
    }

    int my_package(long long wi, long long ri) {
      int arr[4] = {1,2,3,4};
      addto(&arr[wi]);
      return arr[ri];
    })";

  Run({{"wi", 1}, {"ri", 0}}, 1, content);
  Run({{"wi", 1}, {"ri", 1}}, 2 + 3, content);
  Run({{"wi", 1}, {"ri", 2}}, 3 + 3, content);
  Run({{"wi", 1}, {"ri", 3}}, 4, content);

  Run({{"wi", 0}, {"ri", 0}}, 1 + 3, content);
  Run({{"wi", 0}, {"ri", 1}}, 2 + 3, content);
  Run({{"wi", 0}, {"ri", 2}}, 3, content);
  Run({{"wi", 0}, {"ri", 3}}, 4, content);
}

TEST_F(TranslatorPointerTest, ArraySliceOutOfBounds) {
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

TEST_F(TranslatorPointerTest, ArraySliceAssignTrueCondition) {
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

TEST_F(TranslatorPointerTest, ArraySliceAssignFalseCondition) {
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

TEST_F(TranslatorPointerTest, ArraySliceAssignRuntimeCondition) {
  const std::string content = R"(
    void addto(int v[4]) {
      #pragma hls_unroll yes
      for(int i=0;i<4;++i) {
        v[i] += 3;
      }
    }

    int my_package(bool configured) {
      int brr[6] = {10,20,30,40,50,60};
      int arr[4] = {1,2,3,4};
      int* p = &brr[0];
      if(configured) {
        p = &arr[0];
      }
      addto(p);
      return p[3];
    })";

  Run({{"configured", 1}}, 4 + 3, content);
  Run({{"configured", 0}}, 40 + 3, content);
}

TEST_F(TranslatorPointerTest, ArraySliceAssignRuntimeCondition2) {
  const std::string content = R"(
    void addto(int v[4]) {
      #pragma hls_unroll yes
      for(int i=0;i<4;++i) {
        v[i] += 3;
      }
    }

    int my_package(bool c0, bool c1) {
      int brr[6] = {10,20,30,40,50,60};
      int arr[4] = {1,2,3,4};
      int* p = &arr[0];
      if(c0) {
        p = &brr[0];
      }
      if(c1) {
        p = &brr[2];
      }
      addto(p);
      return p[3];
    })";

  Run({{"c0", 0}, {"c1", 0}}, 3 + 4, content);
  Run({{"c0", 1}, {"c1", 0}}, 3 + 40, content);
  Run({{"c0", 1}, {"c1", 1}}, 3 + 60, content);
  Run({{"c0", 0}, {"c1", 1}}, 3 + 60, content);
}

TEST_F(TranslatorPointerTest, ArraySliceAssignTernary) {
  const std::string content = R"(
    void addto(int v[2]) {
      #pragma hls_unroll yes
      for(int i=0;i<2;++i) {
        v[i] += 3;
      }
    }

    int my_package(bool configured) {
      int arr[4] = {1,2,3,4};
      int* p = configured ? &arr[2] : &arr[0];
      addto(p);
      return arr[3];
    })";

  Run({{"configured", 1}}, 4 + 3, content);
  Run({{"configured", 0}}, 4, content);
}

TEST_F(TranslatorPointerTest, SelectDifferentSizes) {
  const std::string content = R"(
    void addto(int v[4]) {
      #pragma hls_unroll yes
      for(int i=0;i<4;++i) {
        v[i] += 3;
      }
    }

    int my_package(bool configured) {
      int brr[6] = {10,20,30,40,50,60};
      int arr[4] = {1,2,3,4};
      int* p = configured ? &arr[0] : &brr[0];
      addto(p);
      return p[3];
    })";

  Run({{"configured", 1}}, 4 + 3, content);
  Run({{"configured", 0}}, 40 + 3, content);
}

TEST_F(TranslatorPointerTest, NestedSelect) {
  const std::string content = R"(
    void addto(int v[2]) {
      #pragma hls_unroll yes
      for(int i=0;i<2;++i) {
        v[i] += 3;
      }
    }

    int my_package(bool c1, bool c2) {
      int brr[4] = {10,20,30,40};
      int arr[4] = {1,2,3,4};
      int* p = c1 ? &brr[1] : (c2 ? &arr[2] : &arr[0]);
      return p[0];
    })";

  Run({{"c1", 0}, {"c2", 0}}, 1, content);
  Run({{"c1", 0}, {"c2", 1}}, 3, content);
  Run({{"c1", 1}, {"c2", 0}}, 20, content);
  Run({{"c1", 1}, {"c2", 1}}, 20, content);
}

TEST_F(TranslatorPointerTest, ArraySliceAssignTernary2) {
  const std::string content = R"(
    void addto(int v[2]) {
      #pragma hls_unroll yes
      for(int i=0;i<2;++i) {
        v[i] += 3;
      }
    }

    int my_package(bool configured) {
      int arr[4] = {1,2,3,4};
      int* p;
      p = configured ? &arr[2] : &arr[0];
      addto(p);
      return arr[3];
    })";

  Run({{"configured", 1}}, 4 + 3, content);
  Run({{"configured", 0}}, 4, content);
}

TEST_F(TranslatorPointerTest, ArraySliceAssignTernaryDirect) {
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

  Run({}, 4 + 3, content);
}

TEST_F(TranslatorPointerTest, ArraySliceAssignTernaryDirect2) {
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
      addto(configured ? &arr[2] : &arr[0]);
      return arr[3];
    })";

  Run({}, 4, content);
}

TEST_F(TranslatorPointerTest, ArraySliceAssignTernaryDirect3) {
  const std::string content = R"(
    void addto(int v[2]) {
      #pragma hls_unroll yes
      for(int i=0;i<2;++i) {
        v[i] += 3;
      }
    }

    int my_package(bool configured) {
      int arr[4] = {1,2,3,4};
      addto(configured ? &arr[2] : &arr[0]);
      return arr[3];
    })";

  Run({{"configured", 1}}, 4 + 3, content);
  Run({{"configured", 0}}, 4, content);
}

TEST_F(TranslatorPointerTest, ArraySliceAssignNestedTernary) {
  const std::string content = R"(
    void addto(int v[2]) {
      #pragma hls_unroll yes
      for(int i=0;i<2;++i) {
        v[i] += 3;
      }
    }
    int my_package(bool w1, bool w2, bool r1, bool r2) {
      int brr[4] = {10,20,30,40};
      int arr[4] = {1,2,3,4};
      int* p = w1 ? &brr[1] : (w2 ? &arr[2] : &arr[0]);
      addto(p);
      return r1 ? brr[1] : (r2 ? arr[2] : arr[0]);
    })";

  for (int w1 = 0; w1 <= 1; ++w1) {
    for (int w2 = 0; w2 <= 1; ++w2) {
      for (int r1 = 0; r1 <= 1; ++r1) {
        for (int r2 = 0; r2 <= 1; ++r2) {
          int brr[4] = {10, 20, 30, 40};
          int arr[4] = {1, 2, 3, 4};
          int* const p = w1 != 0 ? &brr[1] : (w2 != 0 ? &arr[2] : &arr[0]);
          for (int i = 0; i < 2; ++i) {
            p[i] += 3;
          }
          const int read = r1 != 0 ? brr[1] : (r2 != 0 ? arr[2] : arr[0]);
          Run({{"w1", w1}, {"w2", w2}, {"r1", r1}, {"r2", r2}}, read, content);
        }
      }
    }
  }
}

TEST_F(TranslatorPointerTest, ArraySliceReadTernaryDirect) {
  const std::string content = R"(
    int get3(const int v[4]) {
      return v[3];
    }

    int my_package(bool configured) {
      int arr[6] = {1,2,3,4,5,6};
      int* p = configured ? &arr[2] : &arr[0];
      return get3(p);
    })";

  Run({{"configured", 1}}, 6, content);
  Run({{"configured", 0}}, 4, content);
}

TEST_F(TranslatorPointerTest, UninitializedPointer) {
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

TEST_F(TranslatorPointerTest, Aliasing) {
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
