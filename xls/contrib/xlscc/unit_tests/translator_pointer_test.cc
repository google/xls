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

#include <list>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xls/contrib/xlscc/hls_block.pb.h"
#include "xls/contrib/xlscc/unit_tests/unit_test.h"
#include "xls/ir/bits.h"
#include "xls/ir/value.h"

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

  ASSERT_THAT(SourceToIr(content).status(),
              absl_testing::StatusIs(absl::StatusCode::kUnimplemented,
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

  ASSERT_THAT(
      SourceToIr(content).status(),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
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
              absl_testing::StatusIs(absl::StatusCode::kOutOfRange,
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

TEST_F(TranslatorPointerTest, IncrementInPointerSelect) {
  const std::string content = R"(
    void addto(int v[2]) {
      #pragma hls_unroll yes
      for(int i=0;i<2;++i) {
        v[i] += 3;
      }
    }
    int my_package(int a) {
      int arr[4] = {1,2,3,4};
      int* p = (a++) ? &arr[2] : &arr[0];
      addto(p);
      return a + arr[0];
    })";

  Run({{"a", 4}}, 4 + 1 + 1, content);
  Run({{"a", 0}}, 0 + 1 + 1 + 3, content);
}

TEST_F(TranslatorPointerTest, IncrementInPointerSelect2) {
  const std::string content = R"(
    void addto(int v[2]) {
      #pragma hls_unroll yes
      for(int i=0;i<2;++i) {
        v[i] += 3;
      }
    }
    int my_package(int a) {
      int arr[4] = {1,2,3,4};
      addto((a++) ? &arr[2] : &arr[0]);
      return a + arr[0];
    })";

  Run({{"a", 4}}, 4 + 1 + 1, content);
  Run({{"a", 0}}, 0 + 1 + 1 + 3, content);
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

  ASSERT_THAT(SourceToIr(content).status(),
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition,
                                     testing::HasSubstr("uninitialized")));
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

TEST_F(TranslatorPointerTest, Nullptr) {
  const std::string content = R"(
      struct DebugWriter;

      struct dec_ref_store_dataImpl {
        struct DebugWriter* writer_ = nullptr;  // Lazily initialized writer.
      };

      #pragma hls_top
      int test(int a) {
          struct DebugWriter* writer_ = nullptr;  // Lazily initialized writer.
          (void)writer_;
          return a+5;
      })";

  ASSERT_THAT(SourceToIr(content).status(),
              absl_testing::StatusIs(absl::StatusCode::kUnimplemented,
                                     testing::HasSubstr("nullptr")));
}

TEST_F(TranslatorPointerTest, PointerInStruct) {
  const std::string content = R"(
    struct MyPtr {
      int *p;
    };

    int my_package() {
      int arr[3] = {1, 2, 3};
      MyPtr ptr = {.p = &arr[1]};
      return ptr.p[1];
    })";

  Run({}, 3, content);
}

TEST_F(TranslatorPointerTest, PointerInStruct2) {
  const std::string content = R"(
    struct MyPtr {
      int *p;
    };

    int my_package() {
      int arr[3] = {1, 2, 3};
      MyPtr ptr;
      ptr.p = &arr[0];
      return ptr.p[1];
    })";

  // Default constructor causes lvalue translation
  ASSERT_THAT(
      SourceToIr(content).status(),
      absl_testing::StatusIs(
          absl::StatusCode::kUnimplemented,
          testing::HasSubstr("Don't know how to create LValue for member")));
}

TEST_F(TranslatorPointerTest, PointerInStruct3) {
  const std::string content = R"(
    struct MyPtr {
      int *p;
      int get()const {
        return p[1];
      }
    };

    int my_package() {
      int arr[3] = {1, 2, 3};
      MyPtr ptr;
      ptr.p = &arr[0];
      return ptr.get();
    })";

  ASSERT_THAT(
      SourceToIr(content).status(),
      absl_testing::StatusIs(
          absl::StatusCode::kUnimplemented,
          testing::HasSubstr("Don't know how to create LValue for member")));
}

TEST_F(TranslatorPointerTest, ConstPointerToSingleValue) {
  const std::string content = R"(
    int my_package() {
      int foo = 10;
      const int &p = foo;
      ++foo;
      return p;
    })";

  Run({}, 11, content);
}

TEST_F(TranslatorPointerTest, PointerToSingleValue) {
  const std::string content = R"(
    int my_package() {
      int foo = 10;
      int &p = foo;
      p = 3;
      return foo;
    })";

  Run({}, 3, content);
}

TEST_F(TranslatorPointerTest, PointerToSingleValueInc) {
  const std::string content = R"(
    int my_package() {
      int foo = 10;
      int &p = foo;
      p++;
      return foo;
    })";

  Run({}, 11, content);
}

TEST_F(TranslatorPointerTest, ReferenceFromPointer) {
  const std::string content = R"(

    int my_package() {
      int arr[3] = {1, 2, 3};
      int* p = &arr[1];
      int& ret = p[1];
      return ret;
    })";
  Run({}, 3, content);
}

TEST_F(TranslatorPointerTest, ReferenceFromStruct) {
  const std::string content = R"(
    struct MyPtr {
      int &p;
    };

    int my_package() {
      int x = 55;
      MyPtr my = {.p = x};
      x += 10;
      return my.p;
    })";

  Run({}, 65, content);
}

TEST_F(TranslatorPointerTest, ReferenceFromStruct2) {
  const std::string content = R"(
    struct MyPtr {
      int &p;
    };

    int my_package() {
      int x = 55;
      MyPtr my = {.p = x};
      my.p += 11;
      return x;
    })";

  Run({}, 66, content);
}

TEST_F(TranslatorPointerTest, ReferenceInNestedStruct) {
  const std::string content = R"(
    struct MyPtrInner {
      int &p;
      MyPtrInner(int &p) : p(p) { }
    };

    struct MyPtr {
      MyPtrInner inner;
      MyPtr(int& p) : inner(p) { }
    };

    int my_package() {
      int x = 55;
      MyPtr my(x);

      return my.inner.p;
    })";

  ASSERT_THAT(
      SourceToIr(content).status(),
      absl_testing::StatusIs(
          absl::StatusCode::kUnimplemented,
          testing::HasSubstr("Don't know how to create LValue for member")));
}

TEST_F(TranslatorPointerTest, ReferenceInNestedStruct2) {
  const std::string content = R"(
    struct MyPtrInner {
      int &p;
      MyPtrInner(int &p) : p(p) { }
    };

    struct MyPtr {
      MyPtrInner inner;
      MyPtr(int& p) : inner(p) { }
    };

    int my_package() {
      int x = 55;
      MyPtr my(x);

      my.inner.p += 11;

      return x;
    })";

  ASSERT_THAT(
      SourceToIr(content).status(),
      absl_testing::StatusIs(
          absl::StatusCode::kUnimplemented,
          testing::HasSubstr("Don't know how to create LValue for member")));
}

TEST_F(TranslatorPointerTest, ReferenceFuncParam) {
  const std::string content = R"(

    int foo(int& x) { 
      return x*2;
    }

    int my_package() {
      int x = 4;
      int &y = x;
      return foo(y);
    })";

  Run({}, 8, content);
}

TEST_F(TranslatorPointerTest, ReferenceFuncParam2) {
  const std::string content = R"(

    void foo(int& x) { 
      x += 10;
    }

    int my_package() {
      int x = 4;
      int &y = x;
      foo(y);
      return x;
    })";

  Run({}, 14, content);
}

TEST_F(TranslatorPointerTest, ReferenceFuncReturn) {
  const std::string content = R"(
    int* foop(int& xparam) {
      return &xparam;
    }

    int& foo(int& xparam) {
      return xparam;
    }

    int my_package() {
      int x = 4;
      
      int &xr = x;
      int &y = foo(xr);
      return y;
    })";

  ASSERT_THAT(
      SourceToIr(content).status(),
      absl_testing::StatusIs(
          absl::StatusCode::kUnimplemented,
          testing::HasSubstr("Don't know how to handle lvalue return")));
}

TEST_F(TranslatorPointerTest, ReferenceReturnThis) {
  const std::string content = R"(
    struct MyPtr {
      int val;

      MyPtr& operator+=(int x) {
        val += x;
        return *this;
      }
      int getit()const {
        return val;
      }
    };

    int my_package() {
      MyPtr my = {.val = 55};
      MyPtr& ref = (my += 11);
      my += 5;
      ref += 7;
      return ref.getit();
    })";

  Run({}, 55 + 11 + 5 + 7, content);
}

TEST_F(TranslatorPointerTest, ReferenceReturnThis2) {
  const std::string content = R"(
    struct MyPtr {
      int val;

      MyPtr& operator+=(int x) {
        val += x;
        return *this;
      }
      int getit()const {
        return val;
      }
    };

    int my_package() {
      MyPtr my = {.val = 55};
      MyPtr& ref = (my += 11);
      (void)ref.getit();
      return ref.getit();
    })";

  Run({}, 55 + 11, content);
}

TEST_F(TranslatorPointerTest, ReferenceMemberAccess) {
  const std::string content = R"(
    struct MyPtr {
      int val;

      void minus_one() {
        --val;
      }
    };

    int my_package() {
      MyPtr my = {.val = 55};
      MyPtr& ref = my;
      ref.minus_one();
      return ref.val;
    })";

  Run({}, 55 - 1, content);
}

TEST_F(TranslatorPointerTest, ReferenceMemberAssignment) {
  const std::string content = R"(
    struct MyPtr {
      int val;
    };

    int my_package() {
      MyPtr my = {.val = 55};
      MyPtr& ref = my;
      --ref.val;
      return ref.val;
    })";

  Run({}, 55 - 1, content);
}

TEST_F(TranslatorPointerTest, SetThis) {
  const std::string content = R"(
       struct Test {
         void set_this(int v) {
           Test t;
           t.x = v;
           *this = t;
         }
         int x;
         int y;
       };
       int my_package(int a) {
         Test s;
         s.set_this(a);
         s.y = 12;
         return s.x+s.y;
       })";
  Run({{"a", 3}}, 15, content);
}

TEST_F(TranslatorPointerTest, ThisCall) {
  const std::string content = R"(
       struct Test {
         void set_this(int v) {
           Test t;
           t.x = v;
           *this = t;
         }
         void set_this_b(int v) {
           set_this(v);
         }
         int x;
         int y;
       };
       int my_package(int a) {
         Test s;
         s.set_this_b(a);
         s.y = 12;
         return s.x+s.y;
       })";
  Run({{"a", 3}}, 15, content);
}

TEST_F(TranslatorPointerTest, PointerToPointer) {
  const std::string content = R"(
    int my_package() {
      int x = 1;
      int *xp = &x;
      int **xpp = &xp;
      return **xpp;
    })";

  ASSERT_THAT(
      SourceToIr(content).status(),
      absl_testing::StatusIs(absl::StatusCode::kUnimplemented,
                             testing::HasSubstr("Don't know how to convert")));
}

TEST_F(TranslatorPointerTest, MethodUsingMemberReference) {
  const std::string content = R"(
       struct Test {
        int& ref;
        int amt;
        void add() {
          ref += amt;
        }
       };
       int my_package(int a) {
         Test s = {.ref = a, .amt = 3};
         s.add();
         return a;
       })";

  ASSERT_THAT(
      SourceToIr(content).status(),
      absl_testing::StatusIs(
          absl::StatusCode::kUnimplemented,
          testing::HasSubstr("Don't know how to create LValue for member")));
}

TEST_F(TranslatorPointerTest, ReferenceReturnWithIO) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    struct MyPtr {
      int val;

      MyPtr& getit(__xls_channel<int>& in) {
        val += in.read();
        return *this;
      }

      void inc() {
        ++val;
      }
    };

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      MyPtr my = {.val = 55};
      MyPtr& p = my.getit(in);
      p.inc();
      out.write(my.val);
    })";

  ASSERT_THAT(
      SourceToIr(content, /*pfunc=*/nullptr, /*clang_argv=*/{},
                 /*io_test_mode=*/true)
          .status(),
      absl_testing::StatusIs(
          absl::StatusCode::kUnimplemented,
          testing::HasSubstr("eferences to side effecting operations")));
}

TEST_F(TranslatorPointerTest, PipelinedLoopUsingReference) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      int val = in.read();
      int&ref = val;
      #pragma hls_pipeline_init_interval 1
      for(int i=0;i<3;++i) {
        ref += 5;
      }
      out.write(val);
    })";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* ch_in1 = block_spec.add_channels();
    ch_in1->set_name("in");
    ch_in1->set_is_input(true);
    ch_in1->set_type(FIFO);

    HLSChannel* ch_out1 = block_spec.add_channels();
    ch_out1->set_name("out");
    ch_out1->set_is_input(false);
    ch_out1->set_type(FIFO);
  }

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(3, 32))};

  absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
  outputs["out"] = {xls::Value(xls::SBits(3 + 5 * 3, 32))};

  ProcTest(content, block_spec, inputs, outputs);
}

TEST_F(TranslatorPointerTest, PipelinedLoopUsingReferenceAndIO) {
  const std::string content = R"(
    #include "/xls_builtin.h"

    #pragma hls_top
    void foo(__xls_channel<int>& in,
             __xls_channel<int>& out) {
      int val = 5;
      int&ref = val;
      #pragma hls_pipeline_init_interval 1
      for(int i=0;i<3;++i) {
        ref += in.read();
      }
      out.write(val);
    })";

  HLSBlock block_spec;
  {
    block_spec.set_name("foo");

    HLSChannel* ch_in1 = block_spec.add_channels();
    ch_in1->set_name("in");
    ch_in1->set_is_input(true);
    ch_in1->set_type(FIFO);

    HLSChannel* ch_out1 = block_spec.add_channels();
    ch_out1->set_name("out");
    ch_out1->set_is_input(false);
    ch_out1->set_type(FIFO);
  }

  absl::flat_hash_map<std::string, std::list<xls::Value>> inputs;
  inputs["in"] = {xls::Value(xls::SBits(1, 32)), xls::Value(xls::SBits(6, 32)),
                  xls::Value(xls::SBits(8, 32))};

  absl::flat_hash_map<std::string, std::list<xls::Value>> outputs;
  outputs["out"] = {xls::Value(xls::SBits(5 + 1 + 6 + 8, 32))};

  ProcTest(content, block_spec, inputs, outputs);
}

TEST_F(TranslatorPointerTest, ReferenceToTernary) {
  const std::string content = R"(
    int my_package(int a) {
      int x = 11, y = 15;
      int& r = a ? x : y;
      x = 100;
      return r;
    })";

  Run({{"a", 0}}, 15, content);
  Run({{"a", 1}}, 100, content);
}

TEST_F(TranslatorPointerTest, ReferenceToTernarySet) {
  const std::string content = R"(
    int my_package(int a) {
      int x = 11, y = 15;
      int& r = a ? x : y;
      r = 100;
      return x;
    })";

  Run({{"a", 0}}, 11, content);
  Run({{"a", 1}}, 100, content);
}

TEST_F(TranslatorPointerTest, ReferenceToTernarySetClass) {
  const std::string content = R"(  
    struct MyInt {
      MyInt(int val) : val_(val) {
      }
      
      operator int()const {
        return val_;
      }
      
      int val_;
    };

    int my_package(int a) {
      MyInt x = 11, y = 15;
      MyInt& r = a ? x : y;
      r = 100;
      return x;
    })";

  Run({{"a", 0}}, 11, content);
  Run({{"a", 1}}, 100, content);
}

TEST_F(TranslatorPointerTest, ReferenceToTernarySetClassNested) {
  const std::string content = R"(  
    struct MyInt {
      MyInt(int val) : val_(val) {
      }
      
      operator int()const {
        return val_;
      }
      
      int val_;
    };

    int my_package(int a, int b) {
      MyInt x = 11, y = 15, z = 5;
      MyInt& r = b ? (a ? x : y) : z;
      r = 100;
      return x;
    })";

  Run({{"a", 0}, {"b", 1}}, 11, content);
  Run({{"a", 1}, {"b", 1}}, 100, content);
  Run({{"a", 0}, {"b", 0}}, 11, content);
  Run({{"a", 1}, {"b", 0}}, 11, content);
}

TEST_F(TranslatorPointerTest, ReferenceToTernarySetClass3) {
  const std::string content = R"(  
    struct MyInt {
      MyInt(int val) : val_(val) {
      }
      
      operator int()const {
        return val_;
      }
      
      MyInt& operator+=(MyInt& o) {
        val_ += o.val_;
        return *this;
      }
      
      int val_;
    };

    int my_package(int a, int b) {
      MyInt x = 11, y = 15, z = 20;
      MyInt& r = b ? (a ? x : y) : z;
      x += r;
      return x;
    })";

  Run({{"a", 0}, {"b", 1}}, 11 + 15, content);
  Run({{"a", 1}, {"b", 1}}, 11 + 11, content);
  Run({{"a", 0}, {"b", 0}}, 11 + 20, content);
  Run({{"a", 1}, {"b", 0}}, 11 + 20, content);
}

TEST_F(TranslatorPointerTest, ResolveReferenceInTernary) {
  const std::string content = R"(
    int my_package(int a) {
      int x = 11, y = 15;
      int& yr = y;
      int r = a ? x : yr;
      return r;
    })";

  Run({{"a", 0}}, 15, content);
  Run({{"a", 1}}, 11, content);
}

TEST_F(TranslatorPointerTest, ResolveReferenceInTernary2) {
  const std::string content = R"(
    int my_package(int a) {
      int x = 11, y = 15;
      int& yr = y;
      int& r = a ? x : yr;
      x = 100;
      return r;
    })";

  Run({{"a", 0}}, 15, content);
  Run({{"a", 1}}, 100, content);
}

TEST_F(TranslatorPointerTest, AssignReferenceInTernaryDirect) {
  const std::string content = R"(
    int my_package(int a) {
      int x = 11, y = 15;
      int& xr = x;
      int& yr = y;
      (a ? xr : yr) = 100;
      return x;
    })";

  ASSERT_THAT(SourceToIr(content).status(),
              absl_testing::StatusIs(
                  absl::StatusCode::kUnimplemented,
                  testing::HasSubstr("Ternaries in lvalues only supported for "
                                     "pointers or references")));
}

TEST_F(TranslatorPointerTest, ResolveReferenceInTernary3) {
  const std::string content = R"(
    int my_package(int a) {
      int x = 11, y = 15;
      int& xr = x;
      int& yr = y;
      int& r = a ? xr : yr;
      x = 100;
      return r;
    })";

  Run({{"a", 0}}, 15, content);
  Run({{"a", 1}}, 100, content);
}

TEST_F(TranslatorPointerTest, SetPointerInMethod) {
  const std::string content = R"(
    struct MyPtr {
      int *p;
      void setit(int* xp) {
        p = xp;
      }
    };

    int my_package() {
      int arr[3] = {1, 2, 3};
      MyPtr ptr = {.p = &arr[1]};
      ptr.setit(&arr[2]);
      return ptr.p[1];
    })";

  ASSERT_THAT(
      SourceToIr(content).status(),
      absl_testing::StatusIs(
          absl::StatusCode::kUnimplemented,
          testing::HasSubstr("Don't know how to create LValue for member")));
}

TEST_F(TranslatorPointerTest, BaseClassReference) {
  const std::string content = R"(
    struct FooBase {
      int v;
      operator int()const {
        return v+1;
      }
    };
    struct Foo : public FooBase {
    };
    struct Bar {
      Foo y;
    };
    int Test(const Bar& ctrl) {
      const Foo& v = ctrl.y;
      return v;
    }
    #pragma hls_top
    int my_package(int a) {
      Bar bar;
      bar.y.v = a;
      return Test(bar);
    })";

  Run({{"a", 15}}, 16, content);
}

}  // namespace

}  // namespace xlscc
