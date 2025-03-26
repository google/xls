// Copyright 2025 The XLS Authors
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

#include "xls/common/status/status_or_ref.h"

#include <cstdint>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"

namespace xls {
namespace {
using absl_testing::StatusIs;
using testing::HasSubstr;
using testing::StrEq;

TEST(StatusOrRefTest, Construction) {
  int64_t x = 33;
  auto fun = [&]() -> StatusOrRef<int64_t> { return x; };
  XLS_ASSERT_OK_AND_ASSIGN(int64_t& y, fun());
  EXPECT_EQ(y, x);
  EXPECT_EQ(&y, &x);
  x = 44;
  EXPECT_EQ(y, x);
}

TEST(StatusOrRefTest, ConstConstruction) {
  int64_t x = 33;
  auto fun = [&]() -> StatusOrRef<const int64_t> { return x; };
  XLS_ASSERT_OK_AND_ASSIGN(const int64_t& y, fun());
  EXPECT_EQ(y, x);
  EXPECT_EQ(&y, &x);
}

TEST(StatusOrRefTest, Construction2) {
  int64_t x = 33;
  auto fun = [&](bool success) -> StatusOrRef<int64_t> {
    if (success) {
      return x;
    }
    return absl::InternalError("foobar");
  };
  XLS_ASSERT_OK_AND_ASSIGN(int64_t& y, fun(true));
  EXPECT_EQ(y, x);
  EXPECT_EQ(&y, &x);
  x = 44;
  EXPECT_EQ(y, x);

  EXPECT_THAT(fun(false).status(), StatusIs(absl::StatusCode::kInternal));
}

struct Foo {
  virtual ~Foo() = default;
  virtual std::string_view type() const { return "Foo"; }
  bool operator==(const Foo& f) const { return &f == this; }
  bool operator!=(const Foo& f) const { return &f != this; }
};
struct Bar : public Foo {
  std::string_view type() const override { return "Bar"; }
};

TEST(StatusOrRefTest, Cast) {
  Foo f;
  Bar b;
  auto fun = [&](bool foo) -> StatusOrRef<Foo> {
    if (foo) {
      return f;
    }
    return b;
  };
  XLS_ASSERT_OK_AND_ASSIGN(const Foo& y, fun(true));
  EXPECT_THAT(y.type(), StrEq("Foo"));
  XLS_ASSERT_OK_AND_ASSIGN(const Foo& z, fun(false));
  EXPECT_THAT(z.type(), StrEq("Bar"));
}

TEST(StatusOrRefTest, Eq) {
  Foo f;
  Foo g;
  auto fun = [&](bool foo) -> StatusOrRef<Foo> {
    if (foo) {
      return f;
    }
    return g;
  };
  ASSERT_EQ(fun(true), fun(true));
  ASSERT_NE(fun(false), fun(true));
}

TEST(StatusOrRefTest, EqErr) {
  Foo f;
  auto fun = [&](bool foo) -> StatusOrRef<Foo> {
    if (foo) {
      return f;
    }
    return absl::InternalError("not foo");
  };
  ASSERT_EQ(fun(true), fun(true));
  ASSERT_EQ(fun(false), fun(false));
  ASSERT_NE(fun(true), fun(false));
}

TEST(StatusOrRefTest, Macros) {
  auto fun = [&]() -> StatusOrRef<Foo> {
    return absl::InternalError("not foo");
  };
  auto fun2 = [&]() -> StatusOrRef<Foo> {
    XLS_ASSIGN_OR_RETURN(Foo & fr, fun(), _ << "; bar");
    return fr;
  };
  auto res = fun2();
  EXPECT_THAT(res, StatusIs(absl::StatusCode::kInternal, HasSubstr("bar")));
  auto fun3 = [&]() -> absl::Status {
    XLS_RETURN_IF_ERROR(fun().status()) << "; bar";
    return absl::InternalError("unexpected success");
  };
  auto res2 = fun3();
  EXPECT_THAT(res2, StatusIs(absl::StatusCode::kInternal, HasSubstr("bar")));
}

#ifdef INCLUDE_UNCOMPILABLE
TEST(StatusOrRefTest, BadResult) {
  // This won't compile because of a lifetime warning.
  //
  // Something like:
  //
  // error: address of stack memory associated with local variable 'res'
  // returned [-Werror,-Wreturn-stack-address]
  auto fun = [&]() -> StatusOrRef<Foo> {
    Foo f;

    StatusOrRef<Foo> res(f);
    return res;
  };
  fun().IgnoreError();
}
#endif  // INCLUDE_UNCOMPILABLE

}  // namespace
}  // namespace xls
