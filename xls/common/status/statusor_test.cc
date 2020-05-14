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

#include "xls/common/status/statusor.h"

#include <errno.h>

#include <initializer_list>
#include <memory>
#include <type_traits>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/base/casts.h"
#include "absl/memory/memory.h"
#include "absl/types/any.h"
#include "absl/types/variant.h"
#include "absl/utility/utility.h"
#include "xls/common/source_location.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_builder.h"

namespace xabsl {
namespace {

using absl::Status;
using ::testing::AllOf;
using ::testing::AnyWith;
using ::testing::ElementsAre;
using ::testing::Field;
using ::testing::HasSubstr;
using ::testing::Ne;
using ::testing::Not;
using ::testing::Pointee;
using ::testing::VariantWith;
using ::xls::status_testing::HasErrorCode;
using ::xls::status_testing::IsOk;
using ::xls::status_testing::IsOkAndHolds;
using ::xls::status_testing::StatusIs;

struct CopyDetector {
  CopyDetector() = default;
  explicit CopyDetector(int xx) : x(xx) {}
  CopyDetector(CopyDetector&& d) noexcept
      : x(d.x), copied(false), moved(true) {}
  CopyDetector(const CopyDetector& d) : x(d.x), copied(true), moved(false) {}
  CopyDetector& operator=(const CopyDetector& c) {
    x = c.x;
    copied = true;
    moved = false;
    return *this;
  }
  CopyDetector& operator=(CopyDetector&& c) noexcept {
    x = c.x;
    copied = false;
    moved = true;
    return *this;
  }
  int x = 0;
  bool copied = false;
  bool moved = false;
};

testing::Matcher<const CopyDetector&> CopyDetectorHas(int a, bool b, bool c) {
  return AllOf(Field(&CopyDetector::x, a), Field(&CopyDetector::moved, b),
               Field(&CopyDetector::copied, c));
}

class Base1 {
 public:
  virtual ~Base1() {}
  int pad_;
};

class Base2 {
 public:
  virtual ~Base2() {}
  int yetotherpad_;
};

class Derived : public Base1, public Base2 {
 public:
  ~Derived() override {}
  int evenmorepad_;
};

class CopyNoAssign {
 public:
  explicit CopyNoAssign(int value) : foo_(value) {}
  CopyNoAssign(const CopyNoAssign& other) : foo_(other.foo_) {}
  int foo_;

 private:
  const CopyNoAssign& operator=(const CopyNoAssign&);
};

StatusOr<std::unique_ptr<int>> ReturnUniquePtr() {
  // Uses implicit constructor from T&&
  return absl::make_unique<int>(0);
}

TEST(StatusOr, ElementType) {
  static_assert(std::is_same<StatusOr<int>::element_type, int>(), "");
  static_assert(std::is_same<StatusOr<char>::element_type, char>(), "");
}

TEST(StatusOr, TestMoveOnlyInitialization) {
  StatusOr<std::unique_ptr<int>> thing(ReturnUniquePtr());
  ASSERT_TRUE(thing.ok());
  EXPECT_EQ(0, *thing.value());
  int* previous = thing.value().get();

  thing = ReturnUniquePtr();
  EXPECT_TRUE(thing.ok());
  EXPECT_EQ(0, *thing.value());
  EXPECT_NE(previous, thing.value().get());
}

TEST(StatusOr, TestMoveOnlyValueExtraction) {
  StatusOr<std::unique_ptr<int>> thing(ReturnUniquePtr());
  ASSERT_TRUE(thing.ok());
  std::unique_ptr<int> ptr = std::move(thing).value();
  EXPECT_EQ(0, *ptr);

  thing = std::move(ptr);
  ptr = std::move(thing.value());
  EXPECT_EQ(0, *ptr);
}

TEST(StatusOr, TestMoveOnlyInitializationFromTemporaryByValueOrDie) {
  std::unique_ptr<int> ptr(ReturnUniquePtr().value());
  EXPECT_EQ(0, *ptr);
}

TEST(StatusOr, TestValueOrDieOverloadForConstTemporary) {
  static_assert(
      std::is_same<const int&&,
                   decltype(std::declval<const StatusOr<int>&&>().value())>(),
      "value() for const temporaries should return const T&&");
}

TEST(StatusOr, TestMoveOnlyConversion) {
  StatusOr<std::unique_ptr<const int>> const_thing(ReturnUniquePtr());
  EXPECT_TRUE(const_thing.ok());
  EXPECT_EQ(0, *const_thing.value());

  // Test rvalue converting assignment
  const int* const_previous = const_thing.value().get();
  const_thing = ReturnUniquePtr();
  EXPECT_TRUE(const_thing.ok());
  EXPECT_EQ(0, *const_thing.value());
  EXPECT_NE(const_previous, const_thing.value().get());
}

TEST(StatusOr, TestMoveOnlyVector) {
  // Sanity check that StatusOr<MoveOnly> works in vector.
  std::vector<StatusOr<std::unique_ptr<int>>> vec;
  vec.push_back(ReturnUniquePtr());
  vec.resize(2);
  auto another_vec = std::move(vec);
  EXPECT_EQ(0, *another_vec[0].value());
  EXPECT_EQ(absl::UnknownError(""), another_vec[1].status());
}

TEST(StatusOr, TestDefaultCtor) {
  StatusOr<int> thing;
  EXPECT_FALSE(thing.ok());
  EXPECT_TRUE(HasErrorCode(thing.status(), absl::StatusCode::kUnknown));
}

#if GTEST_HAS_DEATH_TEST
TEST(StatusOrDeathTest, TestDefaultCtorValue) {
  StatusOr<int> thing;
  EXPECT_DEATH(thing.value(), "error UNKNOWN");

  const StatusOr<int> thing2;
  EXPECT_DEATH(thing.value(), "error UNKNOWN");
}

TEST(StatusOrDeathTest, TestValueNotOk) {
  StatusOr<int> thing(absl::CancelledError());
  EXPECT_DEATH(thing.value(), "error CANCELLED");
}

TEST(StatusOrDeathTest, TestValueNotOkConst) {
  const StatusOr<int> thing(absl::UnknownError(""));
  EXPECT_DEATH(thing.value(), "error UNKNOWN");
}

TEST(StatusOrDeathTest, TestPointerDefaultCtorValue) {
  StatusOr<int*> thing;
  EXPECT_DEATH(thing.value(), "error UNKNOWN");
}

TEST(StatusOrDeathTest, TestPointerValueNotOk) {
  StatusOr<int*> thing(absl::CancelledError());
  EXPECT_DEATH(thing.value(), "error CANCELLED");
}

TEST(StatusOrDeathTest, TestPointerValueNotOkConst) {
  const StatusOr<int*> thing(absl::CancelledError());
  EXPECT_DEATH(thing.value(), "error CANCELLED");
}

TEST(StatusOrDeathTest, TestStatusCtorStatusOk) {
  EXPECT_DEBUG_DEATH(
      {
        // This will DCHECK
        StatusOr<int> thing(absl::OkStatus());
        // In optimized mode, we are actually going to get error::INTERNAL for
        // status here, rather than crashing, so check that.
        EXPECT_FALSE(thing.ok());
        EXPECT_TRUE(HasErrorCode(thing.status(), absl::StatusCode::kInternal));
      },
      "An OK status is not a valid constructor argument");
}

TEST(StatusOrDeathTest, TestPointerStatusCtorStatusOk) {
  EXPECT_DEBUG_DEATH(
      {
        StatusOr<int*> thing(absl::OkStatus());
        // In optimized mode, we are actually going to get error::INTERNAL for
        // status here, rather than crashing, so check that.
        EXPECT_FALSE(thing.ok());
        EXPECT_TRUE(HasErrorCode(thing.status(), absl::StatusCode::kInternal));
      },
      "An OK status is not a valid constructor argument");
}
#endif

TEST(StatusOr, ValueAccessor) {
  const int kIntValue = 110;
  {
    StatusOr<int> status_or(kIntValue);
    EXPECT_EQ(kIntValue, status_or.value());
    EXPECT_EQ(kIntValue, std::move(status_or).value());
  }
  {
    StatusOr<CopyDetector> status_or(kIntValue);
    EXPECT_THAT(status_or,
                IsOkAndHolds(CopyDetectorHas(kIntValue, false, false)));
    CopyDetector copy_detector = status_or.value();
    EXPECT_THAT(copy_detector, CopyDetectorHas(kIntValue, false, true));
    copy_detector = std::move(status_or).value();
    EXPECT_THAT(copy_detector, CopyDetectorHas(kIntValue, true, false));
  }
}

TEST(StatusOr, BadValueAccess) {
  const absl::Status kError = absl::CancelledError("message");
  StatusOr<int> status_or(kError);
  EXPECT_DEATH_IF_SUPPORTED(status_or.value(), kError.ToString());
}

TEST(StatusOr, TestStatusCtor) {
  StatusOr<int> thing(absl::CancelledError());
  EXPECT_FALSE(thing.ok());
  EXPECT_TRUE(HasErrorCode(thing.status(), absl::StatusCode::kCancelled));
}

TEST(StatusOr, TestValueCtor) {
  const int kI = 4;
  const StatusOr<int> thing(kI);
  EXPECT_TRUE(thing.ok());
  EXPECT_EQ(kI, thing.value());
}

struct Foo {
  const int x;
  explicit Foo(int y) : x(y) {}
};

TEST(StatusOr, InPlaceConstruction) {
  EXPECT_THAT(xabsl::StatusOr<Foo>(absl::in_place, 10),
              IsOkAndHolds(Field(&Foo::x, 10)));
}

struct InPlaceHelper {
  InPlaceHelper(std::initializer_list<int> xs, std::unique_ptr<int> yy)
      : x(xs), y(std::move(yy)) {}
  const std::vector<int> x;
  std::unique_ptr<int> y;
};

TEST(StatusOr, InPlaceInitListConstruction) {
  xabsl::StatusOr<InPlaceHelper> status_or(absl::in_place, {10, 11, 12},
                                           absl::make_unique<int>(13));
  EXPECT_THAT(status_or, IsOkAndHolds(AllOf(
                             Field(&InPlaceHelper::x, ElementsAre(10, 11, 12)),
                             Field(&InPlaceHelper::y, Pointee(13)))));
}

TEST(StatusOr, Emplace) {
  xabsl::StatusOr<Foo> status_or_foo(10);
  status_or_foo.emplace(20);
  EXPECT_THAT(status_or_foo, IsOkAndHolds(Field(&Foo::x, 20)));
  status_or_foo = absl::InvalidArgumentError("msg");
  EXPECT_THAT(status_or_foo,
              StatusIs(absl::StatusCode::kInvalidArgument, "msg"));
  status_or_foo.emplace(20);
  EXPECT_THAT(status_or_foo, IsOkAndHolds(Field(&Foo::x, 20)));
}

TEST(StatusOr, EmplaceInitializerList) {
  xabsl::StatusOr<InPlaceHelper> status_or(absl::in_place, {10, 11, 12},
                                           absl::make_unique<int>(13));
  status_or.emplace({1, 2, 3}, absl::make_unique<int>(4));
  EXPECT_THAT(status_or,
              IsOkAndHolds(AllOf(Field(&InPlaceHelper::x, ElementsAre(1, 2, 3)),
                                 Field(&InPlaceHelper::y, Pointee(4)))));
  status_or = absl::InvalidArgumentError("msg");
  EXPECT_THAT(status_or, StatusIs(absl::StatusCode::kInvalidArgument, "msg"));
  status_or.emplace({1, 2, 3}, absl::make_unique<int>(4));
  EXPECT_THAT(status_or,
              IsOkAndHolds(AllOf(Field(&InPlaceHelper::x, ElementsAre(1, 2, 3)),
                                 Field(&InPlaceHelper::y, Pointee(4)))));
}

TEST(StatusOr, TestCopyCtorStatusOk) {
  const int kI = 4;
  const StatusOr<int> original(kI);
  const StatusOr<int> copy(original);
  XLS_EXPECT_OK(copy.status());
  EXPECT_EQ(original.value(), copy.value());
}

TEST(StatusOr, TestCopyCtorStatusNotOk) {
  StatusOr<int> original(absl::CancelledError());
  StatusOr<int> copy(original);
  EXPECT_TRUE(HasErrorCode(copy.status(), absl::StatusCode::kCancelled));
}

TEST(StatusOr, TestCopyCtorNonAssignable) {
  const int kI = 4;
  CopyNoAssign value(kI);
  StatusOr<CopyNoAssign> original(value);
  StatusOr<CopyNoAssign> copy(original);
  XLS_EXPECT_OK(copy.status());
  EXPECT_EQ(original.value().foo_, copy.value().foo_);
}

TEST(StatusOr, TestCopyCtorStatusOKConverting) {
  const int kI = 4;
  StatusOr<int> original(kI);
  StatusOr<double> copy(original);
  XLS_EXPECT_OK(copy.status());
  EXPECT_DOUBLE_EQ(original.value(), copy.value());
}

TEST(StatusOr, TestCopyCtorStatusNotOkConverting) {
  StatusOr<int> original(absl::CancelledError());
  StatusOr<double> copy(original);
  EXPECT_EQ(copy.status(), original.status());
}

TEST(StatusOr, TestAssignmentStatusOk) {
  // Copy assignmment
  {
    const auto p = std::make_shared<int>(17);
    StatusOr<std::shared_ptr<int>> source(p);

    StatusOr<std::shared_ptr<int>> target;
    target = source;

    ASSERT_TRUE(target.ok());
    XLS_EXPECT_OK(target.status());
    EXPECT_EQ(p, target.value());

    ASSERT_TRUE(source.ok());
    XLS_EXPECT_OK(source.status());
    EXPECT_EQ(p, source.value());
  }

  // Move asssignment
  {
    const auto p = std::make_shared<int>(17);
    StatusOr<std::shared_ptr<int>> source(p);

    StatusOr<std::shared_ptr<int>> target;
    target = std::move(source);

    ASSERT_TRUE(target.ok());
    XLS_EXPECT_OK(target.status());
    EXPECT_EQ(p, target.value());

    ASSERT_TRUE(source.ok());  // NOLINT
    XLS_EXPECT_OK(source.status());
    EXPECT_EQ(nullptr, source.value());
  }
}

TEST(StatusOr, TestAssignmentStatusNotOk) {
  // Copy assignment
  {
    const absl::Status expected = absl::CancelledError();
    StatusOr<int> source(expected);

    StatusOr<int> target;
    target = source;

    EXPECT_FALSE(target.ok());
    EXPECT_EQ(expected, target.status());

    EXPECT_FALSE(source.ok());
    EXPECT_EQ(expected, source.status());
  }

  // Move assignment
  {
    const absl::Status expected = absl::CancelledError();
    StatusOr<int> source(expected);

    StatusOr<int> target;
    target = std::move(source);

    EXPECT_FALSE(target.ok());
    EXPECT_EQ(expected, target.status());

    EXPECT_FALSE(source.ok());  // NOLINT
    EXPECT_THAT(source, StatusIs(absl::StatusCode::kInternal));
  }
}

TEST(StatusOr, TestAssignmentStatusOKConverting) {
  // Copy assignment
  {
    const int kI = 4;
    StatusOr<int> source(kI);

    StatusOr<double> target;
    target = source;

    ASSERT_TRUE(target.ok());
    XLS_EXPECT_OK(target.status());
    EXPECT_DOUBLE_EQ(kI, target.value());

    ASSERT_TRUE(source.ok());
    XLS_EXPECT_OK(source.status());
    EXPECT_DOUBLE_EQ(kI, source.value());
  }

  // Move assignment
  {
    const auto p = new int(17);
    StatusOr<std::unique_ptr<int>> source(absl::WrapUnique(p));

    StatusOr<std::shared_ptr<int>> target;
    target = std::move(source);

    ASSERT_TRUE(target.ok());
    XLS_EXPECT_OK(target.status());
    EXPECT_EQ(p, target.value().get());

    ASSERT_TRUE(source.ok());  // NOLINT
    XLS_EXPECT_OK(source.status());
    EXPECT_EQ(nullptr, source.value().get());
  }
}

TEST(StatusOr, ImplicitConvertingConstructor) {
  struct A {
    int x;
  };
  struct B {
    int x;
    bool moved;
    B(const A& a)  // NOLINT
    : x(a.x), moved(false) {}
    B(A&& a)  // NOLINT
    : x(a.x), moved(true) {}
  };
  EXPECT_THAT(
      absl::implicit_cast<xabsl::StatusOr<B>>(xabsl::StatusOr<A>(A{11})),
      IsOkAndHolds(AllOf(Field(&B::x, 11), Field(&B::moved, true))));
  xabsl::StatusOr<A> a(A{12});
  EXPECT_THAT(absl::implicit_cast<xabsl::StatusOr<B>>(a),
              IsOkAndHolds(AllOf(Field(&B::x, 12), Field(&B::moved, false))));
}

TEST(StatusOr, ExplicitConvertingConstructor) {
  struct A {
    int x;
  };
  struct B {
    int x;
    bool moved;
    explicit B(const A& a) : x(a.x), moved(false) {}
    explicit B(A&& a) : x(a.x), moved(true) {}
  };
  EXPECT_FALSE((std::is_convertible<const xabsl::StatusOr<A>&,
                                    xabsl::StatusOr<B>>::value));
  EXPECT_FALSE(
      (std::is_convertible<xabsl::StatusOr<A>&&, xabsl::StatusOr<B>>::value));
  EXPECT_THAT(xabsl::StatusOr<B>(xabsl::StatusOr<A>(A{11})),
              IsOkAndHolds(AllOf(Field(&B::x, 11), Field(&B::moved, true))));
  xabsl::StatusOr<A> a(A{12});
  EXPECT_THAT(xabsl::StatusOr<B>(a),
              IsOkAndHolds(AllOf(Field(&B::x, 12), Field(&B::moved, false))));
}

TEST(StatusOr, ImplicitBooleanConstructionWithImplicitCasts) {
  struct A {
    A(bool y) : x(y) {}  // NOLINT
    bool x = false;
  };
  struct B {
    explicit B(bool y) : x(y) {}
    operator bool() const { return x; }  // NOLINT
    bool x = false;
  };
  EXPECT_THAT(
      absl::implicit_cast<xabsl::StatusOr<A>>(xabsl::StatusOr<bool>(false)),
      IsOkAndHolds(Field(&A::x, false)));
  EXPECT_FALSE(
      (std::is_convertible<xabsl::StatusOr<B>, xabsl::StatusOr<A>>::value));
}

TEST(StatusOr, BooleanConstructionWithImplicitCasts) {
  struct A {
    A(bool y) : x(y) {}  // NOLINT
    bool x = false;
  };
  struct B {
    explicit B(bool y) : x(y) {}
    operator bool() const { return x; }  // NOLINT
    bool x = false;
  };

  // Conversion to bool or const bool interpret the status.
  EXPECT_THAT(xabsl::StatusOr<A>{xabsl::StatusOr<bool>(false)},
              IsOkAndHolds(Field(&A::x, false)));
  EXPECT_THAT(
      xabsl::StatusOr<A>{xabsl::StatusOr<bool>(absl::InvalidArgumentError(""))},
      Not(IsOk()));

  EXPECT_THAT(xabsl::StatusOr<A>{xabsl::StatusOr<B>(B{false})},
              IsOkAndHolds(Field(&A::x, false)));
  EXPECT_THAT(
      xabsl::StatusOr<A>{xabsl::StatusOr<B>(absl::InvalidArgumentError(""))},
      Not(IsOk()));
}

TEST(StatusOr, ConstImplicitCast) {
  EXPECT_THAT(absl::implicit_cast<xabsl::StatusOr<const std::string>>(
                  xabsl::StatusOr<std::string>("foo")),
              IsOkAndHolds("foo"));
  EXPECT_THAT(absl::implicit_cast<xabsl::StatusOr<std::string>>(
                  xabsl::StatusOr<const std::string>("foo")),
              IsOkAndHolds("foo"));
  EXPECT_THAT(
      absl::implicit_cast<xabsl::StatusOr<std::shared_ptr<const std::string>>>(
          xabsl::StatusOr<std::shared_ptr<std::string>>(
              std::make_shared<std::string>("foo"))),
      IsOkAndHolds(Pointee(std::string("foo"))));
}

TEST(StatusOr, ExplicitConstruction) {
  struct Foo {
    int x;
    explicit Foo(int y) : x(y) {}
  };
  EXPECT_THAT(xabsl::StatusOr<Foo>(10), IsOkAndHolds(Field(&Foo::x, 10)));
}

TEST(StatusOr, ConstCast) {
  EXPECT_THAT(absl::implicit_cast<xabsl::StatusOr<const std::string>>(
                  xabsl::StatusOr<std::string>("foo")),
              IsOkAndHolds("foo"));
  EXPECT_THAT(absl::implicit_cast<xabsl::StatusOr<std::string>>(
                  xabsl::StatusOr<const std::string>("foo")),
              IsOkAndHolds("foo"));
  EXPECT_THAT(
      absl::implicit_cast<xabsl::StatusOr<std::shared_ptr<const std::string>>>(
          xabsl::StatusOr<std::shared_ptr<std::string>>(
              std::make_shared<std::string>("foo"))),
      IsOkAndHolds(Pointee(std::string("foo"))));
}

TEST(StatusOr, ImplicitConstruction) {
  // Check implicit casting works.
  auto status_or =
      absl::implicit_cast<xabsl::StatusOr<absl::variant<int, std::string>>>(10);
  EXPECT_THAT(status_or, IsOkAndHolds(VariantWith<int>(10)));
}

TEST(StatusOr, ImplicitConstructionFromInitliazerList) {
  // Note: dropping the explicit std::initializer_list<int> is not supported
  // by xabsl::StatusOr or absl::optional.
  auto status_or =
      absl::implicit_cast<xabsl::StatusOr<std::vector<int>>>({{10, 20, 30}});
  EXPECT_THAT(status_or, IsOkAndHolds(ElementsAre(10, 20, 30)));
}

TEST(StatusOr, UniquePtrImplicitConstruction) {
  auto status_or = absl::implicit_cast<xabsl::StatusOr<std::unique_ptr<Base1>>>(
      absl::make_unique<Derived>());
  EXPECT_THAT(status_or, IsOkAndHolds(Ne(nullptr)));
}

TEST(StatusOr, NestedStatusOrCopyAndMoveConstructorTests) {
  xabsl::StatusOr<xabsl::StatusOr<CopyDetector>> status_or = CopyDetector(10);
  xabsl::StatusOr<xabsl::StatusOr<CopyDetector>> status_error =
      absl::InvalidArgumentError("foo");
  EXPECT_THAT(status_or,
              IsOkAndHolds(IsOkAndHolds(CopyDetectorHas(10, true, false))));
  xabsl::StatusOr<xabsl::StatusOr<CopyDetector>> a = status_or;
  EXPECT_THAT(a, IsOkAndHolds(IsOkAndHolds(CopyDetectorHas(10, false, true))));
  xabsl::StatusOr<xabsl::StatusOr<CopyDetector>> a_err = status_error;
  EXPECT_THAT(a_err, Not(IsOk()));

  const xabsl::StatusOr<xabsl::StatusOr<CopyDetector>>& cref = status_or;
  xabsl::StatusOr<xabsl::StatusOr<CopyDetector>> b = cref;  // NOLINT
  EXPECT_THAT(b, IsOkAndHolds(IsOkAndHolds(CopyDetectorHas(10, false, true))));
  const xabsl::StatusOr<xabsl::StatusOr<CopyDetector>>& cref_err = status_error;
  xabsl::StatusOr<xabsl::StatusOr<CopyDetector>> b_err = cref_err;  // NOLINT
  EXPECT_THAT(b_err, Not(IsOk()));

  xabsl::StatusOr<xabsl::StatusOr<CopyDetector>> c = std::move(status_or);
  EXPECT_THAT(c, IsOkAndHolds(IsOkAndHolds(CopyDetectorHas(10, true, false))));
  xabsl::StatusOr<xabsl::StatusOr<CopyDetector>> c_err =
      std::move(status_error);
  EXPECT_THAT(c_err, Not(IsOk()));
}

TEST(StatusOr, NestedStatusOrCopyAndMoveAssignment) {
  xabsl::StatusOr<xabsl::StatusOr<CopyDetector>> status_or = CopyDetector(10);
  xabsl::StatusOr<xabsl::StatusOr<CopyDetector>> status_error =
      absl::InvalidArgumentError("foo");
  xabsl::StatusOr<xabsl::StatusOr<CopyDetector>> a;
  a = status_or;
  EXPECT_THAT(a, IsOkAndHolds(IsOkAndHolds(CopyDetectorHas(10, false, true))));
  a = status_error;
  EXPECT_THAT(a, Not(IsOk()));

  const xabsl::StatusOr<xabsl::StatusOr<CopyDetector>>& cref = status_or;
  a = cref;
  EXPECT_THAT(a, IsOkAndHolds(IsOkAndHolds(CopyDetectorHas(10, false, true))));
  const xabsl::StatusOr<xabsl::StatusOr<CopyDetector>>& cref_err = status_error;
  a = cref_err;
  EXPECT_THAT(a, Not(IsOk()));
  a = std::move(status_or);
  EXPECT_THAT(a, IsOkAndHolds(IsOkAndHolds(CopyDetectorHas(10, true, false))));
  a = std::move(status_error);
  EXPECT_THAT(a, Not(IsOk()));
}

TEST(StatusOr, StatusOrCopyAndMoveTestsConstructor) {
  xabsl::StatusOr<CopyDetector> status_or(10);
  ASSERT_THAT(status_or, IsOkAndHolds(CopyDetectorHas(10, false, false)));
  xabsl::StatusOr<CopyDetector> a(status_or);
  EXPECT_THAT(a, IsOkAndHolds(CopyDetectorHas(10, false, true)));
  const xabsl::StatusOr<CopyDetector>& cref = status_or;
  xabsl::StatusOr<CopyDetector> b(cref);  // NOLINT
  EXPECT_THAT(b, IsOkAndHolds(CopyDetectorHas(10, false, true)));
  xabsl::StatusOr<CopyDetector> c(std::move(status_or));
  EXPECT_THAT(c, IsOkAndHolds(CopyDetectorHas(10, true, false)));
}

TEST(StatusOr, StatusOrCopyAndMoveTestsAssignment) {
  xabsl::StatusOr<CopyDetector> status_or(10);
  ASSERT_THAT(status_or, IsOkAndHolds(CopyDetectorHas(10, false, false)));
  xabsl::StatusOr<CopyDetector> a;
  a = status_or;
  EXPECT_THAT(a, IsOkAndHolds(CopyDetectorHas(10, false, true)));
  const xabsl::StatusOr<CopyDetector>& cref = status_or;
  xabsl::StatusOr<CopyDetector> b;
  b = cref;
  EXPECT_THAT(b, IsOkAndHolds(CopyDetectorHas(10, false, true)));
  xabsl::StatusOr<CopyDetector> c;
  c = std::move(status_or);
  EXPECT_THAT(c, IsOkAndHolds(CopyDetectorHas(10, true, false)));
}

TEST(StatusOr, AbslAnyAssignment) {
  EXPECT_FALSE((std::is_assignable<xabsl::StatusOr<absl::any>,
                                   xabsl::StatusOr<int>>::value));
  xabsl::StatusOr<absl::any> status_or;
  status_or = absl::InvalidArgumentError("foo");
  EXPECT_THAT(status_or, Not(IsOk()));
}

TEST(StatusOr, ImplicitAssignment) {
  xabsl::StatusOr<absl::variant<int, std::string>> status_or;
  status_or = 10;
  EXPECT_THAT(status_or, IsOkAndHolds(VariantWith<int>(10)));
}

TEST(StatusOr, SelfDirectInitAssignment) {
  xabsl::StatusOr<std::vector<int>> status_or = {{10, 20, 30}};
  status_or = *status_or;
  EXPECT_THAT(status_or, IsOkAndHolds(ElementsAre(10, 20, 30)));
}

TEST(StatusOr, ImplicitCastFromInitializerList) {
  xabsl::StatusOr<std::vector<int>> status_or = {{10, 20, 30}};
  EXPECT_THAT(status_or, IsOkAndHolds(ElementsAre(10, 20, 30)));
}

TEST(StatusOr, UniquePtrImplicitAssignment) {
  xabsl::StatusOr<std::unique_ptr<Base1>> status_or;
  status_or = absl::make_unique<Derived>();
  EXPECT_THAT(status_or, IsOkAndHolds(Ne(nullptr)));
}

TEST(StatusOr, Pointer) {
  struct A {};
  struct B : public A {};
  struct C : private A {};

  EXPECT_TRUE((std::is_constructible<xabsl::StatusOr<A*>, B*>::value));
}

TEST(StatusOr, TestAssignmentStatusNotOkConverting) {
  // Copy assignment
  {
    const absl::Status expected = absl::CancelledError();
    StatusOr<int> source(expected);

    StatusOr<double> target;
    target = source;

    EXPECT_FALSE(target.ok());
    EXPECT_EQ(expected, target.status());

    EXPECT_FALSE(source.ok());
    EXPECT_EQ(expected, source.status());
  }

  // Move assignment
  {
    const absl::Status expected = absl::CancelledError();
    StatusOr<int> source(expected);

    StatusOr<double> target;
    target = std::move(source);

    EXPECT_FALSE(target.ok());
    EXPECT_EQ(expected, target.status());

    EXPECT_FALSE(source.ok());  // NOLINT
    EXPECT_THAT(source, StatusIs(absl::StatusCode::kInternal));
  }
}

TEST(StatusOr, SelfAssignment) {
  // Copy-assignment, status OK
  {
    // A string long enough that it's likely to defeat any inline representation
    // optimization.
    const std::string long_str(128, 'a');

    StatusOr<std::string> so = long_str;
    so = *&so;

    ASSERT_TRUE(so.ok());
    XLS_EXPECT_OK(so.status());
    EXPECT_EQ(long_str, so.value());
  }

  // Copy-assignment, error status
  {
    StatusOr<int> so = absl::NotFoundError("taco");
    so = *&so;

    EXPECT_FALSE(so.ok());
    EXPECT_THAT(so, StatusIs(absl::StatusCode::kNotFound, "taco"));
  }

  // Move-assignment with copyable type, status OK
  {
    StatusOr<int> so = 17;

    // Fool the compiler, which otherwise complains.
    auto& same = so;
    so = std::move(same);

    ASSERT_TRUE(so.ok());
    XLS_EXPECT_OK(so.status());
    EXPECT_EQ(17, so.value());
  }

  // Move-assignment with copyable type, error status
  {
    StatusOr<int> so = absl::NotFoundError("taco");

    // Fool the compiler, which otherwise complains.
    auto& same = so;
    so = std::move(same);

    EXPECT_FALSE(so.ok());
    EXPECT_THAT(so, StatusIs(absl::StatusCode::kNotFound, "taco"));
  }

  // Move-assignment with non-copyable type, status OK
  {
    const auto raw = new int(17);
    StatusOr<std::unique_ptr<int>> so = absl::WrapUnique(raw);

    // Fool the compiler, which otherwise complains.
    auto& same = so;
    so = std::move(same);

    ASSERT_TRUE(so.ok());
    XLS_EXPECT_OK(so.status());
    EXPECT_EQ(raw, so.value().get());
  }

  // Move-assignment with non-copyable type, error status
  {
    StatusOr<std::unique_ptr<int>> so = absl::NotFoundError("taco");

    // Fool the compiler, which otherwise complains.
    auto& same = so;
    so = std::move(same);

    EXPECT_FALSE(so.ok());
    EXPECT_THAT(so, StatusIs(absl::StatusCode::kNotFound, "taco"));
  }
}

// These types form the overload sets of the constructors and the assignment
// operators of `MockValue`. They distinguish construction from assignment,
// lvalue from rvalue.
struct FromConstructibleAssignableLvalue {};
struct FromConstructibleAssignableRvalue {};
struct FromImplicitConstructibleOnly {};
struct FromAssignableOnly {};

// This class is for testing the forwarding value assignments of `StatusOr`.
// `from_rvalue` indicates whether the constructor or the assignment taking
// rvalue reference is called. `from_assignment` indicates whether any
// assignment is called.
struct MockValue {
  // Constructs `MockValue` from `FromConstructibleAssignableLvalue`.
  MockValue(const FromConstructibleAssignableLvalue&)  // NOLINT
      : from_rvalue(false), assigned(false) {}
  // Constructs `MockValue` from `FromConstructibleAssignableRvalue`.
  MockValue(FromConstructibleAssignableRvalue&&)  // NOLINT
      : from_rvalue(true), assigned(false) {}
  // Constructs `MockValue` from `FromImplicitConstructibleOnly`.
  // `MockValue` is not assignable from `FromImplicitConstructibleOnly`.
  MockValue(const FromImplicitConstructibleOnly&)  // NOLINT
      : from_rvalue(false), assigned(false) {}
  // Assigns `FromConstructibleAssignableLvalue`.
  MockValue& operator=(const FromConstructibleAssignableLvalue&) {
    from_rvalue = false;
    assigned = true;
    return *this;
  }
  // Assigns `FromConstructibleAssignableRvalue` (rvalue only).
  MockValue& operator=(FromConstructibleAssignableRvalue&&) {
    from_rvalue = true;
    assigned = true;
    return *this;
  }
  // Assigns `FromAssignableOnly`, but not constructible from
  // `FromAssignableOnly`.
  MockValue& operator=(const FromAssignableOnly&) {
    from_rvalue = false;
    assigned = true;
    return *this;
  }
  bool from_rvalue;
  bool assigned;
};

// operator=(U&&)
TEST(StatusOr, PerfectForwardingAssignment) {
  // U == T
  constexpr int kValue1 = 10, kValue2 = 20;
  xabsl::StatusOr<CopyDetector> status_or;
  CopyDetector lvalue(kValue1);
  status_or = lvalue;
  EXPECT_THAT(status_or, IsOkAndHolds(CopyDetectorHas(kValue1, false, true)));
  status_or = CopyDetector(kValue2);
  EXPECT_THAT(status_or, IsOkAndHolds(CopyDetectorHas(kValue2, true, false)));

  // U != T
  EXPECT_TRUE(
      (std::is_assignable<xabsl::StatusOr<MockValue>&,
                          const FromConstructibleAssignableLvalue&>::value));
  EXPECT_TRUE((std::is_assignable<xabsl::StatusOr<MockValue>&,
                                  FromConstructibleAssignableLvalue&&>::value));
  EXPECT_FALSE(
      (std::is_assignable<xabsl::StatusOr<MockValue>&,
                          const FromConstructibleAssignableRvalue&>::value));
  EXPECT_TRUE((std::is_assignable<xabsl::StatusOr<MockValue>&,
                                  FromConstructibleAssignableRvalue&&>::value));
  EXPECT_TRUE(
      (std::is_assignable<xabsl::StatusOr<MockValue>&,
                          const FromImplicitConstructibleOnly&>::value));
  EXPECT_FALSE((std::is_assignable<xabsl::StatusOr<MockValue>&,
                                   const FromAssignableOnly&>::value));

  xabsl::StatusOr<MockValue> from_lvalue(FromConstructibleAssignableLvalue{});
  EXPECT_FALSE(from_lvalue->from_rvalue);
  EXPECT_FALSE(from_lvalue->assigned);
  from_lvalue = FromConstructibleAssignableLvalue{};
  EXPECT_FALSE(from_lvalue->from_rvalue);
  EXPECT_TRUE(from_lvalue->assigned);

  xabsl::StatusOr<MockValue> from_rvalue(FromConstructibleAssignableRvalue{});
  EXPECT_TRUE(from_rvalue->from_rvalue);
  EXPECT_FALSE(from_rvalue->assigned);
  from_rvalue = FromConstructibleAssignableRvalue{};
  EXPECT_TRUE(from_rvalue->from_rvalue);
  EXPECT_TRUE(from_rvalue->assigned);

  xabsl::StatusOr<MockValue> from_implicit_constructible(
      FromImplicitConstructibleOnly{});
  EXPECT_FALSE(from_implicit_constructible->from_rvalue);
  EXPECT_FALSE(from_implicit_constructible->assigned);
  // construct a temporary `StatusOr` object and invoke the `StatusOr` move
  // assignment operator.
  from_implicit_constructible = FromImplicitConstructibleOnly{};
  EXPECT_FALSE(from_implicit_constructible->from_rvalue);
  EXPECT_FALSE(from_implicit_constructible->assigned);
}

TEST(StatusOr, TestStatus) {
  StatusOr<int> good(4);
  EXPECT_TRUE(good.ok());
  StatusOr<int> bad(absl::CancelledError());
  EXPECT_FALSE(bad.ok());
  EXPECT_TRUE(HasErrorCode(bad.status(), absl::StatusCode::kCancelled));
}

TEST(StatusOr, OperatorStarRefQualifiers) {
  static_assert(std::is_same<const int&,
                             decltype(*std::declval<const StatusOr<int>&>())>(),
                "Unexpected ref-qualifiers");
  static_assert(std::is_same<int&, decltype(*std::declval<StatusOr<int>&>())>(),
                "Unexpected ref-qualifiers");
  static_assert(
      std::is_same<const int&&,
                   decltype(*std::declval<const StatusOr<int>&&>())>(),
      "Unexpected ref-qualifiers");
  static_assert(
      std::is_same<int&&, decltype(*std::declval<StatusOr<int>&&>())>(),
      "Unexpected ref-qualifiers");
}

TEST(StatusOr, OperatorStar) {
  const StatusOr<std::string> const_lvalue("hello");
  EXPECT_EQ("hello", *const_lvalue);

  StatusOr<std::string> lvalue("hello");
  EXPECT_EQ("hello", *lvalue);

  // Note: Recall that std::move() is equivalent to a static_cast to an rvalue
  // reference type.
  const StatusOr<std::string> const_rvalue("hello");
  EXPECT_EQ("hello", *std::move(const_rvalue));  // NOLINT

  StatusOr<std::string> rvalue("hello");
  EXPECT_EQ("hello", *std::move(rvalue));
}

TEST(StatusOr, OperatorArrowQualifiers) {
  static_assert(
      std::is_same<const int*,
                   decltype(
                       std::declval<const StatusOr<int>&>().operator->())>(),
      "Unexpected qualifiers");
  static_assert(
      std::is_same<int*,
                   decltype(std::declval<StatusOr<int>&>().operator->())>(),
      "Unexpected qualifiers");
  static_assert(
      std::is_same<const int*,
                   decltype(
                       std::declval<const StatusOr<int>&&>().operator->())>(),
      "Unexpected qualifiers");
  static_assert(
      std::is_same<int*,
                   decltype(std::declval<StatusOr<int>&&>().operator->())>(),
      "Unexpected qualifiers");
}

TEST(StatusOr, OperatorArrow) {
  const StatusOr<std::string> const_lvalue("hello");
  EXPECT_EQ(std::string("hello"), const_lvalue->c_str());

  StatusOr<std::string> lvalue("hello");
  EXPECT_EQ(std::string("hello"), lvalue->c_str());
}

TEST(StatusOr, RValueStatus) {
  StatusOr<int> so(absl::NotFoundError("taco"));
  const Status s = std::move(so).status();

  EXPECT_THAT(s, StatusIs(absl::StatusCode::kNotFound, "taco"));

  // Check that !ok() still implies !status().ok(), even after moving out of the
  // object. See the note on the rvalue ref-qualified status method.
  EXPECT_FALSE(so.ok());  // NOLINT
  EXPECT_FALSE(so.status().ok());
  EXPECT_THAT(
      so, StatusIs(absl::StatusCode::kInternal, "Status accessed after move."));
}

TEST(StatusOr, TestValue) {
  const int kI = 4;
  StatusOr<int> thing(kI);
  EXPECT_EQ(kI, thing.value());
}

TEST(StatusOr, TestValueConst) {
  const int kI = 4;
  const StatusOr<int> thing(kI);
  EXPECT_EQ(kI, thing.value());
}

TEST(StatusOr, TestPointerDefaultCtor) {
  StatusOr<int*> thing;
  EXPECT_FALSE(thing.ok());
  EXPECT_TRUE(HasErrorCode(thing.status(), absl::StatusCode::kUnknown));
}

TEST(StatusOr, TestPointerStatusCtor) {
  StatusOr<int*> thing(absl::CancelledError());
  EXPECT_FALSE(thing.ok());
  EXPECT_TRUE(HasErrorCode(thing.status(), absl::StatusCode::kCancelled));
}

TEST(StatusOr, TestPointerValueCtor) {
  const int kI = 4;

  // Construction from a non-null pointer
  {
    StatusOr<const int*> so(&kI);
    EXPECT_TRUE(so.ok());
    XLS_EXPECT_OK(so.status());
    EXPECT_EQ(&kI, so.value());
  }

  // Construction from a null pointer constant
  {
    StatusOr<const int*> so(nullptr);
    EXPECT_TRUE(so.ok());
    XLS_EXPECT_OK(so.status());
    EXPECT_EQ(nullptr, so.value());
  }

  // Construction from a non-literal null pointer
  {
    const int* const p = nullptr;

    StatusOr<const int*> so(p);
    EXPECT_TRUE(so.ok());
    XLS_EXPECT_OK(so.status());
    EXPECT_EQ(nullptr, so.value());
  }
}

TEST(StatusOr, TestPointerCopyCtorStatusOk) {
  const int kI = 0;
  StatusOr<const int*> original(&kI);
  StatusOr<const int*> copy(original);
  XLS_EXPECT_OK(copy.status());
  EXPECT_EQ(original.value(), copy.value());
}

TEST(StatusOr, TestPointerCopyCtorStatusNotOk) {
  StatusOr<int*> original(absl::CancelledError());
  StatusOr<int*> copy(original);
  EXPECT_TRUE(HasErrorCode(copy.status(), absl::StatusCode::kCancelled));
}

TEST(StatusOr, TestPointerCopyCtorStatusOKConverting) {
  Derived derived;
  StatusOr<Derived*> original(&derived);
  StatusOr<Base2*> copy(original);
  XLS_EXPECT_OK(copy.status());
  EXPECT_EQ(static_cast<const Base2*>(original.value()), copy.value());
}

TEST(StatusOr, TestPointerCopyCtorStatusNotOkConverting) {
  StatusOr<Derived*> original(absl::CancelledError());
  StatusOr<Base2*> copy(original);
  EXPECT_TRUE(HasErrorCode(copy.status(), absl::StatusCode::kCancelled));
}

TEST(StatusOr, TestPointerAssignmentStatusOk) {
  const int kI = 0;
  StatusOr<const int*> source(&kI);
  StatusOr<const int*> target;
  target = source;
  XLS_EXPECT_OK(target.status());
  EXPECT_EQ(source.value(), target.value());
}

TEST(StatusOr, TestPointerAssignmentStatusNotOk) {
  StatusOr<int*> source(absl::CancelledError());
  StatusOr<int*> target;
  target = source;
  EXPECT_TRUE(HasErrorCode(target.status(), absl::StatusCode::kCancelled));
}

TEST(StatusOr, TestPointerAssignmentStatusOKConverting) {
  Derived derived;
  StatusOr<Derived*> source(&derived);
  StatusOr<Base2*> target;
  target = source;
  XLS_EXPECT_OK(target.status());
  EXPECT_EQ(static_cast<const Base2*>(source.value()), target.value());
}

TEST(StatusOr, TestPointerAssignmentStatusNotOkConverting) {
  StatusOr<Derived*> source(absl::CancelledError());
  StatusOr<Base2*> target;
  target = source;
  EXPECT_EQ(target.status(), source.status());
}

TEST(StatusOr, TestPointerStatus) {
  const int kI = 0;
  StatusOr<const int*> good(&kI);
  EXPECT_TRUE(good.ok());
  StatusOr<const int*> bad(absl::CancelledError());
  EXPECT_TRUE(HasErrorCode(bad.status(), absl::StatusCode::kCancelled));
}

TEST(StatusOr, TestPointerValue) {
  const int kI = 0;
  StatusOr<const int*> thing(&kI);
  EXPECT_EQ(&kI, thing.value());
}

TEST(StatusOr, TestPointerValueConst) {
  const int kI = 0;
  const StatusOr<const int*> thing(&kI);
  EXPECT_EQ(&kI, thing.value());
}

TEST(StatusOr, StatusOrVectorOfUniquePointerCanReserveAndResize) {
  using EvilType = std::vector<std::unique_ptr<int>>;
  static_assert(std::is_copy_constructible<EvilType>::value, "");
  std::vector<::xabsl::StatusOr<EvilType>> v(5);
  v.reserve(v.capacity() + 10);
  v.resize(v.capacity() + 10);
}

TEST(StatusOr, ConstPayload) {
  // A reduced version of a problematic type found in the wild. All of the
  // operations below should compile.
  xabsl::StatusOr<const int> a;

  // Copy-construction
  xabsl::StatusOr<const int> b(a);

  // Copy-assignment
  b = a;

  // Move-construction
  xabsl::StatusOr<const int> c(std::move(a));

  // Move-assignment
  b = std::move(a);  // NOLINT
}

TEST(StatusOr, MapToStatusOrUniquePtr) {
  // A reduced version of a problematic type found in the wild. All of the
  // operations below should compile.
  using MapType = std::map<std::string, StatusOr<std::unique_ptr<int>>>;

  MapType a;

  // Move-construction
  MapType b(std::move(a));

  // Move-assignment
  a = std::move(b);
}

TEST(StatusOr, ValueOrOk) {
  const StatusOr<int> status_or = 0;
  EXPECT_EQ(status_or.value_or(-1), 0);
}

TEST(StatusOr, ValueOrDefault) {
  const StatusOr<int> status_or = absl::CancelledError();
  EXPECT_EQ(status_or.value_or(-1), -1);
}

TEST(StatusOr, MoveOnlyValueOrOk) {
  EXPECT_THAT(StatusOr<std::unique_ptr<int>>(absl::make_unique<int>(0))
                  .value_or(absl::make_unique<int>(-1)),
              Pointee(0));
}

TEST(StatusOr, MoveOnlyValueOrDefault) {
  EXPECT_THAT(StatusOr<std::unique_ptr<int>>(absl::CancelledError())
                  .value_or(absl::make_unique<int>(-1)),
              Pointee(-1));
}

static StatusOr<int> MakeStatus() { return 100; }

TEST(StatusOr, TestIgnoreError) { MakeStatus().IgnoreError(); }

TEST(StatusOr, BuilderOnNestedType) {
  static const char* const kError = "My custom error.";
  auto return_builder = []() -> StatusOr<StatusOr<int>> {
    return xabsl::NotFoundErrorBuilder(xabsl::SourceLocation::current())
           << kError;
  };
  EXPECT_THAT(return_builder(),
              StatusIs(absl::StatusCode::kNotFound, HasSubstr(kError)));
}

}  // namespace
}  // namespace xabsl
