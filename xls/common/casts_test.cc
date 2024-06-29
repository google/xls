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

#include "xls/common/casts.h"

#include "gtest/gtest.h"
#include "absl/base/casts.h"

namespace xls {
namespace {

// The Base class and the Derived class are used to test the down_cast
// template function.

// A base class.
class Base {
 public:
  // It's important to have at least one virtual function in the base
  // class; otherwise down_cast, which uses dynamic_cast in a debug
  // build, won't work with it.
  virtual ~Base() = default;
};

// A class derived from Base.
class Derived : public Base {};

Base* NewBase(bool is_derived) {
  if (is_derived) {
    return new Derived;
  }
  return new Base;
}

// Tests pointer form of down_cast().
TEST(TemplateFunctionTest, down_cast_pointer) {
  Derived derived;
  Base* const base_ptr = &derived;

  // Tests casting a Base* to a Derived*.
  EXPECT_EQ(&derived, down_cast<Derived*>(base_ptr));

  // Tests casting a const Base* to a const Derived*.
  const Base* const_base_ptr = base_ptr;
  EXPECT_EQ(&derived, down_cast<const Derived*>(const_base_ptr));

  // Tests casting a Base* to a const Derived*.
  EXPECT_EQ(&derived, down_cast<const Derived*>(base_ptr));

  // Tests casting a Base* to a Base* (an identity cast).
  EXPECT_EQ(base_ptr, down_cast<Base*>(base_ptr));

  // Tests down casting NULL.
  EXPECT_EQ(nullptr, (down_cast<Derived*, Base>(nullptr)));

  // Tests a bad downcast. We have to disguise the badness just enough
  // that the compiler doesn't warn about it at compile time.
  Base* base2 = NewBase(false);
#if GTEST_HAS_DEATH_TEST
  EXPECT_DEBUG_DEATH(down_cast<Derived*>(base2), "dynamic_cast<To>");
#endif  // GTEST_HAS_DEATH_TEST
  delete base2;
}

// Tests reference form of down_cast().
TEST(TemplateFunctionTest, down_cast_reference) {
  Derived derived;
  Base& base_ref = derived;

  // Tests casting a Base& to a Derived&.
  EXPECT_EQ(&derived,
            &down_cast<Derived&>(base_ref));  // NOLINT(runtime/casting)

  // Tests casting a const Base& to a const Derived&.
  const Base& const_base_ref = base_ref;
  EXPECT_EQ(&derived, &down_cast<const Derived&>(  // NOLINT(runtime/casting)
                          const_base_ref));

  // Tests casting a Base& to a const Derived&.
  EXPECT_EQ(&derived,
            &down_cast<const Derived&>(base_ref));  // NOLINT(runtime/casting)

  // Tests casting a Base& to a Base& (an identity cast).
  EXPECT_EQ(&base_ref, &down_cast<Base&>(base_ref));  // NOLINT(runtime/casting)

  // Tests a bad downcast. We have to disguise the badness just enough
  // that the compiler doesn't warn about it at compile time.
  Base& base2 = *NewBase(false);
#if GTEST_HAS_DEATH_TEST
  EXPECT_DEBUG_DEATH(down_cast<Derived&>(base2), "dynamic_cast<.*To.*>");
#endif  // GTEST_HAS_DEATH_TEST
  delete &base2;
}

// Helper class for verifying the type of an expression.
template <class Expected>
struct TypeVerifier {
  static Expected Verify(Expected a) { return a; }
  // Catch-all for arguments of unexpected type.
  template <class Actual>
  static void Verify(Actual a);
};

// Verifies that
//   implicit_cast<To>(from)
// gives the same result as
//   To to = from;
template <class To, class From>
void CheckImplicitCast(From from) {
  To to = from;
  EXPECT_EQ(to, TypeVerifier<To>::Verify(absl::implicit_cast<To>(from)));
}

struct IntWrapper {
  IntWrapper(int val) : val(val) {}     // NOLINT: Implicit conversion.
  operator int() const { return val; }  // NOLINT: Implicit conversion.
  int val;
};

TEST(ImplicitCastTest, BasicConversions) {
  CheckImplicitCast<int>('A');
  CheckImplicitCast<char>(65);
  int int_val = 42;
  CheckImplicitCast<const int&, int&>(int_val);
  CheckImplicitCast<const int*>(&int_val);
  Derived derived;
  CheckImplicitCast<Base*>(&derived);
  CheckImplicitCast<IntWrapper>(42);
  CheckImplicitCast<int>(IntWrapper(42));
}

TEST(ImplicitCastTest, NullLiteral) {
  int* null_int = nullptr;
  int (*null_f)() = nullptr;
  EXPECT_EQ(null_int, absl::implicit_cast<int*>(nullptr));
  EXPECT_EQ(null_f, absl::implicit_cast<int (*)()>(nullptr));
}

void OverloadedFunction() {}
void OverloadedFunction(int) {}

TEST(ImplicitCastTest, SelectsOverloadedFunction) {
  void (*expected)() = &OverloadedFunction;
  EXPECT_EQ(expected, absl::implicit_cast<void (*)()>(&OverloadedFunction));
  void (*expected2)(int) = &OverloadedFunction;
  EXPECT_EQ(expected2, absl::implicit_cast<void (*)(int)>(&OverloadedFunction));
}

class WithPrivateBase : private Base {
 public:
  WithPrivateBase() {
    Base* expected = this;
    EXPECT_EQ(expected, absl::implicit_cast<Base*>(this));
  }
};

TEST(ImplicitCastTest, PrivateInheritance) { WithPrivateBase d; }

}  // namespace
}  // namespace xls
