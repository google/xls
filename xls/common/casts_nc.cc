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

// This file is used to test that using casts.h in certain incorrect ways does
// lead to compiler errors.  Compilation of this file is driven by the code in
// casts_nc_test.py.
//
// See casts_test.cc for the normal "positive" tests on casts.h.

#include "xls/common/casts.h"  // IWYU pragma: keep

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

// Another class derived from Base.
class Derived2 : public Base {};

// Yet another class derived from Base, this time with a constructor from
// base. This is used to verify that the existence of such a constructor
// doesn't let us use the reference form of down_cast with non-references.
class Derived3 : public Base {
 public:
  Derived3() = default;
  Derived3(const Base&) {}  // NOLINT(google-explicit-constructor)
};

// A class unrelated to the Base class hierarchy.
class Unrelated {
 public:
  virtual ~Unrelated() = default;
};

// We want to test one compiler error at a time, so we partition the code into
// branches using conditional compilation.  casts_nc_test.py will compile this
// file multiple times, each time with a different TEST_* macro defined, and
// verify that the compilation fails each time.

#if defined(TEST_down_cast_UP)

// Tests using down_cast to do an up cast.
Base* WontCompile(Derived* p) { return down_cast<Base*>(p); }

#elif defined(TEST_down_cast_CROSS)

// Tests using down_cast to do a cross cast.
Derived2* WontCompile(Derived* p) { return down_cast<Derived2*>(p); }

#elif defined(TEST_down_cast_UNRELATED)

// Tests using down_cast to cast between unrelated types.
Unrelated* WontCompile(Base* p) { return down_cast<Unrelated*>(p); }

#elif defined(TEST_down_cast_REMOVE_CONST)

// Tests using down_cast to remove const qualification
Derived* WontCompile(const Base* p) { return down_cast<Derived*>(p); }

#elif defined(TEST_down_cast_UP_REF)

// Tests using down_cast to do an up cast on references.
Base& WontCompile(Derived& r) { return down_cast<Base&>(r); }

#elif defined(TEST_down_cast_CROSS_REF)

// Tests using down_cast to do a cross cast on references.
Derived2& WontCompile(Derived& r) { return down_cast<Derived2&>(r); }

#elif defined(TEST_down_cast_UNRELATED_REF)

// Tests using down_cast to cast between unrelated types on references.
Unrelated& WontCompile(Base& r) { return down_cast<Unrelated&>(r); }

#elif defined(TEST_down_cast_REMOVE_CONST_REF)

// Tests using down_cast to remove const qualification
Derived& WontCompile(const Base& r) { return down_cast<Derived&>(r); }

#elif defined(TEST_down_cast_NOT_A_REF)

// Tests using down_cast when the target type isn't a reference.
Derived3 WontCompile(Base& r) { return down_cast<Derived3>(r); }

#else  // Smoke test

// Finally, tests that good code does compile (i.e. the compiler is
// not blindly failing everything).

static int Identity(int x) { return x; }

#endif
