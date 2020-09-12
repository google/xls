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

#ifndef XLS_DSLX_PYTHON_CPP_AST_H_
#define XLS_DSLX_PYTHON_CPP_AST_H_

#include "absl/base/casts.h"
#include "pybind11/pybind11.h"
#include "xls/dslx/cpp_ast.h"

namespace xls::dslx {

template <typename T>
class PointerOwnedByModule {
 public:
  PointerOwnedByModule(T* pointer, const std::shared_ptr<Module>& owner)
      : pointer_(pointer), owner_(owner) {
    XLS_CHECK(pointer != nullptr);
  }

  T& deref() const { return *pointer_; }
  const std::shared_ptr<Module>& module() { return owner_; }

 private:
  T* pointer_;
  std::shared_ptr<Module> owner_;
};

class AstNodeHolder : public PointerOwnedByModule<AstNode> {
 protected:
  using PointerOwnedByModule::PointerOwnedByModule;
};

#define XLS_DSLX_AST_NODE_CHILD_EACH(X) \
  X(BuiltinNameDef)                     \
  X(ConstantDef)                        \
  X(Enum)                               \
  X(Expr)                               \
  X(Function)                           \
  X(Import)                             \
  X(MatchArm)                           \
  X(Module)                             \
  X(NameDef)                            \
  X(NameDefTree)                        \
  X(Param)                              \
  X(ParametricBinding)                  \
  X(Proc)                               \
  X(QuickCheck)                         \
  X(Slice)                              \
  X(Struct)                             \
  X(Test)                               \
  X(TypeAnnotation)                     \
  X(TypeDef)                            \
  X(WidthSlice)                         \
  X(WildcardPattern)

#define DEFINE_AST_NODE_HOLDER(__subtype)                     \
  struct __subtype##Holder : public AstNodeHolder {           \
    using AstNodeHolder::AstNodeHolder;                       \
    __subtype& deref() const {                                \
      return static_cast<__subtype&>(AstNodeHolder::deref()); \
    }                                                         \
  };

XLS_DSLX_AST_NODE_CHILD_EACH(DEFINE_AST_NODE_HOLDER)

#undef DEFINE_AST_NODE_HOLDER

#define XLS_DSLX_EXPR_CHILD_EACH(X) \
  X(Array)                          \
  X(Attr)                           \
  X(Binop)                          \
  X(Carry)                          \
  X(Cast)                           \
  X(EnumRef)                        \
  X(For)                            \
  X(Index)                          \
  X(Invocation)                     \
  X(Let)                            \
  X(Match)                          \
  X(ModRef)                         \
  X(NameRef)                        \
  X(Next)                           \
  X(Number)                         \
  X(SplatStructInstance)            \
  X(StructInstance)                 \
  X(Ternary)                        \
  X(TypeRef)                        \
  X(Unop)                           \
  X(While)                          \
  X(XlsTuple)

// Define all the expression holders.
#define DEFINE_EXPR_HOLDER(__subtype)                       \
  struct __subtype##Holder : public ExprHolder {            \
    __subtype##Holder(__subtype* pointer,                   \
                      const std::shared_ptr<Module>& owner) \
        : ExprHolder(pointer, owner) {}                     \
    __subtype& deref() const {                              \
      return static_cast<__subtype&>(ExprHolder::deref());  \
    }                                                       \
  };

XLS_DSLX_EXPR_CHILD_EACH(DEFINE_EXPR_HOLDER)

#undef DEFINE_EXPR_HOLDER

#define DEFINE_TYPE_ANNOTATION_HOLDER(__subtype)                     \
  struct __subtype##Holder : public TypeAnnotationHolder {           \
    __subtype##Holder(__subtype* pointer,                            \
                      const std::shared_ptr<Module>& owner)          \
        : TypeAnnotationHolder(pointer, owner) {}                    \
    __subtype& deref() const {                                       \
      return static_cast<__subtype&>(TypeAnnotationHolder::deref()); \
    }                                                                \
  };

DEFINE_TYPE_ANNOTATION_HOLDER(BuiltinTypeAnnotation);
DEFINE_TYPE_ANNOTATION_HOLDER(ArrayTypeAnnotation);
DEFINE_TYPE_ANNOTATION_HOLDER(TupleTypeAnnotation);
DEFINE_TYPE_ANNOTATION_HOLDER(TypeRefTypeAnnotation);

#undef DEFINE_TYPE_ANNOTATION_HOLDER

struct ConstantArrayHolder : public ArrayHolder {
  ConstantArrayHolder(ConstantArray* pointer,
                      const std::shared_ptr<Module>& owner)
      : ArrayHolder(pointer, owner) {}
  ConstantArray& deref() const {
    return static_cast<ConstantArray&>(ArrayHolder::deref());
  }
};

struct ConstRefHolder : public NameRefHolder {
  ConstRefHolder(ConstRef* pointer, const std::shared_ptr<Module>& owner)
      : NameRefHolder(pointer, owner) {}
  ConstRef& deref() const {
    return static_cast<ConstRef&>(NameRefHolder::deref());
  }
};

struct TestFunctionHolder : public TestHolder {
  TestFunctionHolder(TestFunction* pointer,
                     const std::shared_ptr<Module>& owner)
      : TestHolder(pointer, owner) {}
  TestFunction& deref() const {
    return static_cast<TestFunction&>(TestHolder::deref());
  }
};

}  // namespace xls::dslx

// Tell pybind11 how to cast an ExprHolder to a specific type. This is necessary
// to make wrapped C++ methods that return an ExprHolder work, otherwise it's
// not possible to call methods of subtypes in Python.
namespace pybind11 {

#define HANDLE_SUBTYPE(__type)                                      \
  if (dynamic_cast<xls::dslx::__type*>(&src->deref()) != nullptr) { \
    type = &typeid(xls::dslx::__type##Holder);                      \
    return static_cast<const xls::dslx::__type##Holder*>(src);      \
  }

template <>
struct polymorphic_type_hook<xls::dslx::ArrayHolder> {
  static const void* get(const xls::dslx::ArrayHolder* src,
                         const std::type_info*& type) {  // NOLINT
    if (src == nullptr) {
      return nullptr;
    }

    HANDLE_SUBTYPE(ConstantArray);

    return src;
  }
};

template <>
struct polymorphic_type_hook<xls::dslx::NameRefHolder> {
  static const void* get(const xls::dslx::NameRefHolder* src,
                         const std::type_info*& type) {  // NOLINT
    if (src == nullptr) {
      return nullptr;
    }

    HANDLE_SUBTYPE(ConstRef);

    return src;
  }
};

template <>
struct polymorphic_type_hook<xls::dslx::ExprHolder> {
  static const void* get(const xls::dslx::ExprHolder* src,
                         const std::type_info*& type) {  // NOLINT
    if (src == nullptr) {
      return nullptr;
    }

    if (dynamic_cast<xls::dslx::NameRef*>(&src->deref()) != nullptr) {
      auto* a_holder = absl::bit_cast<xls::dslx::NameRefHolder*>(src);
      type = &typeid(xls::dslx::NameRefHolder);
      return polymorphic_type_hook<xls::dslx::NameRefHolder>::get(a_holder,
                                                                  type);
    }
    if (dynamic_cast<xls::dslx::Array*>(&src->deref()) != nullptr) {
      auto* a_holder = absl::bit_cast<xls::dslx::ArrayHolder*>(src);
      type = &typeid(xls::dslx::ArrayHolder);
      return polymorphic_type_hook<xls::dslx::ArrayHolder>::get(a_holder, type);
    }
    XLS_DSLX_EXPR_CHILD_EACH(HANDLE_SUBTYPE)

    return src;
  }
};

template <>
struct polymorphic_type_hook<xls::dslx::TypeAnnotationHolder> {
  static const void* get(const xls::dslx::TypeAnnotationHolder* src,
                         const std::type_info*& type) {  // NOLINT
    if (src == nullptr) {
      return nullptr;
    }

    HANDLE_SUBTYPE(ArrayTypeAnnotation);
    HANDLE_SUBTYPE(BuiltinTypeAnnotation);
    HANDLE_SUBTYPE(TupleTypeAnnotation);
    HANDLE_SUBTYPE(TypeRefTypeAnnotation);

    std::cerr << "Not a subtype: " << src->deref().ToString() << "\n";
    return src;
  }
};

template <>
struct polymorphic_type_hook<xls::dslx::AstNodeHolder> {
  static const void* get(const xls::dslx::AstNodeHolder* src,
                         const std::type_info*& type) {  // NOLINT
    if (src == nullptr) {
      return nullptr;
    }

    if (dynamic_cast<xls::dslx::Expr*>(&src->deref()) != nullptr) {
      auto* e_holder = absl::bit_cast<xls::dslx::ExprHolder*>(src);
      type = &typeid(xls::dslx::ExprHolder);
      return polymorphic_type_hook<xls::dslx::ExprHolder>::get(e_holder, type);
    }
    if (dynamic_cast<xls::dslx::TypeAnnotation*>(&src->deref()) != nullptr) {
      auto* e_holder = absl::bit_cast<xls::dslx::TypeAnnotationHolder*>(src);
      type = &typeid(xls::dslx::TypeAnnotationHolder);
      return polymorphic_type_hook<xls::dslx::TypeAnnotationHolder>::get(
          e_holder, type);
    }
    XLS_DSLX_AST_NODE_CHILD_EACH(HANDLE_SUBTYPE)

    return src;
  }
};

#undef HANDLE_SUBTYPE

}  // namespace pybind11

#endif  // XLS_DSLX_PYTHON_CPP_AST_H_
