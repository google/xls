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

#ifndef XLS_IR_PYTHON_WRAPPER_TYPES_H_
#define XLS_IR_PYTHON_WRAPPER_TYPES_H_

#include <memory>
#include <type_traits>

#include "absl/status/statusor.h"
#include "pybind11/cast.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"

namespace xls {

// A number of C++ IR types contain unowning Package* pointers. In order to
// ensure memory safety, all Python references to such types values are wrapped
// in objects that contain not just the unowning pointer but but also a
// shared_ptr to its Package.
//
// Unfortunately, this wrapping causes quite a lot of friction in defining the
// pybind11 bindings. This file contains type definitions and helpers for this.
// The main helper function is PyBind, which in most cases makes it possible to
// write pybind11 wrapper declarations mostly as usual even for objects that are
// wrapped, and when parameters and return values are of types that need to be
// wrapped.

// Classes that are a wrapper that contains both an unowning pointer and an
// owning pointer should inherit this class to signal to helper functions that
// it is such a wrapper.
//
// Subclasses of this class should have a T& deref() const method that returns
// the wrapped unowned object.
//
// Subclasses of this class are expected to be copyable.
struct PyHolder {};

// Base class for wrapper objects that have objects that are owned by Package.
// (BValue and FunctionBaseHolder need more than just a shared_ptr to Package so
// they don't use this.)
template <typename T>
class PointerOwnedByPackage : public PyHolder {
 public:
  PointerOwnedByPackage(T* pointer, const std::shared_ptr<Package>& owner)
      : pointer_(pointer), owner_(owner) {}

  T& deref() const { return *pointer_; }
  const std::shared_ptr<Package>& package() { return owner_; }

 private:
  T* pointer_;
  std::shared_ptr<Package> owner_;
};

// Abstract base class for wrappers of xls::Type objects. pybind11 expects to be
// able to do static_cast up casts for types that are declared to inherit each
// other so it is necessary that the wrapper class hierarchy mirrors the
// hierarchy of the wrapped objects.
class TypeHolder : public PointerOwnedByPackage<Type> {
 protected:
  using PointerOwnedByPackage::PointerOwnedByPackage;
};

// Wrapper for BitsType* pointers.
struct BitsTypeHolder : public TypeHolder {
  BitsTypeHolder(BitsType* pointer, const std::shared_ptr<Package>& owner)
      : TypeHolder(pointer, owner) {}

  BitsType& deref() const {
    return static_cast<BitsType&>(TypeHolder::deref());
  }
};

// Wrapper for ArrayType* pointers.
struct ArrayTypeHolder : public TypeHolder {
  ArrayTypeHolder(ArrayType* pointer, const std::shared_ptr<Package>& owner)
      : TypeHolder(pointer, owner) {}

  ArrayType& deref() const {
    return static_cast<ArrayType&>(TypeHolder::deref());
  }
};

// Wrapper for TupleType* pointers.
struct TupleTypeHolder : public TypeHolder {
  TupleTypeHolder(TupleType* pointer, const std::shared_ptr<Package>& owner)
      : TypeHolder(pointer, owner) {}

  TupleType& deref() const {
    return static_cast<TupleType&>(TypeHolder::deref());
  }
};

// Wrapper for FunctionType* pointers.
struct FunctionTypeHolder : public PointerOwnedByPackage<FunctionType> {
  using PointerOwnedByPackage::PointerOwnedByPackage;
};

// Wrapper for FunctionBase* pointers.
struct FunctionBaseHolder : public PointerOwnedByPackage<FunctionBase> {
  using PointerOwnedByPackage::PointerOwnedByPackage;
};

// Wrapper for Function* pointers.
struct FunctionHolder : public PointerOwnedByPackage<Function> {
  using PointerOwnedByPackage::PointerOwnedByPackage;
};

// Wrapper for Package objects.
//
// A Package can be referred to through unowned pointers by other objects like
// FunctionBuilder and Type. For this reason, it is held in a shared_ptr, so
// that it can be kept alive for as long as any Python object refers to it, even
// if there is no longer a direct reference to the Package.
//
// Using PointerOwnedByPackage makes this object contain a Package* and a
// std::shared_ptr<Package> which is a little bit weird, but it allows reusing
// helper functions that work on PointerOwnedByPackage.
struct PackageHolder : public PointerOwnedByPackage<Package> {
  using PointerOwnedByPackage::PointerOwnedByPackage;
};

// Wrapper for BValue objects.
//
// A BValue C++ object has unowned pointers both to its Package and to its
// FunctionBuilder. To keep them alive for as long as the Python reference
// exists, this holder keeps shared_ptrs to the Package and FunctionBuilder.
class BValueHolder : public PyHolder {
 public:
  BValueHolder(const BValue& value, const std::shared_ptr<Package>& package,
               const std::shared_ptr<FunctionBuilder>& builder)
      : value_(std::make_shared<BValue>(value)),
        package_(package),
        builder_(builder) {}

  const std::shared_ptr<BValue>& value() const { return value_; }
  const std::shared_ptr<Package>& package() const { return package_; }
  const std::shared_ptr<FunctionBuilder>& builder() const { return builder_; }

  BValue& deref() const { return *value_; }

 private:
  std::shared_ptr<BValue> value_;
  std::shared_ptr<Package> package_;
  std::shared_ptr<FunctionBuilder> builder_;
};

// Wrapper for FunctionBuilder objects.
//
// A FunctionBuilder has unowned pointers to the belonging package. To keep the
// Package alive for as long as the Python reference exists, it is held by
// shared_ptr here. Furthermore, since FunctionBuilder can be referred to by
// other BValue objects, it itself is kept in a shared_ptr too.
class FunctionBuilderHolder : public PyHolder {
 public:
  FunctionBuilderHolder(std::string_view name, PackageHolder package)
      : package_(package.package()),
        builder_(std::make_shared<FunctionBuilder>(name, &package.deref())) {}

  FunctionBuilderHolder(const std::shared_ptr<Package>& package,
                        const std::shared_ptr<FunctionBuilder>& builder)
      : package_(package), builder_(builder) {}

  FunctionBuilder& deref() const { return *builder_; }

  const std::shared_ptr<Package>& package() const { return package_; }
  const std::shared_ptr<FunctionBuilder>& builder() const { return builder_; }

 private:
  std::shared_ptr<Package> package_;
  std::shared_ptr<FunctionBuilder> builder_;
};

// PyHolderTypeTraits is a template that defines a mapping from a wrapped type
// (for example BitsType) to its holder type (in this case BitsTypeHolder).
template <typename T>
struct PyHolderTypeTraits;

template <>
struct PyHolderTypeTraits<Type> {
  using Holder = TypeHolder;
};

template <>
struct PyHolderTypeTraits<BitsType> {
  using Holder = BitsTypeHolder;
};

template <>
struct PyHolderTypeTraits<ArrayType> {
  using Holder = ArrayTypeHolder;
};

template <>
struct PyHolderTypeTraits<TupleType> {
  using Holder = TupleTypeHolder;
};

template <>
struct PyHolderTypeTraits<FunctionType> {
  using Holder = FunctionTypeHolder;
};

template <>
struct PyHolderTypeTraits<FunctionBase> {
  using Holder = FunctionBaseHolder;
};

template <>
struct PyHolderTypeTraits<Function> {
  using Holder = FunctionHolder;
};

template <>
struct PyHolderTypeTraits<Package> {
  using Holder = PackageHolder;
};

template <>
struct PyHolderTypeTraits<BValue> {
  using Holder = BValueHolder;
};

template <>
struct PyHolderTypeTraits<FunctionBuilder> {
  using Holder = FunctionBuilderHolder;
};

// PyHolderType<T> finds the wrapper type of a given type. For example,
// PyHolderType<BitsType> is BitsTypeHolder.
template <typename T>
using PyHolderType = typename PyHolderTypeTraits<T>::Holder;

// Takes something that potentially should be wrapped by a holder object (for
// example BValue) and wraps it in a holder object (for example a BValueHolder).
// The second parameter is a PyHolder object that is used to get the owning
// pointers required to construct the holder object.
//
// If T is not a type that needs to be wrapped, this function returns `value`.
//
// This function is used by PyWrap on return values of wrapped methods.
//
// Making PyWrap able to wrap a new return type is done by adding a new overload
// of the WrapInPyHolder function.
template <typename T, typename Holder>
auto WrapInPyHolderIfHolderExists(T&& value, Holder* holder)
    -> decltype(WrapInPyHolder(std::forward<T>(value), holder)) {
  return WrapInPyHolder(std::forward<T>(value), holder);
}
template <typename T>
auto WrapInPyHolderIfHolderExists(T&& value, ...) {
  return std::forward<T>(value);
}

// Tells PyWrap how to wrap BValue return values, see
// WrapInPyHolderIfHolderExists.
static inline BValueHolder WrapInPyHolder(const BValue& value,
                                          FunctionBuilderHolder* holder) {
  return BValueHolder(value, holder->package(), holder->builder());
}

// Tells PyWrap how to wrap FunctionBuilder return values from BValue methods,
// see WrapInPyHolderIfHolderExists.
static inline FunctionBuilderHolder WrapInPyHolder(FunctionBuilder* builder,
                                                   BValueHolder* holder) {
  return FunctionBuilderHolder(holder->package(), holder->builder());
}

// Tells PyWrap how to wrap return values from BValue methods of pointers owned
// by Package, see WrapInPyHolderIfHolderExists.
template <typename T>
PyHolderType<T> WrapInPyHolder(T* value, BValueHolder* holder) {
  return PyHolderType<T>(value, holder->package());
}

// Tells PyWrap how to wrap return values of pointers owned by Package, see
// WrapInPyHolderIfHolderExists.
template <typename T, typename U>
PyHolderType<T> WrapInPyHolder(T* value, PointerOwnedByPackage<U>* holder) {
  return PyHolderType<T>(value, holder->package());
}

// Tells PyWrap how to wrap absl::StatusOr return values of types that need to
// be wrapped, see WrapInPyHolderIfHolderExists.
template <typename T, typename Holder>
absl::StatusOr<PyHolderType<T>> WrapInPyHolder(const absl::StatusOr<T*>& value,
                                               Holder* holder) {
  if (value.ok()) {
    return PyHolderType<T>(value.value(), holder->package());
  } else {
    return value.status();
  }
}

// HasPyHolderType<T> is true when T is a type that has a PyHolder type.
template <typename, typename = std::void_t<>>
struct HasPyHolderTypeHelper : std::false_type {};
template <typename T>
struct HasPyHolderTypeHelper<
    T, std::void_t<typename PyHolderTypeTraits<T>::Holder>>
    : public std::true_type {};
template <typename T>
constexpr bool HasPyHolderType = HasPyHolderTypeHelper<T>::value;
static_assert(!HasPyHolderType<int>);
static_assert(HasPyHolderType<BValue>);

// WrappedFunctionParameterTraits<T> defines how PyWrap handles parameters of
// functions it wraps: For a function void Foo(T t), the wrapped function will
// be of type void (*)(WrappedFunctionParameterTraits<T>::WrappedType t).
//
// PyWrap then uses WrappedFunctionParameterTraits<T>::Unwrap(t) to extract the
// potentially wrapped underlying type to pass it to the wrapped function.
//
// This primary template specifies how PyWrap behaves for types that are not
// wrapped. Partial specializations define the behavior for wrapped types.
template <typename T, typename = std::void_t<>>
struct WrappedFunctionParameterTraits {
  using WrappedType = T;

  static T Unwrap(T&& t) { return std::move(t); }
};

// WrappedFunctionParameterTraits for absl::Spans of types that have wrapper
// types.
template <typename T>
struct WrappedFunctionParameterTraits<absl::Span<const T>,
                                      std::enable_if_t<HasPyHolderType<T>>> {
  using WrappedType = absl::Span<const PyHolderType<T>>;

  static std::vector<T> Unwrap(const WrappedType& t) {
    std::vector<T> values;
    values.reserve(t.size());
    for (auto& value : t) {
      values.push_back(value.deref());
    }
    return values;
  }
};

// WrappedFunctionParameterTraits for std::optionals of types that have wrapper
// types.
template <typename T>
struct WrappedFunctionParameterTraits<std::optional<T>,
                                      std::enable_if_t<HasPyHolderType<T>>> {
  using WrappedType = std::optional<PyHolderType<T>>;

  static std::optional<T> Unwrap(const WrappedType& t) {
    if (t) {
      return t->deref();
    } else {
      return {};
    }
  }
};

// WrappedFunctionParameterTraits for types that have wrapper types.
template <typename T>
struct WrappedFunctionParameterTraits<
    T, std::enable_if_t<HasPyHolderType<std::remove_pointer_t<T>>>> {
 private:
  using NonPointerType = PyHolderType<std::remove_pointer_t<T>>;

 public:
  using WrappedType =
      std::conditional_t<std::is_pointer_v<T>, NonPointerType*, NonPointerType>;

  static T Unwrap(const WrappedType& t) {
    if constexpr (std::is_pointer_v<T>) {
      return &t->deref();
    } else {
      return t.deref();
    }
  }
};

// Helper function with shared logic for the two PyWrap overloads that take
// pointers to member methods (the const and the non-const variant).
template <typename MethodPointer, typename ReturnT, typename T,
          typename... Args>
auto PyWrapHelper(MethodPointer method_pointer) {
  return
      [method_pointer](
          PyHolderType<T>* t,
          typename WrappedFunctionParameterTraits<Args>::WrappedType... args) {
        auto unwrapped = ((t->deref()).*method_pointer)(
            WrappedFunctionParameterTraits<Args>::Unwrap(
                std::forward<
                    typename WrappedFunctionParameterTraits<Args>::WrappedType>(
                    args))...);
        return WrapInPyHolderIfHolderExists(std::move(unwrapped), t);
      };
}

// PyWrap is the main interface for the code in this file. It is designed to be
// used in pybind11 interface definition code when declaring methods of wrapped
// types. It takes a pointer to a method of a wrapped type that potentially has
// parameters of types that are wrapped, and potentially returns something that
// needs to be wrapped.
//
// Its return value is a functor that unwraps the object, calls the given method
// after unwrapping the parameters and then wraps the return value before
// returning.
//
// Example usage (here, FunctionBaseHolder is a PyHolder type that holds a
// FunctionBase*):
//
// py::class_<FunctionBaseHolder>(m, "FunctionBase")
//     .def("dump_ir", PyWrap(&FunctionBase::DumpIr));
//
// Note that even though FunctionBase* is wrapped in FunctionBaseHolder, the
// type can be declared almost as if the object is not wrapped. For comparison,
// an unwrapped type would have been defined like this:
//
// py::class_<FunctionBase>(m, "FunctionBase")
//     .def("dump_ir", &FunctionBase::DumpIr);
template <typename ReturnT, typename T, typename... Args>
auto PyWrap(ReturnT (T::*method_pointer)(Args...) const) {
  return PyWrapHelper<decltype(method_pointer), ReturnT, T, Args...>(
      method_pointer);
}

// PyWrap for member methods. For more info see the docs on the other overload.
template <typename ReturnT, typename T, typename... Args>
auto PyWrap(ReturnT (T::*method_pointer)(Args...)) {
  return PyWrapHelper<decltype(method_pointer), ReturnT, T, Args...>(
      method_pointer);
}

// PyWrap for static methods and free functions. This behaves like PyWrap for
// methods, except that it is not able to wrap return values, because it doesn't
// have a pointer to a this PyHolder object that it can extract owned pointers
// from.
template <typename ReturnT, typename... Args>
auto PyWrap(ReturnT (*function_pointer)(Args...)) {
  return
      [function_pointer](
          typename WrappedFunctionParameterTraits<Args>::WrappedType... args) {
        // Because there is no wrapped this pointer, it's not possible to wrap
        // the return value. This only unwraps parameters when applicable.
        return (*function_pointer)(WrappedFunctionParameterTraits<Args>::Unwrap(
            std::forward<
                typename WrappedFunctionParameterTraits<Args>::WrappedType>(
                args))...);
      };
}

}  // namespace xls

// Tell pybind11 how to cast a TypeHolder to a specific type. This is necessary
// to make wrapped C++ methods that return a Type* work, otherwise it's not
// possible to call methods of subtypes in Python.
namespace pybind11 {
template <>
struct polymorphic_type_hook<xls::TypeHolder> {
  static const void* get(const xls::TypeHolder* src,
                         const std::type_info*& type) {  // NOLINT
    if (src == nullptr) {
      return nullptr;
    }

    if (src->deref().IsBits()) {
      type = &typeid(xls::BitsTypeHolder);
      return static_cast<const xls::BitsTypeHolder*>(src);
    } else if (src->deref().IsTuple()) {
      type = &typeid(xls::TupleTypeHolder);
      return static_cast<const xls::TupleTypeHolder*>(src);
    } else if (src->deref().IsArray()) {
      type = &typeid(xls::ArrayTypeHolder);
      return static_cast<const xls::ArrayTypeHolder*>(src);
    }

    return src;
  }
};
}  // namespace pybind11

#endif  // XLS_IR_PYTHON_WRAPPER_TYPES_H_
