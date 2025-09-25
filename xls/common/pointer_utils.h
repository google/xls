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

#ifndef XLS_COMMON_POINTER_UTILS_H_
#define XLS_COMMON_POINTER_UTILS_H_

#include <memory>

namespace xls {

namespace internal {
using RawDeletePtr = void (*)(void*);
}

// type-erased version of unique ptr that keeps track of the appropriate
// destructor. To use reinterpret_cast<T*>(x.get()).
using TypeErasedUniquePtr = std::unique_ptr<void, internal::RawDeletePtr>;

template <typename T, typename Deleter = std::default_delete<T>>
TypeErasedUniquePtr EraseType(std::unique_ptr<T, Deleter> ptr) {
  return TypeErasedUniquePtr(
      ptr.release(), [](void* ptr) { Deleter()(reinterpret_cast<T*>(ptr)); });
}

}  // namespace xls

#endif  // XLS_COMMON_POINTER_UTILS_H_
