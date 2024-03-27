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

#ifndef XLS_IR_UNWRAPPING_ITERATOR_H_
#define XLS_IR_UNWRAPPING_ITERATOR_H_

#include <cstddef>
#include <iterator>
#include <utility>

namespace xls {

// UnwrappingIterator is a transforming iterator that calls get() on the
// elements it returns.
//
// Shamelessly copied from jlebar's
// tensorflow/compiler/xla/iterator_util.h.
//
// Together with absl::iterator_range, this lets classes which contain a
// collection of std::unique_ptrs expose a view of raw pointers to consumers.
// For example:
//
//  class MyContainer {
//   public:
//    absl::iterator_range<
//        UnwrappingIterator<std::vector<std::unique_ptr<Thing>>::iterator>>
//    things() {
//      return {MakeUnwrappingIterator(things_.begin()),
//              MakeUnwrappingIterator(things_.end())};
//    }
//
//    absl::iterator_range<UnwrappingIterator<
//        std::vector<std::unique_ptr<Thing>>::const_iterator>>
//    things() const {
//      return {MakeUnwrappingIterator(things_.begin()),
//              MakeUnwrappingIterator(things_.end())};
//    }
//
//   private:
//    std::vector<std::unique_ptr<Thing>> things_;
//  };
//
//  MyContainer container = ...;
//  for (Thing* t : container.things()) {
//    ...
//  }
//
// For simplicity, UnwrappingIterator is currently unconditionally an
// input_iterator -- it doesn't inherit any superpowers NestedIterator may have.
template <std::input_iterator NestedIter>
class UnwrappingIterator {
 public:
  using iterator_category = std::input_iterator_tag;
  using value_type = decltype(std::declval<NestedIter>()->get());
  using difference_type = ptrdiff_t;
  using pointer = value_type*;
  using reference = value_type&;

  explicit UnwrappingIterator(NestedIter iter) : iter_(std::move(iter)) {}

  auto operator*() const -> value_type { return iter_->get(); }
  auto operator->() const -> value_type { return iter_->get(); }
  UnwrappingIterator& operator++() {
    ++iter_;
    return *this;
  }
  UnwrappingIterator operator++(int) {
    UnwrappingIterator temp(iter_);
    operator++();
    return temp;
  }

  friend bool operator==(const UnwrappingIterator& a,
                         const UnwrappingIterator& b) {
    return a.iter_ == b.iter_;
  }

  friend bool operator!=(const UnwrappingIterator& a,
                         const UnwrappingIterator& b) {
    return !(a == b);
  }

 private:
  NestedIter iter_;
};

template <typename NestedIter>
UnwrappingIterator<NestedIter> MakeUnwrappingIterator(NestedIter iter) {
  return UnwrappingIterator<NestedIter>(std::move(iter));
}

}  // namespace xls

#endif  // XLS_IR_UNWRAPPING_ITERATOR_H_
