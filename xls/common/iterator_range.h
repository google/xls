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

// This provides a very simple, boring adaptor for a begin and end iterator
// into a range type. This should be used to build range views that work well
// with range based for loops and range based constructors.

#ifndef XLS_COMMON_ITERATOR_RANGE_H_
#define XLS_COMMON_ITERATOR_RANGE_H_

#include <iterator>
#include <utility>


namespace xabsl {

// A range adaptor for a pair of iterators.
//
// This just wraps two iterators into a range-compatible interface. Nothing
// fancy at all.
template <typename IteratorT>
class iterator_range {
 public:
  using iterator = IteratorT;
  using const_iterator = IteratorT;
  using value_type = typename std::iterator_traits<IteratorT>::value_type;

  iterator_range() : begin_iterator_(), end_iterator_() {}
  iterator_range(IteratorT begin_iterator, IteratorT end_iterator)
      : begin_iterator_(std::move(begin_iterator)),
        end_iterator_(std::move(end_iterator)) {}

  IteratorT begin() const { return begin_iterator_; }
  IteratorT end() const { return end_iterator_; }

 private:
  IteratorT begin_iterator_, end_iterator_;
};

// Convenience function for iterating over sub-ranges.
//
// This provides a bit of syntactic sugar to make using sub-ranges
// in for loops a bit easier. Analogous to std::make_pair().
template <typename T>
iterator_range<T> make_range(T x, T y) {
  return iterator_range<T>(std::move(x), std::move(y));
}

// Converts std::pair<Iter,Iter> to iterator_range<Iter>. E.g.:
//   for (const auto& e : make_range(m.equal_range(k))) ...
template <typename T>
iterator_range<T> make_range(std::pair<T, T> p) {
  return iterator_range<T>(std::move(p.first), std::move(p.second));
}

}  // namespace xabsl

#endif  // XLS_COMMON_ITERATOR_RANGE_H_
