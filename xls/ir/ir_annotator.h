// Copyright 2026 The XLS Authors
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

#ifndef XLS_IR_IR_ANNOTATOR_H_
#define XLS_IR_IR_ANNOTATOR_H_

#include <concepts>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <vector>

#include "absl/strings/str_cat.h"

namespace xls {

class Node;
class FunctionBase;
class Channel;
class ChannelInterface;
class StateElement;
class ProcInstantiation;
class Register;
class Instantiation;
struct ClockPort;

struct Annotation {
  // If true asks that this node be filtered from the emitted IR.
  bool filter = false;
  // Text placed before the node's ToString. Separated by a ' ' if set.
  std::optional<std::string> prefix;
  // Text placed after the node's ToString. Separated by a ' ' if set.
  std::optional<std::string> suffix;

  std::string Decorate(std::string_view node_str) const {
    std::string res;
    if (prefix.has_value()) {
      absl::StrAppend(&res, *prefix, " ");
    }
    absl::StrAppend(&res, node_str);
    if (suffix.has_value()) {
      absl::StrAppend(&res, " ", *suffix);
    }
    return res;
  }

  static Annotation Combine(const Annotation& a, const Annotation& b) {
    Annotation res;
    res.filter = a.filter || b.filter;
    if (a.prefix.has_value() && b.prefix.has_value()) {
      res.prefix = absl::StrCat(*a.prefix, " ", *b.prefix);
    } else if (a.prefix.has_value()) {
      res.prefix = a.prefix;
    } else {
      res.prefix = b.prefix;
    }
    if (a.suffix.has_value() && b.suffix.has_value()) {
      res.suffix = absl::StrCat(*a.suffix, " ", *b.suffix);
    } else if (a.suffix.has_value()) {
      res.suffix = a.suffix;
    } else {
      res.suffix = b.suffix;
    }
    return res;
  }
};
// Helper to allow users to emit custom annotations in the IR. By default,
// no annotations are emitted.
class IrAnnotator {
 public:
  IrAnnotator() = default;
  virtual ~IrAnnotator() = default;

  // If returned, the nodes will be considered to occur in the given order.
  // Defaults to the order of the nodes in the function base.
  virtual std::optional<std::vector<Node*>> NodeOrder(FunctionBase* fb) const {
    return std::nullopt;
  }
  // Additional information to associate with 'node' in the IR.
  virtual Annotation NodeAnnotation(Node* node) const { return {}; }
  // Additional information to associate with 'channel' in the IR.
  virtual Annotation ChannelAnnotation(Channel* channel) const { return {}; }
  // Additional information to associate with 'channel' in the IR.
  virtual Annotation ChannelInterfaceAnnotation(
      const ChannelInterface* channel) const {
    return {};
  }
  // Additional information to associate with 'state_element' in the IR.
  virtual Annotation StateElementAnnotation(
      const StateElement* state_element) const {
    return {};
  }
  virtual Annotation StateElementInitialValueAnnotation(
      const StateElement* state_element) const {
    return {};
  }
  virtual Annotation ProcInstantiationAnnotation(
      const ProcInstantiation* instantiation) const {
    return {};
  }
  virtual Annotation InstantiationAnnotation(
      const Instantiation* instantiation) const {
    return {};
  }
  virtual Annotation RegisterAnnotation(Register* reg) const { return {}; }
  virtual Annotation ClockPortAnnotation(ClockPort* clock_port) const {
    return {};
  }
};

template <typename... Annotators>
  requires(std::tuple_size_v<std::tuple<Annotators...>> >= 1 &&
           ((std::derived_from<Annotators, IrAnnotator> &&
             // Make sure we don't try to copy a base IrAnnotator.
             // You probably want an IrAnnotatorRef in that case.
             !std::is_same_v<std::remove_cvref_t<Annotators>, IrAnnotator>) &&
            ...))
class IrAnnotatorJoiner : public IrAnnotator {
 public:
  IrAnnotatorJoiner(Annotators... annotators)
      : annotators_(std::make_tuple(std::forward<Annotators>(annotators)...)) {}
  std::optional<std::vector<Node*>> NodeOrder(FunctionBase* fb) const override {
    return std::apply(
        [&](const Annotators&... annotators) {
          return ReduceNodeOrder(fb, annotators...);
        },
        annotators_);
  }
  Annotation NodeAnnotation(Node* node) const override {
    return RunAnnotate(
        [](const IrAnnotator& annotator, Node* node) {
          return annotator.NodeAnnotation(node);
        },
        node);
  }

  Annotation ChannelAnnotation(Channel* channel) const override {
    return RunAnnotate(
        [](const IrAnnotator& annotator, Channel* channel) {
          return annotator.ChannelAnnotation(channel);
        },
        channel);
  }
  Annotation ChannelInterfaceAnnotation(
      const ChannelInterface* channel) const override {
    return RunAnnotate(
        [](const IrAnnotator& annotator, const ChannelInterface* channel) {
          return annotator.ChannelInterfaceAnnotation(channel);
        },
        channel);
  }
  Annotation StateElementAnnotation(
      const StateElement* state_element) const override {
    return RunAnnotate(
        [](const IrAnnotator& annotator, const StateElement* state_element) {
          return annotator.StateElementAnnotation(state_element);
        },
        state_element);
  }
  Annotation StateElementInitialValueAnnotation(
      const StateElement* state_element) const override {
    return RunAnnotate(
        [](const IrAnnotator& annotator, const StateElement* state_element) {
          return annotator.StateElementInitialValueAnnotation(state_element);
        },
        state_element);
  }
  Annotation ProcInstantiationAnnotation(
      const ProcInstantiation* instantiation) const override {
    return RunAnnotate(
        [](const IrAnnotator& annotator,
           const ProcInstantiation* instantiation) {
          return annotator.ProcInstantiationAnnotation(instantiation);
        },
        instantiation);
  }

  Annotation InstantiationAnnotation(
      const Instantiation* instantiation) const override {
    return RunAnnotate(
        [](const IrAnnotator& annotator, const Instantiation* instantiation) {
          return annotator.InstantiationAnnotation(instantiation);
        },
        instantiation);
  }

  Annotation RegisterAnnotation(Register* reg) const override {
    return RunAnnotate(
        [](const IrAnnotator& annotator, Register* reg) {
          return annotator.RegisterAnnotation(reg);
        },
        reg);
  }

  Annotation ClockPortAnnotation(ClockPort* clock_port) const override {
    return RunAnnotate(
        [](const IrAnnotator& annotator, ClockPort* clock_port) {
          return annotator.ClockPortAnnotation(clock_port);
        },
        clock_port);
  }

 private:
  template <typename Accept, typename Val>
  Annotation RunAnnotate(Accept acc, Val node) const {
    return std::apply(
        [&](const std::reference_wrapper<const Annotators>&... ann) {
          return AnnotateImpl(acc, node, ann...);
        },
        annotators_);
  }
  template <typename Accept, typename Val, typename Ann1>
  static Annotation AnnotateImpl(Accept acc, Val node, const Ann1& ann1) {
    return acc(ann1, node);
  }
  template <typename Accept, typename Val, typename Ann1, typename... AnnRest>
  static Annotation AnnotateImpl(Accept acc, Val node, const Ann1& ann1,
                                 const AnnRest&... annRest) {
    return Annotation::Combine(acc(ann1, node),
                               AnnotateImpl(acc, node, annRest...));
  }

  static std::optional<std::vector<Node*>> ReduceNodeOrder(FunctionBase* fb) {
    return std::nullopt;
  }
  template <typename Ann1, typename... AnnRest>
  static std::optional<std::vector<Node*>> ReduceNodeOrder(
      FunctionBase* fb, const Ann1& ann1, const AnnRest&... annRest) {
    std::optional<std::vector<Node*>> res = ann1.NodeOrder(fb);
    if (res.has_value()) {
      return res;
    }
    return ReduceNodeOrder(fb, annRest...);
  }

  std::tuple<Annotators...> annotators_;
};

template <typename... T>
IrAnnotatorJoiner(T...) -> IrAnnotatorJoiner<T...>;

template <typename Base>
  requires(std::derived_from<Base, IrAnnotator>)
class IrAnnotatorRef : public IrAnnotator {
 public:
  explicit IrAnnotatorRef(const Base& base) : base_(base) {}
  std::optional<std::vector<Node*>> NodeOrder(FunctionBase* fb) const override {
    return base_.NodeOrder(fb);
  }
  Annotation NodeAnnotation(Node* node) const override {
    return base_.NodeAnnotation(node);
  }
  Annotation ChannelAnnotation(Channel* channel) const override {
    return base_.ChannelAnnotation(channel);
  }
  Annotation ChannelInterfaceAnnotation(
      const ChannelInterface* channel) const override {
    return base_.ChannelInterfaceAnnotation(channel);
  }
  Annotation StateElementAnnotation(
      const StateElement* state_element) const override {
    return base_.StateElementAnnotation(state_element);
  }
  Annotation StateElementInitialValueAnnotation(
      const StateElement* state_element) const override {
    return base_.StateElementInitialValueAnnotation(state_element);
  }
  Annotation ProcInstantiationAnnotation(
      const ProcInstantiation* instantiation) const override {
    return base_.ProcInstantiationAnnotation(instantiation);
  }
  Annotation InstantiationAnnotation(
      const Instantiation* instantiation) const override {
    return base_.InstantiationAnnotation(instantiation);
  }
  Annotation RegisterAnnotation(Register* reg) const override {
    return base_.RegisterAnnotation(reg);
  }
  Annotation ClockPortAnnotation(ClockPort* clock_port) const override {
    return base_.ClockPortAnnotation(clock_port);
  }

 private:
  const Base& base_;
};

template <typename T>
IrAnnotatorRef(const T&) -> IrAnnotatorRef<T>;

// Helper to make the dump topo sorted.
class TopoSortAnnotator : public IrAnnotator {
 public:
  explicit TopoSortAnnotator(bool topo_sort = true) : topo_sort_(topo_sort) {}
  std::optional<std::vector<Node*>> NodeOrder(FunctionBase* fb) const override;

 private:
  bool topo_sort_;
};
}  // namespace xls

#endif  // XLS_IR_IR_ANNOTATOR_H_
