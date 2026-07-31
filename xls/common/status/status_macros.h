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

// Helper macros and methods to return and propagate errors with `absl::Status`.

#ifndef XLS_COMMON_STATUS_STATUS_MACROS_H_
#define XLS_COMMON_STATUS_STATUS_MACROS_H_

#include "absl/status/status_macros.h"

// Evaluates an expression that produces a `absl::Status`. If the status is not
// ok, returns it from the current function.
//
// For example:
//   absl::Status MultiStepFunction() {
//     XLS_RETURN_IF_ERROR(Function(args...));
//     XLS_RETURN_IF_ERROR(foo.Method(args...));
//     return absl::OkStatus();
//   }
//
// The macro ends with a `StatusBuilder` which allows the returned status
// to be extended with more details.  Any chained expressions after the macro
// will not be evaluated unless there is an error.
//
// For example:
//   absl::Status MultiStepFunction() {
//     XLS_RETURN_IF_ERROR(Function(args...)) << "in MultiStepFunction";
//     XLS_RETURN_IF_ERROR(foo.Method(args...))
//         << "while processing query: " << query.DebugString();
//     return absl::OkStatus();
//   }
//
// If using this macro inside a lambda, you need to annotate the return type
// to avoid confusion between a `StatusBuilder` and an `absl::Status` type.
// E.g.
//
//   []() -> absl::Status {
//     XLS_RETURN_IF_ERROR(Function(args...));
//     XLS_RETURN_IF_ERROR(foo.Method(args...));
//     return absl::OkStatus();
//   }
#define XLS_RETURN_IF_ERROR(expr)                           \
  ABSL_INTERNAL_STATUS_MACROS_IMPL_ELSE_BLOCKER_            \
  if (auto status_macro_internal_adaptor =                  \
          ::absl::status_macro_internal::MacroAdaptor(      \
              (expr), ::absl::SourceLocation::current())) { \
  } else /* NOLINT */                                       \
    return status_macro_internal_adaptor.Consume()

// Executes an expression `rexpr` that returns a `StatusOr<T>`. On OK, moves its
// value into the variable defined by `lhs`, otherwise returns from the current
// function. By default the error status is returned unchanged, but it may be
// modified by an `error_expression`. If there is an error, `lhs` is not
// evaluated; thus any side effects that `lhs` may have only occur in the
// success case.
//
// Interface:
//
//   XLS_ASSIGN_OR_RETURN(lhs, rexpr)
//   XLS_ASSIGN_OR_RETURN(lhs, rexpr, error_expression);
//
// WARNING: if lhs is parenthesized, the parentheses are removed. See examples
// for more details.
//
// WARNING: expands into multiple statements; it cannot be used in a single
// statement (e.g. as the body of an if statement without {})!
//
// Example: Declaring and initializing a new variable (ValueType can be anything
//          that can be initialized with assignment, including references):
//   XLS_ASSIGN_OR_RETURN(ValueType value, MaybeGetValue(arg));
//
// Example: Assigning to an existing variable:
//   ValueType value;
//   XLS_ASSIGN_OR_RETURN(value, MaybeGetValue(arg));
//
// Example: Assigning to an expression with side effects:
//   MyProto data;
//   XLS_ASSIGN_OR_RETURN(*data.mutable_str(), MaybeGetValue(arg));
//   // No field "str" is added on error.
//
// Example: Assigning to a std::unique_ptr.
//   XLS_ASSIGN_OR_RETURN(std::unique_ptr<T> ptr, MaybeGetPtr(arg));
//
// Example: Assigning to a map. Because of C preprocessor
// limitation, the type used in XLS_ASSIGN_OR_RETURN cannot contain comma, so
// wrap lhs in parentheses:
//   XLS_ASSIGN_OR_RETURN((absl::flat_hash_map<Foo, Bar> my_map), GetMap());
// Or use auto if the type is obvious enough:
//   XLS_ASSIGN_OR_RETURN(const auto& my_map, GetMapRef());
//
// Example: Assigning to structured bindings. The same situation with comma as
// in map, so wrap the statement in parentheses.
//   XLS_ASSIGN_OR_RETURN((const auto& [first, second]), GetPair());
//
// If passed, the `error_expression` is evaluated to produce the return
// value. The expression may reference any variable visible in scope, as
// well as a `StatusBuilder` object populated with the error and named by a
// single underscore `_`. The expression typically uses the builder to modify
// the status and is returned directly in manner similar to XLS_RETURN_IF_ERROR.
// The expression may, however, evaluate to any type returnable by the function,
// including (void). For example:
//
// Example: Adjusting the error message.
//   XLS_ASSIGN_OR_RETURN(ValueType value, MaybeGetValue(query),
//                        _ << "while processing " << query.DebugString());
//
// Example: Logging the error on failure.
//   XLS_ASSIGN_OR_RETURN(ValueType value, MaybeGetValue(query), _.LogError());
//
#define XLS_ASSIGN_OR_RETURN(...) ABSL_ASSIGN_OR_RETURN(__VA_ARGS__)

#endif  // XLS_COMMON_STATUS_STATUS_MACROS_H_
