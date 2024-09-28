// Copyright 2022 The XLS Authors
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

#ifndef XLS_INTERPRETER_EVALUATOR_OPTIONS_H_
#define XLS_INTERPRETER_EVALUATOR_OPTIONS_H_

#include "xls/ir/format_preference.h"

namespace xls {

// Options when running the IR interpreter and JIT.
class EvaluatorOptions {
 public:
  // When set, values sent and received on channels are recorded as trace
  // messages.
  EvaluatorOptions& set_trace_channels(bool value) {
    trace_channels_ = value;
    return *this;
  }
  bool trace_channels() const { return trace_channels_; }

  EvaluatorOptions& set_format_preference(FormatPreference value) {
    format_preference_ = value;
    return *this;
  }
  FormatPreference format_preference() const { return format_preference_; }

  EvaluatorOptions& set_support_observers(bool value) {
    support_observers_ = value;
    return *this;
  }
  bool support_observers() const { return support_observers_; }

 private:
  bool trace_channels_ = false;
  FormatPreference format_preference_ = FormatPreference::kDefault;
  bool support_observers_ = false;
};

}  // namespace xls

#endif  // XLS_INTERPRETER_EVALUATOR_OPTIONS_H_
