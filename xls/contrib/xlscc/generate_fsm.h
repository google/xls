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

#ifndef XLS_CONTRIB_XLSCC_GENERATE_FSM_H_
#define XLS_CONTRIB_XLSCC_GENERATE_FSM_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/contrib/xlscc/tracked_bvalue.h"
#include "xls/contrib/xlscc/translator_types.h"

namespace xlscc {

// This class implements the New FSM in a separate module from the monolithic
// Translator class. GeneratorBase provides necessary common functionality,
// such as error handling.
class NewFSMGenerator : public GeneratorBase {
 public:
  NewFSMGenerator(TranslatorTypeInterface& translator,
                  DebugIrTraceFlags debug_ir_trace_flags);

  // Analyzes the control and data flow graphs, ie function slices and
  // continuations, for a translated function, and generates a "layout"
  // for the FSM to implement it.
  //
  // This layout is then intended to be followed in generating the FSM in
  // XLS IR.
  absl::StatusOr<NewFSMLayout> LayoutNewFSM(const GeneratedFunction& func,
                                            const xls::SourceInfo& body_loc);

 protected:
  absl::Status LayoutNewFSMStates(NewFSMLayout& layout,
                                  const GeneratedFunction& func,
                                  const xls::SourceInfo& body_loc);

  absl::Status LayoutValuesToSaveForNewFSMStates(
      NewFSMLayout& layout, const GeneratedFunction& func,
      const xls::SourceInfo& body_loc);

  absl::Status SetupNewFSMGenerationContext(const GeneratedFunction& func,
                                            NewFSMLayout& layout,
                                            const xls::SourceInfo& body_loc);
  void PrintNewFSMStates(const NewFSMLayout& layout);

 private:
  DebugIrTraceFlags debug_ir_trace_flags_;
};

}  // namespace xlscc

#endif  // XLS_CONTRIB_XLSCC_GENERATE_FSM_H_
