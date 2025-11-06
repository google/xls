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

#ifndef XLS_INTERPRETER_TRACING_OBSERVER_H_
#define XLS_INTERPRETER_TRACING_OBSERVER_H_

#include <memory>
#include <utility>

#include "riegeli/records/record_writer.h"
#include "xls/interpreter/observer.h"
#include "xls/interpreter/trace_recorder.h"
#include "xls/ir/node.h"
#include "xls/ir/value.h"

namespace xls {

class TracingObserver : public EvaluationObserver {
 public:
  explicit TracingObserver(TraceRecorder& recorder) : recorder_(recorder) {}

  void NodeEvaluated(Node* node, const Value& result) override;
  void Tick() override;

 private:
  TraceRecorder& recorder_;
};

// Observer that manages lifetime of a trace recorder and writes the trace to a
// file upon destruction.
class ScopedTracingObserver final : public EvaluationObserver {
 public:
  explicit ScopedTracingObserver(
      std::unique_ptr<riegeli::RecordWriterBase> writer)
      : writer_(std::move(writer)), recorder_(*writer_), observer_(recorder_) {}

  ~ScopedTracingObserver();

  void NodeEvaluated(Node* node, const Value& result) override;
  void Tick() override;

 private:
  std::unique_ptr<riegeli::RecordWriterBase> writer_;
  TraceRecorder recorder_;
  TracingObserver observer_;
};

}  // namespace xls

#endif  // XLS_INTERPRETER_TRACING_OBSERVER_H_
