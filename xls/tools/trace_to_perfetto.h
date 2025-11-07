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

#ifndef XLS_TOOLS_TRACE_TO_PERFETTO_H_
#define XLS_TOOLS_TRACE_TO_PERFETTO_H_

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "protos/perfetto/trace/trace.pb.h"
#include "riegeli/records/record_reader.h"
#include "xls/interpreter/trace.pb.h"

namespace xls {

// Converts xls::TracePacketProtos to a perfetto::protos::Trace.
absl::StatusOr<perfetto::protos::Trace> TraceToPerfetto(
    riegeli::RecordReaderBase& xls_trace_reader);

}  // namespace xls

#endif  // XLS_TOOLS_TRACE_TO_PERFETTO_H_
