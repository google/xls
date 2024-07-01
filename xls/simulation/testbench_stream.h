// Copyright 2023 The XLS Authors
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

#ifndef XLS_SIMULATION_TESTBENCH_STREAM_H_
#define XLS_SIMULATION_TESTBENCH_STREAM_H_

#include <cstdint>
#include <filesystem>  // NOLINT
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/codegen/vast/vast.h"
#include "xls/common/file/named_pipe.h"
#include "xls/common/thread.h"
#include "xls/ir/bits.h"

namespace xls {
namespace verilog {

// `kInput` direction is data flowing to the testbench. `kOutput` is data
// flowing from the testbench.
enum class TestbenchStreamDirection : int8_t { kInput, kOutput };

// An abstraction representing a stream for communicating with Verilog
// testbench.
struct TestbenchStream {
  std::string name;
  TestbenchStreamDirection direction;

  // The macro name referring to the path of the underlying named pipe. The
  // named pipe path is represented in the verilog as a macro to enable run-time
  // binding of the named pipe path.
  std::string path_macro_name;

  // The width of the data to read/write to the testbench.
  int64_t width;
};

// Class for emitting VAST code for reading and writing values to streams.
class VastStreamEmitter {
 public:
  // Creates an emitter. Also, emits into the module the declarations of
  // variables used to interact with the stream (e.g, the file descriptor).
  static VastStreamEmitter Create(const TestbenchStream& stream, Module* m);

  // Emit a $fopen/$fclose call into `block` which opens/closes the file.
  void EmitOpen(StatementBlock* block) const;
  void EmitClose(StatementBlock* block) const;

  // Emit code which reads a value from the pipe and assigns the value to `lhs`.
  void EmitRead(StatementBlock* block, LogicRef* lhs) const;

  // Emit code which writes `value` into the pipe.
  void EmitWrite(StatementBlock* block, Expression* value) const;

 private:
  explicit VastStreamEmitter(const TestbenchStream& stream) : stream_(stream) {}

  const TestbenchStream& stream_;

  // References to declared variables.
  LogicRef* file_descriptor_;
  LogicRef* count_;
  LogicRef* errno_;
  LogicRef* error_string_;
};

// A wrapper around a thread which read/writes data via a stream to/from a
// testbench under simulation.
class TestbenchStreamThread {
 public:
  // Creates a thread for communicating via a stream to a testbench. The thread
  // is not started until the Run* method is called. `named_pipe_path` is the
  // path at which to create the named pipe which underlies the stream
  // communication.
  static absl::StatusOr<TestbenchStreamThread> Create(
      const TestbenchStream& stream,
      const std::filesystem::path& named_pipe_path);

  // Start running a thread which sends data to the testbench. The stream used
  // to create this TestbenchStreamThread must be an input stream.
  //
  // `producer` is a function to call to generate inputs. The producer function
  // will be called until the producer function returns std::nullopt or the
  // simulation terminates.
  using Producer = absl::FunctionRef<std::optional<Bits>()>;
  void RunInputStream(Producer producer);

  // Start running a thread which reads data from the testbench. The stream used
  // to create this TestbenchStreamThread must be an output stream.
  //
  // `consumer` is a function to call with each output written to the stream by
  // the simulation process. If the consumer function returns an error then that
  // error will be returned by Join.
  using Consumer = absl::FunctionRef<absl::Status(const Bits&)>;
  void RunOutputStream(Consumer consumer);

  absl::Status Join();

 private:
  TestbenchStreamThread(const TestbenchStream& stream, NamedPipe named_pipe)
      : stream_(stream), named_pipe_(std::move(named_pipe)) {}

  // Sets `status_` to the given error status if `status_` does not already hold
  // an error code.
  void MaybeSetError(const absl::Status& status);

  const TestbenchStream& stream_;
  NamedPipe named_pipe_;
  std::unique_ptr<Thread> thread_;

  absl::Status status_ = absl::OkStatus();
};

}  // namespace verilog
}  // namespace xls

#endif  // XLS_SIMULATION_TESTBENCH_STREAM_H_
