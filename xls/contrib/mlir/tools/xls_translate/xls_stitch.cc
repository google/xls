// Copyright 2024 The XLS Authors
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

#include "xls/contrib/mlir/tools/xls_translate/xls_stitch.h"

#include <cassert>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "llvm/include/llvm/ADT/STLExtras.h"
#include "llvm/include/llvm/ADT/SmallVectorExtras.h"
#include "llvm/include/llvm/Support/raw_ostream.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"
#include "mlir/include/mlir/IR/Operation.h"
#include "mlir/include/mlir/IR/SymbolTable.h"
#include "mlir/include/mlir/IR/Visitors.h"
#include "mlir/include/mlir/Support/LLVM.h"
#include "mlir/include/mlir/Support/WalkResult.h"
#include "xls/codegen/vast/vast.h"
#include "xls/contrib/mlir/IR/xls_ops.h"
#include "xls/contrib/mlir/util/identifier.h"
#include "xls/ir/source_location.h"

namespace mlir::xls {

namespace {
namespace vast = ::xls::verilog;
using ::llvm::any_of;
using ::llvm::map_to_vector;
using ::xls::SourceInfo;

struct ChannelPortNames {
  std::string data;
  std::string ready;
  std::string valid;
};

ChannelPortNames getChannelPortNames(StringRef chan,
                                     const XlsStitchOptions& options) {
  ChannelPortNames result;
  std::string scrubbed = CleanupIdentifier(chan);
  result.data = absl::StrCat(scrubbed, options.data_port_suffix);
  result.ready = absl::StrCat(scrubbed, options.ready_port_suffix);
  result.valid = absl::StrCat(scrubbed, options.valid_port_suffix);
  return result;
}
ChannelPortNames getChannelPortNames(ChanOp chan,
                                     const XlsStitchOptions& options) {
  return getChannelPortNames(chan.getName(), options);
}

struct ChannelLogicRefs {
  vast::LogicRef* data;
  vast::LogicRef* ready;
  vast::LogicRef* valid;
};

void AddInstantiation(ArrayRef<StringRef> localChannels,
                      ArrayRef<ChanOp> globalChannels, vast::LogicRef* clk,
                      vast::LogicRef* rst, vast::Module* top,
                      std::string eprocName, std::string instanceName,
                      DenseMap<ChanOp, ChannelLogicRefs>& channelRefs,
                      const XlsStitchOptions& options) {
  std::vector<vast::Connection> connections;
  for (auto [local, global] : llvm::zip(localChannels, globalChannels)) {
    ChanOp globalChan = cast<ChanOp>(global);
    ChannelPortNames localNames = getChannelPortNames(local, options);
    connections.push_back(vast::Connection{
        .port_name = localNames.data,
        .expression = channelRefs[globalChan].data,
    });
    connections.push_back(vast::Connection{
        .port_name = localNames.ready,
        .expression = channelRefs[globalChan].ready,
    });
    connections.push_back(vast::Connection{
        .port_name = localNames.valid,
        .expression = channelRefs[globalChan].valid,
    });
  }
  connections.push_back(vast::Connection{
      .port_name = options.clock_signal_name,
      .expression = clk,
  });
  connections.push_back(vast::Connection{
      .port_name = options.reset_signal_name,
      .expression = rst,
  });
  top->Add<vast::Instantiation>(
      SourceInfo(), eprocName, instanceName,
      /*parameters=*/absl::Span<const vast::Connection>(), connections);
}
}  // namespace

LogicalResult XlsStitch(ModuleOp op, llvm::raw_ostream& output,
                        XlsStitchOptions options) {
  vast::VerilogFile f(vast::FileType::kSystemVerilog);
  vast::Module* top = f.AddModule(op.getName().value_or("top"), SourceInfo());
  SymbolTableCollection symbolTable;
  SymbolUserMap symbolUsers(symbolTable, op);

  vast::DataType* i1 = f.BitVectorType(1, SourceInfo());

  vast::LogicRef* clk =
      top->AddInput(options.clock_signal_name, i1, SourceInfo()).value();
  vast::LogicRef* rst =
      top->AddInput(options.reset_signal_name, i1, SourceInfo()).value();

  DenseMap<ChanOp, ChannelLogicRefs> channelRefs;
  auto result = op->walk([&](ChanOp chan) {
    // If the channel is used by a discardable eproc, then it never appears
    // during stitching.
    bool isEphemeralChannel =
        any_of(symbolUsers.getUsers(chan), [](Operation* user) {
          auto eproc = user->getParentOfType<EprocOp>();
          return eproc && eproc.getDiscardable();
        });
    if (isEphemeralChannel) {
      return mlir::WalkResult::advance();
    }

    ChannelPortNames names = getChannelPortNames(chan, options);
    vast::DataType* dataType;
    if (chan.getType().isIntOrFloat()) {
      dataType =
          f.BitVectorType(chan.getType().getIntOrFloatBitWidth(), SourceInfo());
    } else if (auto arrayType = dyn_cast<xls::ArrayType>(chan.getType())) {
      dataType =
          f.PackedArrayType(arrayType.getElementType().getIntOrFloatBitWidth(),
                            arrayType.getShape(), SourceInfo());
    } else {
      op->emitError("unsupported channel type: ") << chan.getType();
      return mlir::WalkResult::interrupt();
    }
    ChannelLogicRefs refs;
    if (chan.getSendSupported() && chan.getRecvSupported()) {
      // Interior port; this becomes a wire.
      refs.data = top->AddWire(names.data, dataType, SourceInfo()).value();
      refs.ready = top->AddWire(names.ready, i1, SourceInfo()).value();
      refs.valid = top->AddWire(names.valid, i1, SourceInfo()).value();
    } else if (chan.getSendSupported()) {
      // Output port; this becomes an output port.
      refs.data = top->AddOutput(names.data, dataType, SourceInfo()).value();
      refs.ready = top->AddInput(names.ready, i1, SourceInfo()).value();
      refs.valid = top->AddOutput(names.valid, i1, SourceInfo()).value();
    } else {
      assert(chan.getRecvSupported());
      // Input port; this becomes an input port.
      refs.data = top->AddInput(names.data, dataType, SourceInfo()).value();
      refs.ready = top->AddOutput(names.ready, i1, SourceInfo()).value();
      refs.valid = top->AddInput(names.valid, i1, SourceInfo()).value();
    }
    channelRefs[chan] = refs;
    return mlir::WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    return failure();
  }

  DenseMap<StringRef, int> instantiationCount;
  op->walk([&](Operation* op) {
    if (!isa<InstantiateEprocOp, InstantiateExternEprocOp>(op)) {
      return;
    }

    if (auto instantiate = dyn_cast<InstantiateEprocOp>(op)) {
      auto localChannels = map_to_vector(
          instantiate.getLocalChannels().getAsRange<FlatSymbolRefAttr>(),
          [](FlatSymbolRefAttr ref) { return ref.getValue(); });
      auto globalChannels = map_to_vector(
          instantiate.getGlobalChannels().getAsRange<FlatSymbolRefAttr>(),
          [&](FlatSymbolRefAttr ref) {
            return symbolTable.lookupNearestSymbolFrom<ChanOp>(op, ref);
          });
      StringRef idealInstanceName =
          instantiate.getName().value_or(instantiate.getEproc());
      CHECK(idealInstanceName == CleanupIdentifier(idealInstanceName))
          << "Invalid name when stitching: " << idealInstanceName.str();
      std::string instanceName =
          absl::StrCat(idealInstanceName.str(), "_",
                       instantiationCount[idealInstanceName]++);
      AddInstantiation(localChannels, globalChannels, clk, rst, top,
                       instantiate.getEproc().str(), instanceName, channelRefs,
                       options);
      return;
    }

    InstantiateExternEprocOp instantiate = cast<InstantiateExternEprocOp>(op);
    auto localChannels = map_to_vector(
        instantiate.getBoundaryChannelNames().getAsRange<StringAttr>(),
        [](StringAttr ref) { return ref.getValue(); });
    auto globalChannels = map_to_vector(
        instantiate.getGlobalChannels().getAsRange<FlatSymbolRefAttr>(),
        [&](FlatSymbolRefAttr ref) {
          return symbolTable.lookupNearestSymbolFrom<ChanOp>(op, ref);
        });
    StringRef idealInstanceName =
        instantiate.getName().value_or(instantiate.getEprocName());
    CHECK(idealInstanceName == CleanupIdentifier(idealInstanceName))
        << "Invalid name when stitching: " << idealInstanceName.str();
    std::string instanceName = absl::StrCat(
        idealInstanceName.str(), "_", instantiationCount[idealInstanceName]++);
    AddInstantiation(localChannels, globalChannels, clk, rst, top,
                     instantiate.getEprocName().str(), instanceName,
                     channelRefs, options);
  });

  output << f.Emit();
  return success();
}

}  // namespace mlir::xls
