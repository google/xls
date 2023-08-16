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

#ifndef XLS_SIMULATION_RENODE_RENODE_PROTOCOL_NATIVE_H_
#define XLS_SIMULATION_RENODE_RENODE_PROTOCOL_NATIVE_H_

// -- Renode native interface --
// The interface implemented here specifically targets
// Antmicro.Renode.Peripherals.Verilated.VerilatedPeripheral class.
// Interface expected by other Verilated.* classes may be a bit different.
//
// The library is loaded by Renode at runtime using dlopen()-like
// mechanism.
//
// -- Renode -> native calls --
// To call functions from the library, Renode uses dlsym()-like mechanism to
// resolve their addresses and calls them directly.
//
// -- Native -> Renode calls --
// To allow the library to call functions exported by Renode, a custom indirect
// call mechanism is used:
//  1. Let's assume the plugin wants to call .NET function 'DotNetFuncName'
//    exported by Renode (marked as `[Export]` in Renode's source).
//  2. For that, the plugin library exports a function with a special name:
//    "renode_external_attach__{DotNetFuncPtrType}__{DotNetFuncName}".
//    This function is dubbed "attacher function" in Renode terminology.
//  3. When loading a plugin, Renode finds all symbols exported by the
//    plugin library that have "renode_external_attach" in their name, and
//    parses them according to the above pattern.
//  4. Renode looks up specified managed .NET function 'DotNetFuncName'
//    and creates a runtime trampoline function that can be called from
//    unmanaged (native) code.
//  5. Renode calls the attacher function and passes it a single argument:
//    a pointer to the trampoline function from the previous step.
//  6. Whenever native code wants to call 'DotNetFuncName', it does so by
//    calling the trampoline via the function pointer received by the attacher.

#include "xls/simulation/renode/renode_protocol.h"

namespace renode {
using ProtocolMessageHandler = void(ProtocolMessage* msg);

// Functions marked as `[Import]` in Renode's source.
extern "C" void initialize_context(const char* context);
extern "C" void initialize_native();
extern "C" void handle_request(::renode::ProtocolMessage* request);
extern "C" void reset_peripheral();

// Attacher functions (used to set up calls to functions marked as `[Export]` in
// Renode's source, see comment above for more details).
extern "C" void renode_external_attach__ActionIntPtr__HandleMainMessage(
    ::renode::ProtocolMessageHandler* fn);
extern "C" void renode_external_attach__ActionIntPtr__HandleSenderMessage(
    ::renode::ProtocolMessageHandler* fn);
extern "C" void renode_external_attach__ActionIntPtr__Receive(
    ::renode::ProtocolMessageHandler* fn);

}  // namespace renode

#endif  // XLS_SIMULATION_RENODE_RENODE_PROTOCOL_NATIVE_H_
