# Copyright 2023 The XLS Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Renode C++ sources/headers."""

BASE_DIR = "plugins/VerilatorIntegrationLibrary/"

def get_socket_lib():
    hdrs = native.glob([
        BASE_DIR + "libs/socket-cpp/Socket/*.h",
        BASE_DIR + "libs/socket-cpp/Socket/*.hpp",
    ])
    srcs = native.glob([
        BASE_DIR + "libs/socket-cpp/Socket/*.c",
        BASE_DIR + "libs/socket-cpp/Socket/*.cpp",
    ])
    return (hdrs, srcs, [])

def get_renode_base():
    hdrs = [
        BASE_DIR + "src/renode.h",
        BASE_DIR + "src/renode_action_enumerators.txt",
        BASE_DIR + "src/renode_imports.h",
        BASE_DIR + "src/renode_imports_generated.h",
        BASE_DIR + "src/buses/bus.h",
    ]
    return (hdrs, [], [])

def get_renode_communication():
    hdrs = native.glob([
        BASE_DIR + "src/communication/*.h",
        BASE_DIR + "src/communication/*.hpp",
    ])
    srcs = native.glob([
        BASE_DIR + "src/communication/*.c",
        BASE_DIR + "src/communication/*.cpp",
    ])
    return (hdrs, srcs, ["renode_base", "socket_lib"])

def get_dpi_lib():
    hdrs = [BASE_DIR + "src/renode_dpi.h"]
    srcs = [BASE_DIR + "src/renode_dpi.cpp"]
    return (hdrs, srcs, ["renode_communication"])

def get_cfu_lib():
    hdrs = [
        BASE_DIR + "src/renode_cfu.h",
        BASE_DIR + "src/buses/cfu.h",
    ]
    srcs = [
        BASE_DIR + "src/renode_cfu.cpp",
        BASE_DIR + "src/buses/cfu.cpp",
    ]
    return (hdrs, srcs, ["renode_base"])

def get_renode_bus_lib():
    hdrs = [BASE_DIR + "src/renode_bus.h"]
    srcs = [BASE_DIR + "src/renode_bus.cpp"]
    return (hdrs, srcs, ["renode_base", "renode_communication"])

def get_apb3_lib():
    hdrs = [BASE_DIR + "src/buses/apb3.h"]
    srcs = [BASE_DIR + "src/buses/apb3.cpp"]
    return (hdrs, srcs, ["renode_base_bus_lib"])

def get_axi_lib():
    hdrs = native.glob([BASE_DIR + "src/buses/axi*.h"])
    srcs = native.glob([BASE_DIR + "src/buses/axi*.cpp"])
    return (hdrs, srcs, ["renode_base_bus_lib"])

def get_wishbone_lib():
    hdrs = native.glob([BASE_DIR + "src/buses/wishbone*.h"])
    srcs = native.glob([BASE_DIR + "src/buses/wishbone*.cpp"])
    return (hdrs, srcs, ["renode_base_bus_lib"])

def get_peripherals_lib():
    hdrs = native.glob([
        BASE_DIR + "src/peripherals/*.h",
        BASE_DIR + "src/peripherals/*.hpp",
    ])
    srcs = native.glob([
        BASE_DIR + "src/peripherals/*.c",
        BASE_DIR + "src/peripherals/*.cpp",
    ])
    return (hdrs, srcs, ["renode_base_bus_lib"])
