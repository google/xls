// Copyright 2021 The XLS Authors
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

#ifndef XLS_NOC_CONFIG_NG_NETWORK_CONNECTION_H_
#define XLS_NOC_CONFIG_NG_NETWORK_CONNECTION_H_

namespace xls::noc {

// Forward declaration: a port has a reference to a network view and two
// pointers to a port (a source port and a sink port).
class NetworkComponentPort;
class NetworkView;

// A network connection connects two ports.
class NetworkConnection final {
 public:
  NetworkConnection(const NetworkConnection&) = delete;
  NetworkConnection& operator=(const NetworkConnection&) = delete;

  NetworkConnection(const NetworkConnection&&) = delete;
  NetworkConnection& operator=(const NetworkConnection&&) = delete;

  // Connects to a source port. Does not take ownership of the source port. The
  // source port must refer to a valid object that outlives this object.
  // If the connection has an existing source port, then it disconnects from the
  // existing source port prior to connecting to the new source port. If the new
  // source port is nullptr, then it disconnects from existing source port, and
  // no other connections are performed.
  NetworkConnection& ConnectToSourcePort(NetworkComponentPort* source_port);

  // Get source port of the connection. The return value may be nullptr.
  NetworkComponentPort* GetSourcePort() const;

  // Connects to a sink port. Does not take ownership of the sink port. The
  // sink port must refer to a valid object that outlives this object.
  // If the connection has an existing sink port, then it disconnects from the
  // existing sink port prior to connecting to the new sink port. If the new
  // sink port is nullptr, then it disconnects from existing source port, and no
  // other connections are performed.
  NetworkConnection& ConnectToSinkPort(NetworkComponentPort* sink_port);

  // Get sink port of the connection. The return value may be nullptr.
  NetworkComponentPort* GetSinkPort() const;

  // Get network view of the connection.
  NetworkView& GetNetworkView() const;

 private:
  // The private constructor is accessed by the network view.
  friend class NetworkView;
  // network_view: the network view that the connection belongs to. Does not
  // take ownership of the network view. The network view must refer to a valid
  // object that outlives this object.
  explicit NetworkConnection(NetworkView* network_view);

  NetworkComponentPort* source_port_;
  NetworkComponentPort* sink_port_;
  NetworkView& network_view_;
};

}  // namespace xls::noc

#endif  // XLS_NOC_CONFIG_NG_NETWORK_CONNECTION_H_
