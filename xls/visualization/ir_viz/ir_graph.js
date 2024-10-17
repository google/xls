/**
 * Copyright 2020 The XLS Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Library for generating IR graphs with the Cytoscape library.
goog.module('xls.irGraph');

/**
 * Abstract representation of an operation in an XLS IR graph.
 */
class IrNode {
  /** @param {!Object} jsonNode */
  constructor(jsonNode) {
    /** @const {string} */
    this.id = jsonNode['id'];

    /** @const {string} */
    this.ir = jsonNode['ir'];

    /** @const {string} */
    this.name = jsonNode['name'];

    /** @const {string} */
    this.opcode = jsonNode['opcode'];

    /** @const {!Object<string, string | number>} */
    this.attributes = jsonNode['attributes'];
  }
}

/**
 * Abstract representation of a dependency between operations in an XLS IR
 * graph.
 */
class IrEdge {
  /** @param {!Object} jsonEdge */
  constructor(jsonEdge) {
    /** @const {string} */
    this.id = jsonEdge['id'];

    /** @const {string} */
    this.sourceId = jsonEdge['source_id'];

    /** @const {string} */
    this.targetId = jsonEdge['target_id'];

    /** @const {string} */
    this.type = jsonEdge['type'];

    /** @const {number} */
    this.bit_width = jsonEdge['bit_width'];
  }
}

/**
 * Abstract representation of the IR graph of an XLS function. Includes nodes
 * (operations), edges (dependencies), and some metadata about each.
 */
class IrGraph {
  /**
   * @param {!Object} jsonGraph An object constructed by JSON.parsing the
   *      "parse IR" response from the server.
   */
  constructor(jsonGraph) {
    /** @private @const {!Object} */
    this.jsonGraph_ = jsonGraph;

    /** @private @const {!Array<!IrNode>} */
    this.nodes_ = [];

    /**
     * Map from node id string to the underlying IR node object.
     * @private @const {!Object<string, !IrNode>}
     */
    this.nodeById_ = {};

    /** @private @const {!Array<!IrEdge>} */
    this.edges_ = [];

    /**
     * Map from edge id string to the underlying IR edge object.
     * @private @const {!Object<string, !IrEdge>}
     */
    this.edgeById_ = {};

    /**
     * Map from node id to the ids of the neighbors (operands and users) of the
     * node.
     * @private @const {!Object<string, !Array<string>>}
     */
    this.neighborNodeIds_ = {};

    for (let n of jsonGraph['nodes']) {
      let node = new IrNode(n);
      this.nodes_.push(node);
      this.nodeById_[node.id] = node;
      this.neighborNodeIds_[node.id] = [];
    }

    // The graph may have no edges ('edges' property undefined).
    for (let e of (jsonGraph['edges'] || [])) {
      let edge = new IrEdge(e);
      this.edges_.push(edge);
      this.edgeById_[edge.id] = edge;

      if (!this.neighborNodeIds_[edge.sourceId].includes(edge.targetId)) {
        this.neighborNodeIds_[edge.sourceId].push(edge.targetId);
      }
      if (!this.neighborNodeIds_[edge.targetId].includes(edge.sourceId)) {
        this.neighborNodeIds_[edge.targetId].push(edge.sourceId);
      }
    }
  }

  /**
   * The graph object constructed by JSON.parsing the "parse IR" response from
   * the server.
   * @return {!Object}
   */
  jsonGraph() {
    return this.jsonGraph_;
  }

  /**
   * Returns the node object with the given id.
   * @param {string} nodeId The id of the node.
   * @return {!IrNode} The node object.
   */
  node(nodeId) {
    return this.nodeById_[nodeId];
  }

  /**
   * Returns the edge object with the given id.
   * @param {string} edgeId The id of the edge.
   * @return {!IrEdge} The edge object.
   */
  edge(edgeId) {
    return this.edgeById_[edgeId];
  }

  /**
   * Returns the array of all node objects in the graph.
   * @return {!Array<!IrNode>} Returns the node objects in the graph.
   */
  nodes() {
    return this.nodes_;
  }

  /**
   * Returns the array of all edge objects in the graph.
   * @return {!Array<!IrEdge>} Returns the edge objects in the graph.
   */
  edges() {
    return this.edges_;
  }

  /**
   * Returns the ids of the neighbors (operands and users) of the node with the
   * given id.
   * @param {string} nodeId
   * @return {!Array<string>} Returns the node ids of the nodes adjacent to the
   * given node in the graph.
   */
  neighborsOf(nodeId) {
    return this.neighborNodeIds_[nodeId];
  }
}

exports = {
  IrNode,
  IrEdge,
  IrGraph
};
