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

goog.module('xls.irVisualization');

const graphView = goog.require('xls.graphView');
const irGraph = goog.require('xls.irGraph');
const selectableGraph = goog.require('xls.selectableGraph');

/**
 * Returns the offset of the selection (cursor) within a text element.
 * TODO(meheff): Move this into a separate file and share with hls/xls/ui tool.
 * @param {!Element} node
 * @return {?number}
 */
function getOffsetWithin(node) {
  let sumPrevSiblings = (node) => {
    let total = 0;
    for (let sib of node.parentNode.childNodes) {
      if (sib === node) {
        break;
      }
      total += sib.textContent.length;
    }
    return total;
  };

  let sel = window.getSelection();
  let offset = sel.focusOffset;
  let currentNode = sel.focusNode;
  if (!currentNode) {
    return null;
  }
  while (currentNode !== node) {
    let prevSiblings = sumPrevSiblings(currentNode);
    offset += prevSiblings;
    currentNode = currentNode.parentNode;
    if (!currentNode || currentNode === document) {
      return null;
    }
  }
  return offset;
}

/**
 * Sets the offset of the selection (cursor) within a text element.
 * TODO(meheff): Move this into a separate file and share with hls/xls/ui tool.
 * @param {!Element} node The text element.
 * @param {number} offset The offset within the text element.
 */
function setPositionAtOffset(node, offset) {
  let allTextNodesUnder = (node) => {
    let result = [];
    for (let child of node.childNodes) {
      if (child.nodeType === Node.TEXT_NODE) {
        result.push(child);
      } else {
        result = result.concat(allTextNodesUnder(child));
      }
    }
    return result;
  };

  let textNodes = allTextNodesUnder(node);
  for (let textNode of textNodes) {
    if (offset < textNode.textContent.length) {
      window.getSelection().setPosition(textNode, offset);
      return;
    } else {
      offset -= textNode.textContent.length;
    }
  }
}

/**
 * Sets the function selector UI element to the given options.
 * @param {!Element} selectElement
 * @param {!Array<{name: string, kind: string, id: string}>} functions
 */
function setupFunctionSelector(selectElement, functions) {
  let options = [];
  for (let f of functions) {
    options.push(`<option value="${f.id}">[${f.kind}] ${f.name}</option>`);
  }
  setInnerHtml(selectElement, options.join('\n'));
}

/**
 * Sets the option of the given select UI element to the given function.
 * @param {!Element} selectElement
 * @param {string} functionId
 */
function setFunctionSelector(selectElement, functionId) {
  selectElement.value = functionId;
}

/**
 * Returns the node ids of nodes which should be highlighted when the specified
 * node is highlighted.
 *
 * @param {string} nodeId
 * @return {!Array<string>}
 */
function getCohighlightedNodeIds(nodeId) {
  let nodeDef = document.getElementById(`ir-node-def-${nodeId}`);
  let cohighlightedNodeIds = new Set();
  if (nodeDef && 'stateElement' in nodeDef.dataset) {
    for (let index of nodeDef.dataset.stateElement.split(',')) {
      document.querySelectorAll(`.state-element-${index}`).forEach(e => {
        if ('nodeId' in e.dataset) {
          cohighlightedNodeIds.add(e.dataset.nodeId);
        }
      });
    }
  }
  return Array.from(cohighlightedNodeIds).sort();
}

/**
 * Class for visualizing IR graphs. Manages the text area containing the IR and
 * the element in which the graph is drawn.
 */
class IrVisualizer {
  /**
   * @param {!Element} graphElement DOM element to hold the graph.
   * @param {!Element} irElement Input DOM element holding IR text.
   * @param {!Element} functionSelector Function selector element.
   * @param {?Element=} nodeMetadataElement DOM element to write node metadata
   *     text into.
   */
  constructor(
      graphElement, irElement, functionSelector,
      nodeMetadataElement = undefined) {
    this.graphElement_ = graphElement;
    this.irElement_ = irElement;
    this.functionSelector_ = functionSelector;
    this.nodeMetadataElement_ = nodeMetadataElement;

    /**
     * The graph view object.
     * @private {?graphView.GraphView}
     */
    this.graphView_ = null;

    /**
     * @private {?irGraph.IrGraph}
     */
    this.irGraph_ = null;

    /**
     * The SelectableGraph object.
     * @private {?selectableGraph.SelectableGraph}
     */
    this.graph_ = null;

    /**
     * Whether a parse request is in flight. Only a single request is allowed at
     * a time.
     * @private {boolean}
     */
    this.parseInFlight_ = false;

    /**
     * Callbacks for success/error status when parsing and graphifying the IR.
     * @private {function()|undefined}
     */
    this.sourceOkCallback_ = undefined;

    /** @private {function(string)|undefined} */
    this.sourceErrorCallback_ = undefined;

    /**
     *  Object containing the graph structure for the package. The format is
     *  defined by the proto xls.viz.package.
     *  @private {?Object}
     */
    this.package_ = null;

    /**
     *  The unique identifier of the selected function to view.
     *  @private {?string}
     */
    this.selectedFunctionId_ = null;

    let self = this;
    this.functionSelector_.addEventListener('change', e => {
      if (e.target.value) {
        self.selectFunction(e.target.value);
      }
    });
  }

  /**
   * Sets the callback to call when the IR is successfully parsed.
   * @param {function()} callback
   */
  setSourceOkHandler(callback) {
    this.sourceOkCallback_ = callback;
  }

  /**
   * Sets the callback to call when the IR parsing encountered an error.
   * Callback takes a single argument, the error message.
   * @param {function(string)} callback
   */
  setSourceErrorHandler(callback) {
    this.sourceErrorCallback_ = callback;
  }

  /**
   * Selects the nodes on the critical path through the graph. All other nodes
   * are unselected.
   */
  selectCriticalPath() {
    let criticalPathNodeIds =
        this.irGraph_.nodes()
            .filter((n) => 'on_critical_path' in n.attributes)
            .map((n) => n.id);
    this.applyChange_(this.graph_.selectOnlyNodes(criticalPathNodeIds));
  }

  /**
   * Sets whether to show only the selected and frontier node and elements in
   * the graph depending on parameter 'value'.
   * @param {boolean} value
   */
  setShowOnlySelected(value) {
    if (this.graphView_) {
      this.graphView_.setShowOnlySelected(value);
    }
  }

  /**
   * Highlights the node with the given id in the graph and the textual IR.
   * @param {string} nodeId
   */
  highlightNode(nodeId) {
    document.querySelectorAll('.ir-node-identifier-' + nodeId).forEach(e => {
      e.classList.add('ir-node-identifier-highlighted');
    });

    let cohighlightedNodeIds = getCohighlightedNodeIds(nodeId);
    for (let id of cohighlightedNodeIds) {
      document.querySelectorAll(`.ir-node-def-${id}`).forEach(e => {
        e.classList.add('ir-node-identifier-highlighted');
      });
    }

    if (this.irGraph_ && this.nodeMetadataElement_) {
      let text = '<b>node:</b> ' + this.irGraph_.node(nodeId).ir;
      let delay = this.irGraph_.node(nodeId).attributes['delay_ps'];
      if (delay != null) {
        text += '\n<b>delay:</b> ' + delay + 'ps';
      }
      let known = this.irGraph_.node(nodeId).attributes['known_bits'];
      if (known) {
        text += '\n<b>known bits:</b> ' + known;
      }
      let cycle = this.irGraph_.node(nodeId).attributes['cycle'];
      if (cycle != undefined) {
        text += '\n<b>cycle:</b> ' + cycle;
      }
      let state_param_index =
          this.irGraph_.node(nodeId).attributes['state_param_index'];
      if (state_param_index !== null) {
        text += '\n<b>state param index:</b> ' + state_param_index;
      }
      let initial_value =
          this.irGraph_.node(nodeId).attributes['initial_value'];
      if (initial_value !== null) {
        text += '\n<b>initial state value:</b> ' + initial_value;
      }
      setInnerHtml(this.nodeMetadataElement_, text);
    }
    if (this.graphView_) {
      this.graphView_.highlightNode(nodeId);
      for (let id of cohighlightedNodeIds) {
        this.graphView_.highlightNode(id);
      }
    }
  }

  /**
   * Unhighlights the node with the given id in the graph and the textual IR.
   * @param {string} nodeId
   */
  unhighlightNode(nodeId) {
    document.querySelectorAll('.ir-node-identifier-' + nodeId).forEach(e => {
      e.classList.remove('ir-node-identifier-highlighted');
    });

    let cohighlightedNodeIds = getCohighlightedNodeIds(nodeId);
    for (let id of cohighlightedNodeIds) {
      document.querySelectorAll(`.ir-node-def-${id}`).forEach(e => {
        e.classList.remove('ir-node-identifier-highlighted');
      });
    }

    if (this.irGraph_ && this.nodeMetadataElement_) {
      this.nodeMetadataElement_.textContent = '';
    }
    if (this.graphView_) {
      this.graphView_.unhighlightNode(nodeId);
      for (let id of cohighlightedNodeIds) {
        this.graphView_.unhighlightNode(id);
      }
    }
  }

  /**
   * Highlights the edge with the given id in the graph and the textual IR.
   * @param {string} edgeId
   */
  highlightEdge(edgeId) {
    let srcId = this.irGraph_.edge(edgeId).sourceId;
    let tgtId = this.irGraph_.edge(edgeId).targetId;
    document.getElementById(`ir-node-def-${srcId}`)
        .classList.add('ir-node-identifier-highlighted');
    document.querySelectorAll(`.ir-edge-${srcId}-${tgtId}`).forEach(e => {
      e.classList.add('ir-node-identifier-highlighted');
    });
    if (this.graphView_) {
      this.graphView_.highlightEdge(edgeId);
    }
    // Highlight the source node as well so the metadata about the value carried
    // by the edge is shown in the metadata box.
    this.highlightNode(srcId);
  }

  /**
   * Highlights the edge with the given id in the graph and the textual IR.
   * @param {string} edgeId
   */
  unhighlightEdge(edgeId) {
    let srcId = this.irGraph_.edge(edgeId).sourceId;
    let tgtId = this.irGraph_.edge(edgeId).targetId;
    document.getElementById(`ir-node-def-${srcId}`)
        .classList.remove('ir-node-identifier-highlighted');
    document.querySelectorAll(`.ir-edge-${srcId}-${tgtId}`).forEach(e => {
      e.classList.remove('ir-node-identifier-highlighted');
    });
    if (this.graphView_) {
      this.graphView_.unhighlightEdge(edgeId);
    }
    this.unhighlightNode(srcId);
  }

  /**
   * Applies the change object as returned by SelectionGraph::computeChanges_
   * to the textual IR and graph. Selects and deselects nodes and edges as well
   * as adds/removes elements if showOnlySelected is set.
   * @param {!selectableGraph.SelectionChangeSet} changes
   * @private
   */
  applyChange_(changes) {
    // (De)select identifier elements in the textual IR.
    for (const change of changes.nodes) {
      if (change.from == selectableGraph.SelectState.SELECTED) {
        document.querySelectorAll(`.ir-node-identifier-${change.id}`)
            .forEach(e => {
              e.classList.remove('ir-node-identifier-selected');
            });
      }
      if (change.to == selectableGraph.SelectState.SELECTED) {
        document.querySelectorAll(`.ir-node-identifier-${change.id}`)
            .forEach(e => {
              e.classList.add('ir-node-identifier-selected');
            });
      }
    }

    if (this.graphView_) {
      this.graphView_.applyChange(changes);
    }
  }

  /**
   * Set the selection state of the node with the given id to 'value'.
   * @param {string} nodeId
   * @param {boolean} value
   */
  selectNode(nodeId, value) {
    this.applyChange_(this.graph_.selectNode(nodeId, value));
  }

  /**
   * Sets the function to visualize to the function with the given id.
   * @param {string} functionId
   */
  selectFunction(functionId) {
    this.clearGraph();
    this.selectedFunctionId_ = null;

    // Scrape the function/proc/block names from the package.
    let graph = null;
    for (let func of this.package_['function_bases']) {
      if (functionId == func['id']) {
        graph = func;
      }
    }
    if (graph == null) {
      return;
    }
    this.irGraph_ = new irGraph.IrGraph(graph);
    this.graph_ = new selectableGraph.SelectableGraph(this.irGraph_);
    this.highlightIr_(graph);
    this.setIrTextListeners_();
    this.selectedFunctionId_ = functionId;

    if (this.graphView_) {
      this.draw(document.getElementById('only-selected-checkbox').checked);
    }

    // Scroll the start of the selected function definition into view.
    document.getElementById(`ir-function-def-${functionId}`).scrollIntoView();

    // Unselect all functions. This will grey out the IR text of all functions.
    document.querySelectorAll('.ir-function').forEach(e => {
      e.classList.add('ir-function-unselected');
    });

    // For the selected function, add the class `ir-function-selected` which
    // will display this function in normal (not greyed out) text.
    document.getElementById(`ir-function-${functionId}`)
        .classList.remove('ir-function-unselected');
    document.getElementById(`ir-function-${functionId}`)
        .classList.add('ir-function-selected');
  }

  /**
   * Sets various listeners for hovering over and selecting identifiers in the
   * IR text.
   */
  setIrTextListeners_() {
    let self = this;
    document.querySelectorAll('.ir-node-identifier').forEach(elem => {
      elem.addEventListener('mouseenter', e => {
        if (e.target.dataset.functionId == self.selectedFunctionId_) {
          self.highlightNode(
              /** @type {string} */ (e.target.dataset.nodeId));
        }
      });
      elem.addEventListener('mouseleave', e => {
        if (e.target.dataset.functionId == self.selectedFunctionId_) {
          self.unhighlightNode(
              /** @type {string} */ (e.target.dataset.nodeId));
        }
      });
      elem.addEventListener('click', e => {
        if (e.target.dataset.functionId != self.selectedFunctionId_) {
          // Ignore clicks on nodes which are not in the selected function.
          return;
        }
        let nodeId = /** @type {string} */ (e.target.dataset.nodeId);
        if (e.ctrlKey && self.graphView_) {
          // If the control key is down. Zoom in on the node in the graph.
          self.graphView_.focusOnNode(nodeId);
        } else {
          // Toggle the selection state.
          self.selectNode(nodeId, !self.graph_.isNodeSelected(nodeId));
        }
      });
    });

    document.querySelectorAll('.ir-function-identifier').forEach(elem => {
      elem.addEventListener('click', e => {
        // Clicking on a function identifier selects the function for viewing.
        if (e.target.dataset.identifier == self.selectedFunctionId_) {
          return;
        }
        setFunctionSelector(
            self.functionSelector_, e.target.dataset.identifier);
        self.selectFunction(e.target.dataset.identifier);
      });
    });
  }

  /**
   * Highlights the IR text source using the JSON graphified IR. This puts the
   * marked up IR from the server into the IR text element.
   * @param {!Object} jsonGraph
   * @private
   */
  highlightIr_(jsonGraph) {
    let focusOffset = getOffsetWithin(this.irElement_);
    setInnerHtml(this.irElement_, this.package_['ir_html']);
    if (focusOffset != null) {
      setPositionAtOffset(this.irElement_, focusOffset);
    }
  }

  /**
   * Sends the IR to the server for parsing and, if successful, highlights the
   * IR.
   * @param {?function()} cb
   */
  parseAndHighlightIr(cb) {
    if (this.parseInFlight_) {
      return;
    }
    this.parseInFlight_ = true;
    let text = this.irElement_.textContent;
    let xmr = new XMLHttpRequest();
    xmr.open('POST', '/graph');
    let self = this;
    xmr.addEventListener('load', function() {
      if (xmr.status < 200 || xmr.status >= 400) {
        return;
      }

      // TODO: define a type for the graph object.
      let response = /** @type {!Object} */ (JSON.parse(xmr.responseText));
      self.parseInFlight_ = false;
      if (self.irElement_.textContent != text) {
        return self.parseAndHighlightIr(cb);
      }

      self.package_ = null;
      if (response['error_code'] == 'ok') {
        if (self.sourceOkCallback_) {
          self.sourceOkCallback_();
        }
        self.package_ = response['graph'];

        // Fill in the names and ids of function in the select element.
        let functions = [];
        for (let func of self.package_['function_bases']) {
          functions.push(
              {name: func['name'], kind: func['kind'], id: func['id']});
        }
        setupFunctionSelector(self.functionSelector_, functions);

        // Select the function entry if entry is specified.
        if (self.package_['entry_id']) {
          setFunctionSelector(
              self.functionSelector_, self.package_['entry_id']);
          self.selectFunction(self.package_['entry_id']);
        }
      } else {
        self.clearGraph();
        if (!!self.sourceErrorCallback_) {
          self.sourceErrorCallback_(response['message']);
        }
        self.package_ = null;
        let focusOffset = getOffsetWithin(self.irElement_);
        setInnerHtml(self.irElement_, self.irElement_.textContent);
        if (focusOffset) {
          setPositionAtOffset(self.irElement_, focusOffset);
        }
      }
      if (cb) {
        cb();
      }
    });
    let data = new FormData();
    data.append('text', text);
    xmr.send(data);
  }

  /**
   * Clears the fields containing graph elements (cyctoscape and IR graph) and
   * clears the visualization window.
   */
  clearGraph() {
    self.irGraph_ = null;
    self.graph_ = null;
    if (self.graphView_) {
      self.graphView_.destroy();
      self.graphView_ = null;
    }
  }

  /**
   * Creates and renders a graph of the IR using Cytoscape. This should only be
   * called after the IR has been successfully parsed and the SelectionGraph
   * constructed.
   * @param {boolean} showOnlySelected Whether to include only selected and
   *     frontier nodes and edges in the graph.
   */
  draw(showOnlySelected) {
    if (!this.graph_) {
      console.log(
          'SelectableGraph not yet constructed. Parse error or slow response?');
      return;
    }
    this.graphView_ = new graphView.GraphView(
        this.graph_, this.graphElement_, showOnlySelected);

    this.graphView_.setHoverOnNodeCallback((nodeId, hoverOn) => {
      if (hoverOn) {
        this.highlightNode(nodeId);
      } else {
        this.unhighlightNode(nodeId);
      }
    });
    this.graphView_.setHoverOnEdgeCallback((edgeId, hoverOn) => {
      if (hoverOn) {
        this.highlightEdge(edgeId);
      } else {
        this.unhighlightEdge(edgeId);
      }
    });
    this.graphView_.setClickCallback((nodeId, ctrlPressed) => {
      if (nodeId) {
        if (ctrlPressed) {
          // Scroll the node into view in the IR text window.
          document.getElementById(`ir-node-def-${nodeId}`).scrollIntoView();
        } else {
          // Toggle the selection state.
          this.selectNode(nodeId, !this.graph_.isNodeSelected(nodeId));
        }
      } else {
        // Clicking on the graph outside of a node (nodeId == null) clears the
        // selection state.
        this.applyChange_(this.graph_.selectOnlyNodes([]));
      }
    });
  }
}

exports = {IrVisualizer};
