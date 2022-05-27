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

// Library for rendering IR graphs with the Cytoscape library.
goog.module('xls.graphView');

const irGraph = goog.require('xls.irGraph');
const selectableGraph = goog.require('xls.selectableGraph');

const SELECT_COLOR = 'blue';
const FRONTIER_COLOR = 'orange';
const HOVER_COLOR = 'skyblue';

const CYTOSCAPE_LAYOUT = {
  'name': 'dagre',
  'nodeSep': 20,
  'rankSep': 30,
  'ranker': 'network-simplex',
  'animate': false
};

/**
 * Returns the style object for the Cytoscape graph. This uses selectors
 * analogous to CSS. See http://cytoscape.org for details.
 * @param {number} maxNodeDelay The maximum delay of any node in the graph.
 *     Used for coloring nodes by delay.
 * @return {!Object} The cytoscape style object.
 */
function generateCytoscapeStyle(maxNodeDelay) {
  return [
    {
      // Default node style.
      selector: 'node',
      style: {
        'font-family': 'monospace',
        'font-weight': 'bold',
        'background-color': function(e) {
          let gb = 255;
          if (maxNodeDelay) {
            gb = Math.max(0, 255 - (255 * e.data('xls-delay')) / maxNodeDelay);
          }
          return `rgb(255, ${gb}, ${gb})`;
        },
        'border-width': 2,
        'border-color': 'black',
        'shape': 'roundrectangle',
        'text-valign': 'center',
        'text-halign': 'center',
        'width': 'label',
        'height': 'label',
        'padding': 10,
        'label': 'data(label)'
      },
    },
    {
      // Default edge style.
      selector: 'edge',
      style: {
        'font-family': 'monospace',
        'font-weight': 'bold',
        'text-background-color': HOVER_COLOR,
        'text-background-opacity': 1,
        'text-background-padding': '4px',
        'text-border-opacity': 1,
        'text-border-width': '2px',
        'text-border-color': 'black',
        'curve-style': 'bezier',
        'target-arrow-shape': 'triangle',
        'target-arrow-color': 'darkgray',
        'line-color': 'darkgray',
        'width': function(e) {
          let bit_width = e.data('xls-bit-width');
          return bit_width ? Math.round(1 + 2 * Math.log2(bit_width)) : 1;
        },

        'min-zoomed-font-size': 15,
      }
    },
    {
      // A selected node or edge
      selector: '.selected',
      style: {
        'border-width': 4,
        'border-color': SELECT_COLOR,
        'line-color': SELECT_COLOR,
        'target-arrow-color': SELECT_COLOR,
      }
    },
    {
      // Nodes and edges which are adjacent to selected nodes, but not selected
      // themselves.
      selector: '.frontier',
      style: {
        'line-color': FRONTIER_COLOR,
        'target-arrow-color': FRONTIER_COLOR,

        'border-width': 4,
        'border-color': FRONTIER_COLOR
      }
    },
    {
      // Literal nodes.
      selector: '.opcode-literal',
      style: {'background-color': 'palegreen'}
    },
    {
      // Nodes which are being hovered over. This must be after the 'selected'
      // style because if a node is selected and hovered over we want the hover
      // style.
      selector: '.hover',
      style: {
        'background-color': HOVER_COLOR,
        'z-compound-depth': 'top',
      }
    },
    {
      // Edges which are being hovered over. This must be after the 'selected'
      // style because if an edge is selected and hovered over we want the hover
      // style.
      selector: '.hover-edge',
      style: {
        'line-color': HOVER_COLOR,
        'target-arrow-color': HOVER_COLOR,
        // Brings the hovered-over edge up from behind any other nodes and edges
        // to make it easier to trace.
        'z-compound-depth': 'top',
        'label': function(ele) {
          return ele.data('xls-type');
        },
      }
    },
    // The active class is set on nodes and edges in the mouse-down state. By
    // default it creates a gray overlay. This style eliminates the distracting
    // rectangle.
    {
      selector: 'node:active',
      style: {
        'overlay-opacity': 0,
      }
    },
    {
      selector: 'edge:active',
      style: {
        'overlay-opacity': 0,
      }
    },
    {
      // Edges which span cycle boundaries
      selector: '.crosscycle',
      style: {'line-style': 'dashed', 'line-dash-pattern': [7, 5]}
    },
  ];
}



/**
 * Converts an IrNode to a bare JS object of the form expected by Cytoscape when
 * constructing the graph or adding nodes.
 * @param {!irGraph.IrNode} node
 * @return {!Object}
 */
function toCytoscapeNodeObject(node) {
  let label = '';
  switch (node.opcode) {
    case 'literal':
      // The label of a literal should be its value. The value is given as a
      // hex string. Special case 0 or 1 to omit the leading 0x. Also, if the
      // value is exceptionally long, cur out the middle and replace with
      // '...'.
      let value = node.attributes.value;
      if (!!value.match(/^0x0+$/)) {
        label = '0';
      } else if (!!value.match(/^0x0*1$/)) {
        label = '1';
      } else if (value.length > 11) {
        label = value.slice(0, 4) + '...' + value.slice(-4);
      } else {
        label = value;
      }
      break;
    case 'bit_slice':
      label = `bit_slice[${node.attributes['start']}:${
          node.attributes['start'] + node.attributes['width']}]`;
      break;
    case 'tuple_index':
      label = `tuple_index(${node.attributes['index']})`;
      break;
    case 'param':
      label = node.name;
      break;
    default:
      label = node.opcode;
  }
  return {
    group: 'nodes',
    data: {
      id: node.id,
      label: label,
      'xls-delay': node.attributes['delay_ps'] ? node.attributes['delay_ps'] : 0
    },
    classes: `opcode-${node.opcode}`,
    selectable: false
  };
}

/**
 * Converts an IrEdge to a bare JS object of the form expected by Cytoscape when
 * constructing the graph or adding edges.
 * @param {!irGraph.IrEdge} edge
 * @return {!Object}
 */
function toCytoscapeEdgeObject(edge) {
  return {
    data: {
      group: 'edges',
      id: edge.id,
      source: edge.sourceId,
      target: edge.targetId,
      'xls-type': edge.type,
      'xls-bit-width': edge.bit_width
    },
    selectable: false
  };
}

/**
 * Creates a Cytoscape graph description object from the given SelectableGraph.
 * @param {!selectableGraph.SelectableGraph} selectableGraph
 * @param {boolean} showOnlySelected Whether to include only selected and
 *     frontier nodes and edges in the graph.
 * @return {!Object}
 */
function toCytoscapeGraphObject(selectableGraph, showOnlySelected) {
  let maxNodeDelay = 0;
  selectableGraph.irGraph().nodes().forEach((node, index) => {
    if (node.attributes['delay_ps']) {
      maxNodeDelay = Math.max(maxNodeDelay, node.attributes['delay_ps']);
    }
  });
  let cytoscapeObject = {
    layout: CYTOSCAPE_LAYOUT,
    selectionType: 'additive',
    style: generateCytoscapeStyle(maxNodeDelay)
  };

  let shouldShowNode = (id) => !showOnlySelected ||
      selectableGraph.isNodeSelected(id) ||
      selectableGraph.isNodeOnFrontier(id);

  // Higher performance options for large graphs:
  //  layout: 'grid' or 'breadthfirst'
  //  no curve-style
  // TODO(meheff): Add options screen for selecting these for large
  // graphs. As is, rendering falls over when the graph has 1000's of nodes.
  cytoscapeObject['elements'] = [];
  selectableGraph.irGraph().nodes().forEach((node, index) => {
    if (shouldShowNode(node.id)) {
      cytoscapeObject['elements'].push(toCytoscapeNodeObject(node));
    }
  });
  selectableGraph.irGraph().edges().forEach((edge, index) => {
    if (shouldShowNode(edge.sourceId) && shouldShowNode(edge.targetId)) {
      cytoscapeObject['elements'].push(toCytoscapeEdgeObject(edge));
    }
  });
  return cytoscapeObject;
}

/**
 * Builds and returns a Cytoscape graph object.
 * @param {!Element} graphElement The DOM element to render the graph in.
 * @param {!selectableGraph.SelectableGraph} selectableGraph The IR graph and
 *     selection state to render.
 * @param {boolean} showOnlySelected Whether to show only selected nodes in the
 *     graph.
 * @return {!cy} The Cytoscape graph object.
 */
function buildCytoscapeGraph(graphElement, selectableGraph, showOnlySelected) {
  let cytoscapeObject =
      toCytoscapeGraphObject(selectableGraph, showOnlySelected);
  cytoscapeObject['container'] = graphElement;

  /** @suppress {undefinedVars} */
  let graph = cytoscape(cytoscapeObject);

  // Set initial style on selected/frontier nodes and edges.
  graph.nodes()
      .filter((element, i) => {
        return selectableGraph.isNodeSelected(element.id());
      })
      .addClass('selected');
  graph.nodes()
      .filter((element, i) => {
        return selectableGraph.isNodeOnFrontier(element.id());
      })
      .addClass('frontier');

  graph.edges()
      .filter((element, i) => {
        return selectableGraph.isEdgeSelected(element.id());
      })
      .addClass('selected');
  graph.edges()
      .filter((element, i) => {
        return selectableGraph.isEdgeOnFrontier(element.id());
      })
      .addClass('frontier');
  graph.edges()
      .filter((element, i) => {
        let edge = selectableGraph.irGraph().edge(element.id());
        return selectableGraph.irGraph()
                   .node(edge.sourceId)
                   .attributes['cycle'] !=
            selectableGraph.irGraph().node(edge.targetId).attributes['cycle'];
      })
      .addClass('crosscycle');
  return graph;
}

/**
 * Class for rendering a graph of an XLS IR function.
 */
class GraphView {
  /**
   * @param {!selectableGraph.SelectableGraph} selectableGraphInstance The IR
   *     graph and selection state
   * @param {!Element} graphElement The DOM element to render the graph into.
   * @param {boolean} showOnlySelected Whether to show only selected nodes in
   *     the graph.
   */
  constructor(selectableGraphInstance, graphElement, showOnlySelected) {
    /**
     * @private @const {!selectableGraph.SelectableGraph}
     */
    this.selectableGraph_ = selectableGraphInstance;

    /**
     * The DOM element to hold the graph visualization.
     * @private @const {!Element}
     */
    this.graphElement_ = graphElement;

    /**
     * @private {boolean}
     */
    this.showOnlySelected_ = showOnlySelected;

    /**
     * The Cytoscape graph object.
     * @private @const {?cy}
     */
    this.cyGraph_ = buildCytoscapeGraph(
        this.graphElement_, this.selectableGraph_, this.showOnlySelected_);

    this.setGraphListeners_();

    /**
     * @private {?function(string, boolean)}
     */
    this.hoverOnNodeCallback_ = null;

    /**
     * @private {?function(string, boolean)}
     */
    this.hoverOnEdgeCallback_ = null;

    /**
     * @private {?function(?string, boolean)}
     */
    this.clickCallback_ = null;
  }

  /**
   * @param {?function(string, boolean)} cb
   */
  setHoverOnNodeCallback(cb) {
    this.hoverOnNodeCallback_ = cb;
  }

  /**
   * @param {?function(string, boolean)} cb
   */
  setHoverOnEdgeCallback(cb) {
    this.hoverOnEdgeCallback_ = cb;
  }

  /**
   * @param {?function(?string, boolean)} cb
   */
  setClickCallback(cb) {
    this.clickCallback_ = cb;
  }

  /**
   * Destroys the graph instance.
   */
  destroy() {
    this.cyGraph_.destroy();
  }

  /**
   * Sets the show-only-selected state of the graph to the given value. Nodes
   * will be added/removed from the graph.
   * @param {boolean} value
   */
  setShowOnlySelected(value) {
    if ((!!this.showOnlySelected_) == (!!value)) {
      return;
    }

    this.showOnlySelected_ = value;
    let selectedOrFrontier = (nodeId) =>
        this.selectableGraph_.isNodeSelected(nodeId) ||
        this.selectableGraph_.isNodeOnFrontier(nodeId);
    if (this.showOnlySelected_) {
      // Remove unselected nodes from the graph.
      this.cyGraph_.nodes().filter((n) => !selectedOrFrontier(n.id())).remove();
    } else {
      // Add all nodes back to the graph.
      let elements = [];
      for (let node of this.selectableGraph_.irGraph().nodes()) {
        if (!selectedOrFrontier(node.id)) {
          elements.push(toCytoscapeNodeObject(node));
        }
      }
      for (let edge of this.selectableGraph_.irGraph().edges()) {
        if (!this.selectableGraph_.isEdgeSelected(edge.id) &&
            !this.selectableGraph_.isEdgeOnFrontier(edge.id)) {
          elements.push(toCytoscapeEdgeObject(edge));
        }
      }
      this.setGraphListeners_(this.cyGraph_.add(elements));
    }

    this.cyGraph_.edges()
        .filter((element, i) => {
          let edge = this.selectableGraph_.irGraph().edge(element.id());
          return this.selectableGraph_.irGraph()
                     .node(edge.sourceId)
                     .attributes['cycle'] !=
              this.selectableGraph_.irGraph()
                  .node(edge.targetId)
                  .attributes['cycle'];
        })
        .addClass('crosscycle');

    this.relayoutGraph_();
  }

  /**
   * @param {string} nodeId
   */
  highlightNode(nodeId) {
    this.cyGraph_.getElementById(nodeId).addClass('hover');
  }

  /**
   * @param {string} nodeId
   */
  unhighlightNode(nodeId) {
    this.cyGraph_.getElementById(nodeId).removeClass('hover');
  }

  /**
   * @param {string} edgeId
   */
  highlightEdge(edgeId) {
    let e = this.cyGraph_.getElementById(edgeId);
    e.addClass('hover-edge');
    e.source().addClass('hover');
    e.target().addClass('hover');
  }

  /**
   * @param {string} edgeId
   */
  unhighlightEdge(edgeId) {
    let e = this.cyGraph_.getElementById(edgeId);
    e.removeClass('hover-edge');
    e.source().removeClass('hover');
    e.target().removeClass('hover');
  }

  /**
   * Re-lays out the Cytoscape graph. This should be called after adding or
   * removing nodes from the graph.
   * @param {string=} focusNodeId Optional node to focus on after rendering.
   * @private
   */
  relayoutGraph_(focusNodeId = undefined) {
    let zoomLevel = this.cyGraph_.zoom();
    let layout =
        /** @type {!Object} */ (this.cyGraph_.makeLayout(CYTOSCAPE_LAYOUT));
    layout['one']('layoutstop', (event) => {
      if (focusNodeId !== undefined) {
        let focusNodePosition =
            this.cyGraph_.getElementById(focusNodeId).position();
        this.cyGraph_.zoom({level: zoomLevel, position: focusNodePosition});
      }
    });
    layout['run']();
  }

  /**
   * Sets listeners on the nodes and edges in the graph.
   * @param {!Array<!ele>=} newElements A Cytoscape selection of the graph
   *     elements to add listeners to. If not given then listeners are added to
   *     all elements in the graph.
   * @private
   */
  setGraphListeners_(newElements = undefined) {
    let nodes = newElements != undefined ?
        newElements.filter((e) => e.isNode()) :
        this.cyGraph_.nodes();
    let edges = newElements != undefined ?
        newElements.filter((e) => e.isEdge()) :
        this.cyGraph_.edges();

    nodes.on('mouseover', (ele) => {
      if (this.hoverOnNodeCallback_) {
        this.hoverOnNodeCallback_(ele.target.id(), true);
      }
    });
    nodes.on('mouseout', (ele) => {
      if (this.hoverOnNodeCallback_) {
        this.hoverOnNodeCallback_(ele.target.id(), false);
      }
    });

    edges.on('mouseover', (ele) => {
      if (this.hoverOnEdgeCallback_) {
        this.hoverOnEdgeCallback_(ele.target.id(), true);
      }
    });
    edges.on('mouseout', (ele) => {
      if (this.hoverOnEdgeCallback_) {
        this.hoverOnEdgeCallback_(ele.target.id(), false);
      }
    });

    if (newElements == undefined) {
      // Clicking on the canvas outside of nodes and edges should deselect all
      // nodes.
      this.cyGraph_.on('click', (e) => {
        // HACK: Target id is not defined for clicks outside of nodes and edges.
        if (!e.target.id && this.clickCallback_) {
          // A null nodeID indicates that the click was outside of all nodes.
          this.clickCallback_(/*nodeId=*/ null, e['originalEvent'].ctrlKey);
        }
      });
    }

    nodes.on('click', (e) => {
      if (this.clickCallback_) {
        this.clickCallback_(
            /*nodeId=*/ e.target.id(), e.originalEvent.ctrlKey);
      }
    });
  }


  /**
   * Applies the change object as returned by SelectionGraph::computeChanges_
   * to the textual IR and graph. Selects and deselects nodes and edges as well
   * as adds/removes elements if showOnlySelected is set.
   * @param {!selectableGraph.SelectionChangeSet} changes
   */
  applyChange(changes) {
    if (this.showOnlySelected_) {
      // Add/remove elements to/from the graph. An element should be in the
      // graph iff it does not have a selection state of NONE.
      let newElements = [];

      let newlySelectedNodeIds = [];
      for (const change of changes.nodes) {
        if (change.from == selectableGraph.SelectState.NONE) {
          newElements.push(toCytoscapeNodeObject(
              this.selectableGraph_.irGraph().node(change.id)));
        }
        if (change.to == selectableGraph.SelectState.NONE) {
          this.cyGraph_.getElementById(change.id).remove();
        }
        if (change.to == selectableGraph.SelectState.SELECTED) {
          newlySelectedNodeIds.push(change.id);
        }
      }
      for (const change of changes.edges) {
        if (change.from == selectableGraph.SelectState.NONE) {
          newElements.push(toCytoscapeEdgeObject(
              this.selectableGraph_.irGraph().edge(change.id)));
        }
      }
      if (newElements.length > 0) {
        this.setGraphListeners_(this.cyGraph_.add(newElements));
        this.relayoutGraph_(
            newlySelectedNodeIds.length == 1 ? newlySelectedNodeIds[0] :
                                               undefined);
      }
    }

    // Change style of selected/frontier nodes.
    for (const change of changes.nodes) {
      let cyNode = this.cyGraph_.getElementById(change.id);
      if (change.from == selectableGraph.SelectState.SELECTED) {
        cyNode.removeClass('selected');
      } else if (change.from == selectableGraph.SelectState.FRONTIER) {
        cyNode.removeClass('frontier');
      }

      if (change.to == selectableGraph.SelectState.SELECTED) {
        cyNode.addClass('selected');
      } else if (change.to == selectableGraph.SelectState.FRONTIER) {
        cyNode.addClass('frontier');
      }
    }

    // Change style of selected/frontier edges.
    for (const change of changes.edges) {
      let cyEdge = this.cyGraph_.getElementById(change.id);
      if (change.from == selectableGraph.SelectState.SELECTED) {
        cyEdge.removeClass('selected');
      } else if (change.from == selectableGraph.SelectState.FRONTIER) {
        cyEdge.removeClass('frontier');
      }
      if (change.to == selectableGraph.SelectState.SELECTED) {
        cyEdge.addClass('selected');
      } else if (change.to == selectableGraph.SelectState.FRONTIER) {
        cyEdge.addClass('frontier');
      }
      let edge = this.selectableGraph_.irGraph().edge(change.id);
      if (this.selectableGraph_.irGraph()
              .node(edge.sourceId)
              .attributes['cycle'] !=
          this.selectableGraph_.irGraph()
              .node(edge.targetId)
              .attributes['cycle']) {
        cyEdge.addClass('crosscycle');
      }
    }
  }

  /**
   * Centers the graph view port on the node with the given ID (if it exists in
   * the graph).
   * @param {string} nodeId
   */
  focusOnNode(nodeId) {
    let node = this.cyGraph_.getElementById(nodeId);
    if (!node.empty()) {
      this.cyGraph_.animate({zoom: 2, center: {eles: node}});
    }
  }
}

goog.exportSymbol('xls.GraphView', GraphView);
exports = {GraphView};
