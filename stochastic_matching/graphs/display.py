import json
import uuid

from IPython.display import HTML, display

VIS_LOCATION = 'http://unpkg.com/vis-network/standalone/umd/vis-network.min'
"""Default location of vis-network.js ."""

VIS_OPTIONS = {
    'interaction': {'navigationButtons': True},
    'width': '600px',
    'height': '600px'
}
"""Default options for the vis-network engine."""

HYPER_GRAPH_VIS_OPTIONS = {
    'groups': {
        'HyperEdge': {'fixed': {'x': False}, 'color': {'background': 'black'}, 'shape': 'dot', 'size': 5},
        'Node': {'fixed': {'x': False}}
    },
    'bipartite_display': False
}
"""Default additional options for hypergraphs in the vis-network engine."""

HTML_TEMPLATE = """
<div id="%(name)s"></div>
<script>
require.config({
    paths: {
        vis: '%(vis)s'
    }
});
require(['vis'], function(vis){
var nodes = %(nodes)s;
var edges = %(edges)s;
var data= {
    nodes: nodes,
    edges: edges,
};
var options = %(options)s;
var container = document.getElementById('%(name)s');
var network = new vis.Network(container, data, options);
network.fit({
  maxZoomLevel: 1000});
});
</script>
"""
"""Default template."""


def vis_code(nodes=None, edges=None, options=None, template=None,
             vis=None, div_name=None):
    """
    Create HTML to display a Vis network graph.

    Parameters
    ----------
    nodes: :class:`list` of :class:`dict`
        List the nodes of the graph. Each node is a dictionary with mandatory key `id`.
    edges: :class:`list` of :class:`dict`
        List the edges of the graph. Each node is a dictionary with mandatory keys `from` and `to`.
    options: :class:`dict`, optional
        Options to pass to Vis.
    template: :class:`str`, optional
        Template to use. Default to :obj:`~stochastic_matching.graphs.display.HTML_TEMPLATE`.
    vis: :class:`str`, optional
        Location of vis.js. Default to :obj:`~stochastic_matching.graphs.display.VIS_LOCATION`
    div_name: :class:`str`, optional
        Id of the div that will host the display.

    Returns
    -------
    :class:`str`
        Vis code (HTML by default).

    Examples
    --------
    >>> node_list = [{'id': 0}, {'id': 1}, {'id': 2}, {'id': 3}]
    >>> edge_list = [{'from': 0, 'to': 1}, {'from': 0, 'to': 2},
    ...          {'from': 1, 'to': 3}, {'from': 2, 'to': 3}]
    >>> print(vis_code(nodes=node_list, edges=edge_list)) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    <div id="..."></div>
    <script>
    require.config({
        paths: {
            vis: 'http://unpkg.com/vis-network/standalone/umd/vis-network.min'
        }
    });
    require(['vis'], function(vis){
    var nodes = [{"id": 0}, {"id": 1}, {"id": 2}, {"id": 3}];
    var edges = [{"from": 0, "to": 1}, {"from": 0, "to": 2}, {"from": 1, "to": 3}, {"from": 2, "to": 3}];
    var data= {
        nodes: nodes,
        edges: edges,
    };
    var options = {"interaction": {"navigationButtons": true}, "width": "600px", "height": "600px"};
    var container = document.getElementById('...');
    var network = new vis.Network(container, data, options);
    network.fit({
      maxZoomLevel: 1000});
    });
    </script>
    """
    if div_name is None:
        div_name = str(uuid.uuid4())
    if nodes is None:
        nodes = [{'id': 0}, {'id': 1}]
    if edges is None:
        edges = [{'from': 0, 'to': 1}]
    if options is None:
        options = dict()
    if template is None:
        template = HTML_TEMPLATE
    if vis is None:
        vis = VIS_LOCATION
    dic = {'name': div_name,
           'nodes': json.dumps(nodes),
           'edges': json.dumps(edges),
           'options': json.dumps({**VIS_OPTIONS, **options}),
           'vis': vis}
    return template % dic


def vis_show(nodes=None, edges=None, options=None, template=None,
             vis=None, div_name=None):
    """
    Display a Vis graph (within a IPython / Jupyter session).

    Parameters
    ----------
    nodes: :class:`list` of :class:`dict`
        List the nodes of the graph. Each node is a dictionary with mandatory key `id`.
    edges: :class:`list` of :class:`dict`
        List the edges of the graph. Each node is a dictionary with mandatory keys `from` and `to`.
    options: :class:`dict`, optional
        Options to pass to Vis.
    template: :class:`str`, optional
        Template to use. Default to :obj:`~stochastic_matching.graphs.display.HTML_TEMPLATE`.
    vis: :class:`str`, optional
        Location of vis.js. Default to :obj:`~stochastic_matching.graphs.display.VIS_LOCATION`
    div_name: :class:`str`, optional
        Id of the div that will host the display.

    Returns
    -------
    :class:`~IPython.display.HTML`

    Examples
    --------

    >>> vis_show()
    <IPython.core.display.HTML object>
    """
    # noinspection PyTypeChecker
    display(HTML(vis_code(nodes=nodes, edges=edges, options=options, template=template,
                          vis=vis, div_name=div_name)))
