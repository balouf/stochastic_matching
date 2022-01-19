import json
import uuid
import numpy as np

from IPython.display import HTML, display

from stochastic_matching.common import neighbors


def int_2_str(model, i):
    """
    Parameters
    ----------
    model: :class:`~stochastic_matching.model.Model`
        A stochastic model.
    i: :class:`int`
        Node index.

    Returns
    -------
    :class:`str`
        Name of the node.

    Examples
    --------

    >>> from stochastic_matching.graphs import CycleChain
    >>> diamond = CycleChain()
    >>> int_2_str(diamond, 2)
    '2'
    >>> diamond.names = ['One', 'Two', 'Three', 'Four']
    >>> int_2_str(diamond, 2)
    'Three'
    >>> diamond.names = 'alpha'
    >>> int_2_str(diamond, 2)
    'C'
    """
    if model.names is None:
        return str(i)
    else:
        return model.names[i]


VIS_LOCATION = 'https://unpkg.com/vis-network/standalone/umd/vis-network.min'
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
    }
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

PNG_TEMPLATE = """
<div id="%(name)s"></div>
<img id="canvasImg" alt="Right click to save me!">
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
network.on("afterDrawing", function (ctx) {
    var dataURL = ctx.canvas.toDataURL();
    document.getElementById('canvasImg').src = dataURL;
  });
network.fit({
  maxZoomLevel: 1000});
});
</script>
"""
"""Alternate template with a mirror PNG that ca be saved."""


def vis_code(vis_nodes=None, vis_edges=None, vis_options=None, template=None,
             vis=None, div_name=None):
    """
    Create HTML to display a Vis network graph.

    Parameters
    ----------
    vis_nodes: :class:`list` of :class:`dict`
        List the nodes of the graph. Each node is a dictionary with mandatory key `id`.
    vis_edges: :class:`list` of :class:`dict`
        List the edges of the graph. Each node is a dictionary with mandatory keys `from` and `to`.
    vis_options: :class:`dict`, optional
        Options to pass to Vis.
    template: :class:`str`, optional
        Template to use. Default to :obj:`~stochastic_matching.display.HTML_TEMPLATE`.
    vis: :class:`str`, optional
        Location of vis.js. Default to :obj:`~stochastic_matching.display.VIS_LOCATION`
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
    >>> print(vis_code(vis_nodes=node_list, vis_edges=edge_list)) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    <div id="..."></div>
    <script>
    require.config({
        paths: {
            vis: 'https://unpkg.com/vis-network/standalone/umd/vis-network.min'
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
    if vis_nodes is None:
        vis_nodes = [{'id': 0}, {'id': 1}]
    if vis_edges is None:
        vis_edges = [{'from': 0, 'to': 1}]
    if vis_options is None:
        vis_options = dict()
    if template is None:
        template = HTML_TEMPLATE
    if vis is None:
        vis = VIS_LOCATION
    dic = {'name': div_name,
           'nodes': json.dumps(vis_nodes),
           'edges': json.dumps(vis_edges),
           'options': json.dumps({**VIS_OPTIONS, **vis_options}),
           'vis': vis}
    return template % dic


def vis_show(vis_nodes=None, vis_edges=None, vis_options=None, template=None,
             vis=None, div_name=None):
    """
    Display a Vis graph (within a IPython / Jupyter session).

    Parameters
    ----------
    vis_nodes: :class:`list` of :class:`dict`
        List the nodes of the graph. Each node is a dictionary with mandatory key `id`.
    vis_edges: :class:`list` of :class:`dict`
        List the edges of the graph. Each node is a dictionary with mandatory keys `from` and `to`.
    vis_options: :class:`dict`, optional
        Options to pass to Vis.
    template: :class:`str`, optional
        Template to use. Default to :obj:`~stochastic_matching.display.HTML_TEMPLATE`.
    vis: :class:`str`, optional
        Location of vis.js. Default to :obj:`~stochastic_matching.display.VIS_LOCATION`
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
    display(HTML(vis_code(vis_nodes=vis_nodes, vis_edges=vis_edges, vis_options=vis_options,
                          template=template, vis=vis, div_name=div_name)))


def vis_maker_simple(model, nodes_info=None, edges_info=None):
    """
    The method provides a Vis-ready description of the graph.

    Parameters
    ----------
    model: :class:`~stochastic_matching.model.Model`
        A stochastic model.
    nodes_info: :class:`list` of :class:`dict`
        Additional / overriding attributed for the nodes.
    edges_info: :class:`list` of :class:`dict`
        Additional / overriding attributed for the edges.

    Returns
    -------
    :class:`tuple`
        Node and edge inputs for the vis engine.

    Examples
    ---------

    >>> from stochastic_matching.graphs import Tadpole
    >>> paw = Tadpole()
    >>> vis_maker_simple(paw) # doctest: +NORMALIZE_WHITESPACE
    ([{'id': 0, 'label': '0', 'title': '0'},
    {'id': 1, 'label': '1', 'title': '1'},
    {'id': 2, 'label': '2', 'title': '2'},
    {'id': 3, 'label': '3', 'title': '3'}],
    [{'from': 0, 'to': 1, 'title': '0: (0, 1)', 'label': '(0, 1)'},
    {'from': 0, 'to': 2, 'title': '1: (0, 2)', 'label': '(0, 2)'},
    {'from': 1, 'to': 2, 'title': '2: (1, 2)', 'label': '(1, 2)'},
    {'from': 2, 'to': 3, 'title': '3: (2, 3)', 'label': '(2, 3)'}])

    Nodes can have names.

    >>> paw.names = ['One', 'Two', 'Three', 'Four']
    >>> vis_maker_simple(paw) # doctest: +NORMALIZE_WHITESPACE
    ([{'id': 0, 'label': 'One', 'title': '0: One'},
    {'id': 1, 'label': 'Two', 'title': '1: Two'},
    {'id': 2, 'label': 'Three', 'title': '2: Three'},
    {'id': 3, 'label': 'Four', 'title': '3: Four'}],
    [{'from': 0, 'to': 1, 'title': '0: (One, Two)', 'label': '(One, Two)'},
    {'from': 0, 'to': 2, 'title': '1: (One, Three)', 'label': '(One, Three)'},
    {'from': 1, 'to': 2, 'title': '2: (Two, Three)', 'label': '(Two, Three)'},
    {'from': 2, 'to': 3, 'title': '3: (Three, Four)', 'label': '(Three, Four)'}])

    Pass 'alpha' to name for automatic letter labeling.

    >>> paw.names = 'alpha'
    >>> vis_maker_simple(paw) # doctest: +NORMALIZE_WHITESPACE
    ([{'id': 0, 'label': 'A', 'title': '0: A'},
    {'id': 1, 'label': 'B', 'title': '1: B'},
    {'id': 2, 'label': 'C', 'title': '2: C'},
    {'id': 3, 'label': 'D', 'title': '3: D'}],
    [{'from': 0, 'to': 1, 'title': '0: (A, B)', 'label': '(A, B)'},
    {'from': 0, 'to': 2, 'title': '1: (A, C)', 'label': '(A, C)'},
    {'from': 1, 'to': 2, 'title': '2: (B, C)', 'label': '(B, C)'},
    {'from': 2, 'to': 3, 'title': '3: (C, D)', 'label': '(C, D)'}])
    """
    vis_nodes = [{'id': i, 'label': int_2_str(model, i),
                  'title': f"{i}: {int_2_str(model, i)}" if model.names is not None else str(i)}
                 for i in range(model.n)]
    if nodes_info is not None:
        vis_nodes = [{**internal, **external} for internal, external in zip(vis_nodes, nodes_info)]

    vis_edges = [{'from': int(e[0]), 'to': int(e[1]),
                  'title': f"{j}: ({', '.join([int_2_str(model, i) for i in e])})",
                  'label': f"({', '.join([int_2_str(model, i) for i in e])})"}
                 for j, e in [(j, neighbors(j, model.incidence_csc)) for j in range(model.m)]]
    if edges_info is not None:
        vis_edges = [{**internal, **external} for internal, external in zip(vis_edges, edges_info)]

    return vis_nodes, vis_edges


def vis_maker_hypergraph(model, nodes_info=None, edges_info=None, vis_options=None, bipartite=False):
    """
    The method provides a Vis-ready description of the graph.

    Parameters
    ----------
    model: :class:`~stochastic_matching.model.Model`
        A stochastic model.
    nodes_info: :class:`list` of :class:`dict`, optional
        Additional / overriding attributes for the nodes.
    edges_info: :class:`list` of :class:`dict`, optional
        Additional / overriding attributes for the edges.
    vis_options: :class:`dict`, optional
        Additional / overriding options to pass to the vis engine.
        One specific key, *bipartite_display*,
    bipartite: :class:`bool`, optional
        Tells if the bipartite node/edge structure should be explicitly shown.

    Returns
    -------
    :class:`tuple`
        Inputs for the vis engine.

    Examples
    ---------

    >>> from stochastic_matching.graphs import HyperPaddle
    >>> vis_maker_hypergraph(HyperPaddle(), bipartite=True) # doctest: +NORMALIZE_WHITESPACE
    ([{'id': 0, 'label': '0', 'title': '0', 'x': 0, 'group': 'Node'},
      {'id': 1, 'label': '1', 'title': '1', 'x': 0, 'group': 'Node'},
      {'id': 2, 'label': '2', 'title': '2', 'x': 0, 'group': 'Node'},
      {'id': 3, 'label': '3', 'title': '3', 'x': 0, 'group': 'Node'},
      {'id': 4, 'label': '4', 'title': '4', 'x': 0, 'group': 'Node'},
      {'id': 5, 'label': '5', 'title': '5', 'x': 0, 'group': 'Node'},
      {'id': 6, 'label': '6', 'title': '6', 'x': 0, 'group': 'Node'},
      {'id': 7, 'title': '0: (0, 1)', 'group': 'HyperEdge', 'x': 480},
      {'id': 8, 'title': '1: (0, 2)', 'group': 'HyperEdge', 'x': 480},
      {'id': 9, 'title': '2: (1, 2)', 'group': 'HyperEdge', 'x': 480},
      {'id': 10, 'title': '3: (4, 5)', 'group': 'HyperEdge', 'x': 480},
      {'id': 11, 'title': '4: (4, 6)', 'group': 'HyperEdge', 'x': 480},
      {'id': 12, 'title': '5: (5, 6)', 'group': 'HyperEdge', 'x': 480},
      {'id': 13, 'title': '6: (2, 3, 4)', 'group': 'HyperEdge', 'x': 480}],
    [{'from': 0, 'to': 7, 'title': '0 <-> 0: (0, 1)'},
     {'from': 0, 'to': 8, 'title': '0 <-> 1: (0, 2)'},
     {'from': 1, 'to': 7, 'title': '1 <-> 0: (0, 1)'},
     {'from': 1, 'to': 9, 'title': '1 <-> 2: (1, 2)'},
     {'from': 2, 'to': 8, 'title': '2 <-> 1: (0, 2)'},
     {'from': 2, 'to': 9, 'title': '2 <-> 2: (1, 2)'},
     {'from': 2, 'to': 13, 'title': '2 <-> 6: (2, 3, 4)'},
     {'from': 3, 'to': 13, 'title': '3 <-> 6: (2, 3, 4)'},
     {'from': 4, 'to': 10, 'title': '4 <-> 3: (4, 5)'},
     {'from': 4, 'to': 11, 'title': '4 <-> 4: (4, 6)'},
     {'from': 4, 'to': 13, 'title': '4 <-> 6: (2, 3, 4)'},
     {'from': 5, 'to': 10, 'title': '5 <-> 3: (4, 5)'},
     {'from': 5, 'to': 12, 'title': '5 <-> 5: (5, 6)'},
     {'from': 6, 'to': 11, 'title': '6 <-> 4: (4, 6)'},
     {'from': 6, 'to': 12, 'title': '6 <-> 5: (5, 6)'}],
     {'groups': {'HyperEdge': {'fixed': {'x': True}, 'color': {'background': 'black'},
                               'shape': 'dot', 'size': 5},
     'Node': {'fixed': {'x': True}}}})
    """
    if vis_options is None:
        vis_options = dict()

    vis_options = {**HYPER_GRAPH_VIS_OPTIONS, **vis_options}
    vis_options['groups']['HyperEdge']['fixed']['x'] = bipartite
    vis_options['groups']['Node']['fixed']['x'] = bipartite

    inner_width = round(.8 * vis_options.get('width', 600))

    vis_nodes = [{'id': i,
                  'label': int_2_str(model, i),
                  'title': f"{i}: {int_2_str(model, i)}" if model.names is not None else str(i),
                  'x': 0, 'group': 'Node'} for i in range(model.n)]
    if nodes_info is not None:
        vis_nodes = [{**internal, **external} for internal, external in zip(vis_nodes, nodes_info)]

    vis_edges = [{'id': model.n + j,
                  'title': f"{j}: ({', '.join([int_2_str(model, i) for i in neighbors(j, model.incidence_csc)])})",
                  'group': 'HyperEdge', 'x': inner_width} for j in range(model.m)]
    if edges_info is not None:
        vis_edges = [{**internal, **external} for internal, external in zip(vis_edges, edges_info)]

    vis_links = [{'from': i, 'to': model.n + int(j),
                  'title': f"{int_2_str(model, i)} <-> {vis_edges[j]['title']}"} for i in range(model.n)
                 for j in neighbors(i, model.incidence_csr)]
    return vis_nodes + vis_edges, vis_links, vis_options


def show_graph(model, nodes_info=None, edges_info=None, vis_options=None, bipartite=False, png=False):
    """
    Shows the graph.

    Parameters
    ----------
    model: :class:`~stochastic_matching.model.Model`
        A stochastic model.
    nodes_info: :class:`list` of :class:`dict`
        Additional / overriding attributes for the nodes.
    edges_info: :class:`list` of :class:`dict`
        Additional / overriding attributes for the edges.
    vis_options: :class:`dict`
        Additional / overriding options to pass to the vis engine.
    bipartite: :class:`bool`, optional
        Tells if the bipartite node/edge structure should be explicitly shown.
    png: :class:`bool`
        Make a mirror PNG that can be saved.

    Returns
    -------
    :class:`~IPython.display.HTML`

    Examples
    ---------

    >>> from stochastic_matching.graphs import Tadpole
    >>> paw = Tadpole()
    >>> show_graph(paw)
    <IPython.core.display.HTML object>

    If you need to save your graph, pass the option `png`. It will display a mirror png picture that you can save.

    >>> show_graph(paw, png=True)
    <IPython.core.display.HTML object>
    """
    if png:
        template = PNG_TEMPLATE
    else:
        template = None
    if model.adjacency is not None:
        vis_nodes, vis_edges = vis_maker_simple(model,
                                                nodes_info=nodes_info,
                                                edges_info=edges_info,
                                                )
    else:
        vis_nodes, vis_edges, vis_options = vis_maker_hypergraph(model,
                                                                 nodes_info=nodes_info,
                                                                 edges_info=edges_info,
                                                                 vis_options=vis_options,
                                                                 bipartite=bipartite)
    vis_show(vis_nodes, vis_edges, vis_options, template=template)


def make_kernel_options(model, flow=None):
    """
    Parameters
    ----------
    model: :class:`~stochastic_matching.model.Model`
        A stochastic model.
    flow: :class:`~numpy.ndarray` ot :class:`bool`, optional
        Base flow of the kernel representation. If False, no base flow is displayed, only the kernel shifts.
        If no flow is given, the model base flow is used.

    Returns
    -------
    :class:`list` of :class:`dict`
        An edge description dictionary to pass to :meth:`~stochastic_matching.display.show_kernel`.

    Examples
    --------

    >>> from stochastic_matching.graphs import CycleChain, KayakPaddle
    >>> diamond = CycleChain()
    >>> diamond.base_flow
    array([1., 1., 1., 1., 1.])
    >>> make_kernel_options(diamond)
    [{'label': '1+α1'}, {'label': '1-α1'}, {'label': '1', 'color': 'black'}, {'label': '1-α1'}, {'label': '1+α1'}]
    >>> make_kernel_options(diamond, flow=False)
    [{'label': '+α1'}, {'label': '-α1'}, {'label': '', 'color': 'black'}, {'label': '-α1'}, {'label': '+α1'}]

    # >>> min_flow = problem.optimize_edge(0, -1)
    #  >>> min_flow
    array([0., 2., 1., 2., 0.])
    # >>> problem.kernel_dict(flow=min_flow)
    [{'label': '+α1'}, {'label': '2-α1'}, {'label': '1', 'color': 'black'}, {'label': '2-α1'}, {'label': '+α1'}]

    >>> kayak = KayakPaddle(l=3)
    >>> make_kernel_options(kayak) # doctest: +NORMALIZE_WHITESPACE
    [{'label': '1-α1'}, {'label': '1+α1'}, {'label': '1+α1'},
     {'label': '1-2α1'}, {'label': '1+2α1'}, {'label': '1-2α1'},
     {'label': '1+α1'}, {'label': '1+α1'}, {'label': '1-α1'}]
    """
    d, m = model.kernel.right.shape
    edge_description = [dict() for _ in range(m)]
    for e in range(m):
        label = ""
        for i in range(d):
            alpha = model.kernel.right[i, e]
            if alpha == 0:
                continue
            if alpha == 1:
                label += f"+"
            elif alpha == -1:
                label += f"-"
            else:
                label += f"{alpha:+.3g}"
            label += f"α{i + 1}"
        edge_description[e]['label'] = label
        if not label:
            edge_description[e]['color'] = 'black'
    if flow is False:
        return edge_description
    if flow is None:
        flow = model.base_flow
    for e, dico in enumerate(edge_description):
        if np.abs(flow[e]) > model.tol:
            dico['label'] = f"{flow[e]:.3g}{dico['label']}"
    return edge_description


def show_kernel(model, rates=True, flow=None, *args, **kwargs):
    """
    Parameters
    ----------
    model: :class:`~stochastic_matching.model.Model`
        A stochastic model.
    rates: :class:`bool`
        Display model arrival rates. If False, the node names are displayed.
    flow: :class:`~numpy.ndarray` ot :class:`bool`, optional
        Base flow of the kernel representation. If False, no base flow is displayed, only the kernel shifts.
        If no flow is given, the model base flow is used.
    args: :class:`list`, optional
        Positional parameters for :meth:`~stochastic_matching.display.show_graph`.
    kwargs: :class:`dict`, optional
        Keyword parameters for :meth:`~stochastic_matching.display.show_graph`.

    Returns
    -------
    :class:`~IPython.display.HTML`

    Examples
    --------

    >>> from stochastic_matching.graphs import CycleChain, HyperPaddle
    >>> diamond = CycleChain()
    >>> show_kernel(diamond)
    <IPython.core.display.HTML object>
    >>> show_kernel(diamond, rates=False)
    <IPython.core.display.HTML object>
    >>> candy = HyperPaddle()
    >>> show_kernel(candy)
    <IPython.core.display.HTML object>
    """
    if rates:
        rates = model.rates
        nodes_description = [{'label': f"{rates[i]:.3g}"} for i in range(model.n)]
    else:
        nodes_description = None
    show_graph(model, nodes_info=nodes_description,
               edges_info=make_kernel_options(model, flow=flow), *args, **kwargs)
