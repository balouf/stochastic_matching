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

    >>> import stochastic_matching as sm
    >>> diamond = sm.CycleChain()
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


def default_description(model):
    """
    Parameters
    ----------
    model: :class:`stochastic_matching.model.Model`
        Model to visualize.

    Returns
    -------
    nodes_info :class:`list` of `dict`
        Skeleton node description.
    nodes_info :class:`list` of `dict`
        Skeleton node description.

    Examples
    --------

    >>> import stochastic_matching as sm
    >>> diamond = sm.CycleChain()
    >>> default_description(diamond) # doctest: +NORMALIZE_WHITESPACE
    ([{'id': 0, 'label': '', 'title': '0'},
      {'id': 1, 'label': '', 'title': '1'},
      {'id': 2, 'label': '', 'title': '2'},
      {'id': 3, 'label': '', 'title': '3'}],
    [{'title': '0: (0, 1)', 'label': ''},
     {'title': '1: (0, 2)', 'label': ''},
     {'title': '2: (1, 2)', 'label': ''},
     {'title': '3: (1, 3)', 'label': ''},
     {'title': '4: (2, 3)', 'label': ''}])
    >>> diamond.names = 'alpha'
    >>> default_description(diamond) # doctest: +NORMALIZE_WHITESPACE
    ([{'id': 0, 'label': '', 'title': '0: A'},
      {'id': 1, 'label': '', 'title': '1: B'},
      {'id': 2, 'label': '', 'title': '2: C'},
      {'id': 3, 'label': '', 'title': '3: D'}],
    [{'title': '0: (A, B)', 'label': ''},
     {'title': '1: (A, C)', 'label': ''},
     {'title': '2: (B, C)', 'label': ''},
     {'title': '3: (B, D)', 'label': ''},
     {'title': '4: (C, D)', 'label': ''}])
    """
    nodes_info = [{'id': i, 'label': '',
                  'title': f"{i}: {int_2_str(model, i)}" if model.names is not None else str(i)}
                 for i in range(model.n)]
    edges_info = [{'title': f"{j}: ({', '.join([int_2_str(model, i) for i in e])})",
                   'label': ''}
                     for j, e in [(j, neighbors(j, model.incidence_csc)) for j in range(model.m)]]
    return nodes_info, edges_info


def vis_converter(model, nodes_info, edges_info):
    """
    Parameters
    ----------
    model: :class:`~stochastic_matching.model.Model`
        Model to visualize.
    nodes_info: :class:`list` of :class:`dict`
        Description of nodes.
    edges_info: :class:`list` of :class:`dict`
        Description of edges.

    Returns
    -------
    vis_nodes: :class:`list` of :class:`dict`
        Description of the nodes that will be displayed in vis.
        If the graph is simple, this is just the input nodes.
        For hypergraphs, both nodes and hyperedges are displayed as nodes in vis.
    vis_edges: :class:`list` of :class:`dict`
        Description of the edges that will be displayed in vis.
        If the graph is simple, this is just the input edges, with endpoints info added for vis.
        For hypergraphs, each edge in vis links a node and a hyperedge..


    Examples
    --------
    >>> import stochastic_matching as sm
    >>> diamond = sm.CycleChain()
    >>> nodes, edges = default_description(diamond)
    >>> vis_converter(diamond, nodes, edges) # doctest: +NORMALIZE_WHITESPACE
    ([{'id': 0, 'label': '', 'title': '0'},
      {'id': 1, 'label': '', 'title': '1'},
      {'id': 2, 'label': '', 'title': '2'},
      {'id': 3, 'label': '', 'title': '3'}],
     [{'title': '0: (0, 1)', 'label': '', 'from': 0, 'to': 1},
      {'title': '1: (0, 2)', 'label': '', 'from': 0, 'to': 2},
      {'title': '2: (1, 2)', 'label': '', 'from': 1, 'to': 2},
      {'title': '3: (1, 3)', 'label': '', 'from': 1, 'to': 3},
      {'title': '4: (2, 3)', 'label': '', 'from': 2, 'to': 3}])
    >>> diamond.adjacency = None
    >>> vis_converter(diamond, nodes, edges) # doctest: +NORMALIZE_WHITESPACE
    ([{'id': 0, 'label': '', 'title': '0', 'x': 0, 'group': 'Node'},
      {'id': 1, 'label': '', 'title': '1', 'x': 0, 'group': 'Node'},
      {'id': 2, 'label': '', 'title': '2', 'x': 0, 'group': 'Node'},
      {'id': 3, 'label': '', 'title': '3', 'x': 0, 'group': 'Node'},
      {'title': '0: (0, 1)', 'label': '', 'from': 0, 'to': 1, 'id': 4, 'group': 'HyperEdge', 'x': 600},
      {'title': '1: (0, 2)', 'label': '', 'from': 0, 'to': 2, 'id': 5, 'group': 'HyperEdge', 'x': 600},
      {'title': '2: (1, 2)', 'label': '', 'from': 1, 'to': 2, 'id': 6, 'group': 'HyperEdge', 'x': 600},
      {'title': '3: (1, 3)', 'label': '', 'from': 1, 'to': 3, 'id': 7, 'group': 'HyperEdge', 'x': 600},
      {'title': '4: (2, 3)', 'label': '', 'from': 2, 'to': 3, 'id': 8, 'group': 'HyperEdge', 'x': 600}],
     [{'from': 0, 'to': 4, 'title': '0 <-> 0: (0, 1)'},
      {'from': 0, 'to': 5, 'title': '0 <-> 1: (0, 2)'},
      {'from': 1, 'to': 4, 'title': '1 <-> 0: (0, 1)'},
      {'from': 1, 'to': 6, 'title': '1 <-> 2: (1, 2)'},
      {'from': 1, 'to': 7, 'title': '1 <-> 3: (1, 3)'},
      {'from': 2, 'to': 5, 'title': '2 <-> 1: (0, 2)'},
      {'from': 2, 'to': 6, 'title': '2 <-> 2: (1, 2)'},
      {'from': 2, 'to': 8, 'title': '2 <-> 4: (2, 3)'},
      {'from': 3, 'to': 7, 'title': '3 <-> 3: (1, 3)'},
      {'from': 3, 'to': 8, 'title': '3 <-> 4: (2, 3)'}])
    >>> candy = sm.HyperPaddle()
    >>> vis_nodes, vis_edges = vis_converter(candy, *default_description(candy))
    >>> vis_nodes[2]
    {'id': 2, 'label': '', 'title': '2', 'x': 0, 'group': 'Node'}
    >>> vis_nodes[13]
    {'title': '6: (2, 3, 4)', 'label': '', 'id': 13, 'group': 'HyperEdge', 'x': 600}
    >>> vis_edges[6]
    {'from': 2, 'to': 13, 'title': '2 <-> 6: (2, 3, 4)'}
    """
    simple = model.adjacency is not None
    if simple:
        vis_nodes = nodes_info
        vis_edges = edges_info
        for e, dico in enumerate(vis_edges):
            endpoints = neighbors(e, model.incidence_csc)
            dico['from'] = int(endpoints[0])
            dico['to'] = int(endpoints[1])
        return vis_nodes, vis_edges
    else:
        vis_nodes_1 = nodes_info
        vis_nodes_2 = edges_info
        for d in vis_nodes_1:
            d['x'] = 0
            d['group'] = 'Node'
        for j, d in enumerate(vis_nodes_2):
            d['id'] = model.n + j
            d['group'] = 'HyperEdge'
            d['x'] = 600
        vis_edges = [{'from': i, 'to': model.n + int(j),
                      'title': f"{vis_nodes_1[i]['title']} <-> {vis_nodes_2[j]['title']}"} for i in range(model.n)
                     for j in neighbors(i, model.incidence_csr)]
        return vis_nodes_1+vis_nodes_2, vis_edges


def info_maker(model, disp_rates=True, disp_flow=True, flow=None, disp_kernel=False, disp_zero=True,
               check_flow=False, check_tolerance=1e-2):
    """
    Parameters
    ----------
    model: :class:`~stochastic_matching.model.Model`
        Model to visualize.
    disp_rates: :class:`bool`, optional
        Labels the nodes with their rates. Otherwise the names are used.
    disp_flow: :class:`bool`, optional
        Label the edges with the given flow.
    flow: :class:`~numpy.ndarray`, optional
        Flow to use. If None, the base flow will be used. If not None, overrides `disp_flow`.
    disp_kernel: :class:`bool`, optional
        Display the kernel basis on the edges. Compatible with the display of a flow.
    disp_zero: :class:`bool`, optional
        If False, do not label the edge with a null flow.
    check_flow: :class:`bool`, optional
        If True, color the edges with their positivity and the nodes with their compliance to the conservation law.
    check_tolerance: :class:`float`, optional
        Relative error when checking conservation law on nodes.
        For simulations, a relatively high value is recommended, for example 1e-2.

    Returns
    -------
    nodes_info: :class:`list` of :class:`dict`
        Description of nodes.
    edges_info: :class:`list` of :class:`dict`
        Description of edges.

    Examples
    ---------

    It is probably best to play a bit with the options, but the following examples should give the general idea.

    We start with the so-called *pyramid* graph (the names comes from one of its kernel,
    and not from the shape of the graph itself).

    >>> import stochastic_matching as sm
    >>> pyramid = sm.Pyramid(names='alpha')

    By default, the label of a node (its displayed name) is its arrival rate,
    and the label of an edge is its Moore_penrose flow.

    >>> n_i, e_i = info_maker(pyramid)
    >>> n_i[3]
    {'id': 3, 'label': '2', 'title': '3: D'}
    >>> e_i[2]
    {'title': '2: (B, C)', 'label': '1'}

    We can disable the display of the arrival rates, so the actual name of the node will be displayed.

    >>> n_i, e_i = info_maker(pyramid, disp_rates=False)
    >>> n_i[3]
    {'id': 3, 'label': 'D', 'title': '3: D'}

    We ask for no label on edges.

    >>> n_i, e_i = info_maker(pyramid, disp_flow=False)
    >>> e_i[2]
    {'title': '2: (B, C)', 'label': ''}

    We can set custom weights on the edges, for instance use a different flow vector.

    >>> flow = np.array([0., 2., 0., 3., 1., 1., 0., 2., 0., 2., 0., 1., 1.])
    >>> n_i, e_i = info_maker(pyramid, flow=flow)
    >>> assert np.allclose([float(e_i[i]['label']) for i in range(pyramid.m)], flow)

    We can ask for the kernel basis to be indicated as well.

    >>> n_i, e_i = info_maker(pyramid, flow=flow, disp_kernel=True)
    >>> e_i[2]
    {'title': '2: (B, C)', 'label': '0+α1'}
    >>> e_i[3]
    {'title': '3: (B, F)', 'label': '3-α1+α2-α3'}

    We can remove the flow to have just the kernel basis.

    >>> n_i, e_i = info_maker(pyramid, disp_flow=False, disp_kernel=True)
    >>> e_i[2]
    {'title': '2: (B, C)', 'label': '+α1'}
    >>> e_i[3]
    {'title': '3: (B, F)', 'label': '-α1+α2-α3'}

    By asking null values to be silent, we get avoid things like `0+...`.

    >>> n_i, e_i = info_maker(pyramid, flow=flow, disp_zero=False, disp_kernel=True)
    >>> e_i[2]
    {'title': '2: (B, C)', 'label': '+α1'}
    >>> e_i[3]
    {'title': '3: (B, F)', 'label': '3-α1+α2-α3'}

    If we ask to check the flow, null edges are displayed in orange.

    >>> n_i, e_i = info_maker(pyramid, flow=flow, check_flow=True)
    >>> e_i[2]
    {'title': '2: (B, C)', 'label': '0', 'color': 'orange'}

    Note that the kernel basis is not necessarily +/- 1, even on simple graphs.

    >>> kayak = sm.KayakPaddle()
    >>> n_i, e_i = info_maker(kayak, disp_kernel=True)
    >>> e_i[3]
    {'title': '3: (2, 3)', 'label': '1+2α1'}

    When kernel is displayed, edges that are not part of any kernel are shown in black.

    >>> diamond = sm.CycleChain()
    >>> n_i, e_i = info_maker(diamond, disp_kernel=True)
    >>> e_i[2]
    {'title': '2: (1, 2)', 'label': '1', 'color': 'black'}

    Nodes that do not check the conservation law and negative edges are shown in red.

    >>> n_i, e_i = info_maker(diamond, flow=[-1]*5, check_flow=True)
    >>> n_i[2]
    {'id': 2, 'label': '3', 'title': '2', 'color': 'red'}
    >>> e_i[2]
    {'title': '2: (1, 2)', 'label': '-1', 'color': 'red'}
    """
    nodes_info, edges_info = default_description(model)
    if flow is not None:
        disp_flow = True
    for i, node in enumerate(nodes_info):
        if disp_rates:
            node['label'] = f"{model.rates[i]:.3g}"
        else:
            node['label'] = int_2_str(model, i)
    if disp_flow:
        if flow is None:
            flow = model.base_flow
        for e, edge in enumerate(edges_info):
            if np.abs(flow[e]) > model.tol:
                edge['label'] = f"{flow[e]:.3g}"
            elif disp_zero:
                edge['label'] = "0"
    if disp_kernel:
        d, m = model.kernel.right.shape
        for e, edge in enumerate(edges_info):
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
            edge['label'] += label
            if not label:
                edge['color'] = 'black'
    if check_flow and disp_flow:
        out_rate = model.incidence_csr @ flow
        for i, node in enumerate(nodes_info):
            if np.abs(model.rates[i] - out_rate[i]) / model.rates[i] > check_tolerance:
                node['color'] = 'red'
        for e, edge in enumerate(edges_info):
            if flow[e] < -model.tol:
                edge['color'] = 'red'
            elif flow[e] < model.tol:
                edge['color'] = 'orange'
            else:
                edge['color'] = edge.get('color', 'blue')
    return nodes_info, edges_info


def show(model, bipartite=False, png=False, **kwargs):
    """
    End-to-end display solution for model.
    It is basically a pipe
    :class:`~stochastic_matching.model.Model` ->
    :meth:`~stochastic_matching.display.info_maker` ->
    :meth:`~stochastic_matching.display.vis_converter` ->
    :meth:`~stochastic_matching.display.vis_show`.

    The extra arguments are passed when needed on the right spot along the pipe, allowing maximal flexibility.

    Parameters
    ----------
    model: :class:`~stochastic_matching.model.Model`
        The model to display.
    bipartite: :class:`bool`, optional
        Tells if the bipartite node/edge structure should be explicitly shown.
    png: :class:`bool`
        Make a mirror PNG that can be saved.
    **kwargs
        Keyword arguments. See :meth:`~stochastic_matching.display.info_maker`,
        :meth:`~stochastic_matching.display.vis_converter`, and
        :meth:`~stochastic_matching.display.vis_show` for details.


    Returns
    -------
    :class:`~IPython.display.HTML`

    Examples
    --------

    >>> import stochastic_matching as sm
    >>> pyramid = sm.Pyramid()

    Basic display.

    >>> show(pyramid)
    <IPython.core.display.HTML object>

    With this, the nodes and edges will not show anything. The result can be exported as png.

    >>> nodes_info=[{'label': ''} for _ in range(pyramid.n)]
    >>> edges_info=[{'label': ''} for _ in range(pyramid.m)]
    >>> show(pyramid, nodes_info=nodes_info, edges_info=edges_info, png=True)
    <IPython.core.display.HTML object>

    Fan is a true hypergraph.

    >>> fan = sm.Fan()

    To display:

    >>> show(fan)
    <IPython.core.display.HTML object>

    To display in bipartite mode:

    >>> show(fan, bipartite=True)
    <IPython.core.display.HTML object>
    """
    info_kwds = {'disp_rates', 'disp_flow', 'flow', 'disp_kernel', 'disp_zero',
                 'check_flow', 'check_tolerance'}
    converter_kwds = {'nodes_info', 'edges_info'}
    vis_kwds = {'vis_options', 'template', 'vis', 'div_name'}

    info_kwargs = {k: v for k, v in kwargs.items() if k in info_kwds}
    converter_kwargs = {k: v for k, v in kwargs.items() if k in converter_kwds}
    vis_kwargs = {k: v for k, v in kwargs.items() if k in vis_kwds}

    nodes_info, edges_info = info_maker(model, **info_kwargs)

    if 'nodes_info' in converter_kwargs:
        nodes_info = [ {**n1, **n2} for n1, n2 in zip(nodes_info, converter_kwargs['nodes_info'])]
    if 'edges_info' in converter_kwargs:
        edges_info = [ {**e1, **e2} for e1, e2 in zip(edges_info, converter_kwargs['edges_info'])]

    vis_nodes, vis_edges = vis_converter(model, nodes_info, edges_info)

    if model.adjacency is None:
        vis_kwargs['vis_options'] = {**vis_kwargs.get('vis_options', dict()), **HYPER_GRAPH_VIS_OPTIONS.copy()}
        vis_options = vis_kwargs['vis_options']
        if bipartite:
            vis_options['groups']['HyperEdge']['fixed']['x'] = True
            vis_options['groups']['Node']['fixed']['x'] = True
            inner_width = round(.8 * vis_options.get('width', 600))
            for vis_node in vis_nodes:
                if vis_node.get('group') == 'HyperEdge':
                    vis_node['x'] = inner_width
        else:
            vis_options['groups']['HyperEdge']['fixed']['x'] = False
            vis_options['groups']['Node']['fixed']['x'] = False
    if png:
        vis_kwargs['template'] = PNG_TEMPLATE
    vis_show(vis_nodes, vis_edges, **vis_kwargs)

