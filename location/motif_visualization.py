# -*- coding: utf-8 -*-
"""
    Module for motif visualization.
"""

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from matplotlib import patches


def _get_connected_edge(pt, ep, curr_edges):
    """
    Find edges that are connected to the node.

    Parameters:
    -----------
    pt: int
        Point index.

    ep: dict
        Edges to end points dictionary, returned by _get_edge_matrix.
        Defaults to None, in which case the dictionary
        is created using _get_edge_matrix().

    curr_edges: list
        Current edges.

    Returns:
    --------
    edges: list
        Connected edges.
    """
    edges = []
    for e in curr_edges:
        p1, p2 = ep[e]
        if pt == p1 or pt == p2:
            edges.append(e)
    return edges


def _node_on_edge_matrix(pos, em=None, ep=None, is_debug=False):
    """
    Create an matrix storing whether a node is on an edge.

    Parameters:
    -----------
    pos: list
        Positions for nodes.
        Returned by _generate_node_pos

    em: numpy.matrix
        Edge matrix, returned by _get_edge_matrix.
        Defaults to None, in which case the matrix
        is created using _get_edge_matrix().

    ep: dict
        Edges to end points dictionary, returned by _get_edge_matrix.
        Defaults to None, in which case the dictionary
        is created using _get_edge_matrix().

    Returns:
    --------
    nem: numpy.matrix
        Node on edge matrix.
    """
    n = len(pos)
    ne = int(n * (n - 1) / 2)
    if em is None:
        em, ep = _get_edge_matrix(n)
    nem = np.matrix(np.zeros(shape=(n, ne), dtype=np.float16))
    for i in range(n):
        for j in range(ne):
            curr_ep = ep[j]
            p1 = pos[curr_ep[0]]
            p2 = pos[curr_ep[1]]
            p = pos[i]
            k, b = _get_line_para(p1, p2)
            is_on_line = _is_on_line(p, k, b,
                                     ep1=p1,
                                     ep2=p2,
                                     inc_endpoint=False,
                                     is_debug=False)
            if is_on_line:
                nem[i, j] = 1
    return nem


def _adjacency_matrix(g, is_debug=False):
    """
    Create an adjacency matrix for given
    nodes and edges.

    Parameters:
    -----------
    g: networkx.DiGraph
        A directed graph.

    Returns:
    --------
    am: numpy.matrix
        Adjacency matrix.
    """
    nodes = g.nodes()
    edges = g.edges()
    n = len(nodes)
    am = np.matrix(np.zeros(shape=(n, n), dtype=np.int16))
    for e in edges:
        i, j = e
        am[i, j] = 1
    if is_debug:
        print('nodes:')
        print(nodes)
        print('edges:')
        print(edges)
    return am


def _assign_node_pos(am, pos, is_debug=False):
    """
    Assign nodes to the best positions
    so that the plotted graph has no or least
    number of crossing edges.

    Parameters:
    -----------
    m: numpy.matrix
        Adjacency matrix.

    pos: list
        Coordinates of node positions.

    Returns:
    --------
    node_pos: list
        Node positions. Each element is a
        coordinate corresponding to a node
        in the underlying adjacency list.
    """
    n = len(am)
    node_id = list(range(n))

    # edge matrix, edge-to-node matrix
    em, ep = _get_edge_matrix(n)
    # edge crossing matrix
    ecm = _edge_crossing_matrix(pos=pos, em=em, ep=ep)
    # edge angle matrix
    eam = _edge_angle_matrix(pos=pos,
                             em=em, ep=ep,
                             use_degrees=True)
    # node on edge matrix
    nem = _node_on_edge_matrix(pos=pos, em=em, ep=ep)
    # search for assginment
    r = _search_pos(am=am,
                    em=em,
                    ep=ep,
                    ecm=ecm,
                    eam=eam,
                    pos=pos,
                    nem=nem,
                    curr_cross_cnt=0,
                    curr_angle_cnt=0,
                    curr_asgmt=[],
                    curr_edges=[],
                    rem_nodes=node_id,
                    max_cross_cnt=len(ecm) ** 2,
                    max_angle_cnt=len(ecm) ** 2,
                    best_asgmt=node_id,
                    angle_th=45,
                    is_debug=False)
    node_pos, goodness_cnt, search_info = r
    if is_debug:
        print('\nGoodness of crossing edges: {}'.format(goodness_cnt))
        print('Search info: '.format(search_info))
    return node_pos


def _search_pos(am, em, ep, ecm, eam, pos, nem,
                curr_cross_cnt,
                curr_angle_cnt,
                curr_asgmt,
                curr_edges,
                rem_nodes,
                max_cross_cnt,
                max_angle_cnt,
                best_asgmt,
                angle_th=50,
                is_debug=False,
                run_info=[]):
    """
    Use depth-first-search to generate all
    possible node positions and corresponding
    number of crossing edges.

    Parameters:
    -----------
    am: numpy.matrix
        Adjacency matrix.

    em: numpy.matrix
        Edge matrix returned by _get_edge_matrix().

    ep: dict
        Edges to end points dictionary, returned by _get_edge_matrix.
        Defaults to None, in which case the dictionary
        is created using _get_edge_matrix().

    ecm: numpy.matrix
        Edge crossing matrix returned by _edge_crossing_matrix().

    eam: numpy.matrix
        Edge angle matrix returned by _edge_angle_matrix().

    pos: list
        List of positions for the nodes.

    nem: numpy.matrix
        Node on edge matrix.

    curr_cross_cnt: int
        Current cross edge counts.

    angle_th: float
        The threshold angle (in degrees) between
        two edges.
        Defaults to 50 degree.

    curr_angle_cnt: int
        Number of angles that larger than the threshold.

    curr_asgmt: list
        Current node assignment.

    curr_edges: list
        Current edges.

    rem_nodes: list
        Remaining unassigned nodes.

    max_cross_cnt: int
        Current largest cross edge count for any assignment.

    curr_angle_cnt: int
        The best number of angles that larger
        than the threshold.

    best_asgmt: list
        Best position assigments.

    Returns:
    --------
    best_asgmt: list
        Best position assigments.

    max_cnt: tuple
        Goodness of the assignments.
        (crossing edge count, small angle count)

    run_info: list
        Additional information abou the search.
    """
    if is_debug:
        print()
        print('current assignment: {}'.format(curr_asgmt))
        print('remaining nodes: {}'.format(rem_nodes))
        print('current crossing number: {}'.format(curr_cross_cnt))
        print('max crossing count: {}'.format(max_cross_cnt))

    # base case
    if len(rem_nodes) == 0:
        if curr_cross_cnt <= max_cross_cnt:
            if curr_cross_cnt < max_cross_cnt:
                max_cross_cnt = curr_cross_cnt
                max_angle_cnt = curr_angle_cnt
                best_asgmt = list(curr_asgmt)
            elif curr_angle_cnt < max_angle_cnt:
                    max_cross_cnt = curr_cross_cnt
                    max_angle_cnt = curr_angle_cnt
                    best_asgmt = list(curr_asgmt)
        return list(best_asgmt), (max_cross_cnt, max_angle_cnt), run_info

    rem_nodes1 = []
    rem_nodes2 = []
    if len(curr_asgmt) == 0:
        rem_nodes2 = list(rem_nodes)
    else:
        last_assign_node = curr_asgmt[-1]
        for rem_n in rem_nodes:
            if (am[last_assign_node, rem_n] == 1) or \
               (am[rem_n, last_assign_node] == 1):
                rem_nodes1.append(rem_n)
            else:
                rem_nodes2.append(rem_n)
    rem_nodes1 = list(reversed(rem_nodes1))
    rem_nodes2 = list(reversed(rem_nodes2))

#     for next_node in rem_nodes:
    for _ in rem_nodes:
        if len(rem_nodes1) > 0:
            next_node = rem_nodes1.pop()
        else:
            next_node = rem_nodes2.pop()
        tmp_new_edges = []
        cnt_added = 0
        angle_cnt_added = 0
        exceed_max = False

        # count number of new crossing edges
        for exg_node_idx in range(len(curr_asgmt)):
            connected_edges = _get_connected_edge(pt=len(curr_asgmt),
                                                  ep=ep,
                                                  curr_edges=curr_edges)
            connected_edges2 = _get_connected_edge(pt=exg_node_idx,
                                                   ep=ep,
                                                   curr_edges=curr_edges)
            connected_edges.extend(connected_edges2)
            # check if current two nodes are connected
            exg_node = curr_asgmt[exg_node_idx]

            if (am[exg_node, next_node] == 1) or \
               (am[next_node, exg_node] == 1):
                exg_new_edge = True
            else:
                exg_new_edge = False
            if exg_new_edge:
                new_edge = em[exg_node_idx, len(curr_asgmt)]
                # check crossing edge
                for exg_edge in curr_edges:
                    if ecm[exg_edge, new_edge] == 1:
                        cnt_added += 1
                    if (exg_edge in connected_edges) and \
                       (eam[exg_edge, new_edge] < angle_th):
                        angle_cnt_added += 1

                    tmp_cross_cnt = curr_cross_cnt + cnt_added
                    tmp_angle_cnt = curr_angle_cnt + angle_cnt_added

                    if (tmp_cross_cnt >= max_cross_cnt):
                        if tmp_cross_cnt > max_cross_cnt:
                            exceed_max = True
                            break
                        elif tmp_angle_cnt >= max_angle_cnt:
                            exceed_max = True
                            break

#                 check if any nodes is on the edge
                if not exceed_max:
                    for node_idx2 in range(len(am)):
                        if nem[node_idx2, new_edge] == 1:
                            exceed_max = True
                            break

                if exceed_max:
                    break
                else:
                    tmp_new_edges.append(new_edge)
                    curr_edges.append(new_edge)
        if exceed_max:
            for e in tmp_new_edges:
                curr_edges.remove(e)
            continue
        else:
            curr_asgmt.append(next_node)
            rem_nodes_copy = list(rem_nodes)
            rem_nodes_copy.remove(next_node)
            r = _search_pos(am,
                            em,
                            ep,
                            ecm,
                            eam,
                            pos,
                            nem,
                            curr_cross_cnt + cnt_added,
                            curr_angle_cnt + angle_cnt_added,
                            list(curr_asgmt),
                            list(curr_edges),
                            rem_nodes_copy,
                            max_cross_cnt,
                            max_angle_cnt,
                            best_asgmt,
                            angle_th=angle_th,
                            is_debug=is_debug)
            best_asgmt, max_cnt, a_info = r
            max_cross_cnt, max_angle_cnt = max_cnt
            if (max_cross_cnt == 0) and (max_angle_cnt == 0):
                return list(best_asgmt), (max_cross_cnt, max_angle_cnt), []
            curr_asgmt.pop()
            for e in tmp_new_edges:
                curr_edges.remove(e)
    return list(best_asgmt), (max_cross_cnt, max_angle_cnt), run_info


def _compute_angle(e1, e2, ret_cos=True):
    """
    Compute the angle of the two lines.
    The angle is in [0, 90] degrees.

    Parameters:
    -----------
    e1, e2: list
        Coordinates of the two lines.

    ret_cos: bool
        Whether return the cosine
        of the angle.
        Defaults to True.

    Returns:
    --------
    a: float
        Angle between the two lines.
        Either the cosine value or radians.
    """
    # get the vector of the two line
    xy1, xy2 = e1
    x1, y1 = (xy2[0] - xy1[0], xy2[1] - xy1[1])
    xy1, xy2 = e2
    x2, y2 = (xy2[0] - xy1[0], xy2[1] - xy1[1])
    cos_v = (x1 * x2 + y1 * y2) / \
            (math.sqrt(x1 ** 2 + y1 ** 2) *
             math.sqrt(x2 ** 2 + y2 ** 2))
    if ret_cos:
        a = abs(cos_v)
    else:
        cos_v = abs(cos_v)
        if cos_v > 1:
            cos_v = 1
        a = math.acos(abs(cos_v))
    return a


def _edge_angle_matrix(pos, em=None, ep=None,
                       is_debug=False,
                       use_degrees=False):
    """
    Create an matrix storing the angle between two edges.

    Parameters:
    -----------
    pos: list
        Positions for nodes.
        Returned by _generate_node_pos

    em: numpy.matrix
        Edge matrix, returned by _get_edge_matrix.
        Defaults to None, in which case the matrix
        is created using _get_edge_matrix().

    em: dict
        Endpoint matrix, returned by _get_edge_matrix.
        Defaults to None, in which case the dictionary
        is created using _get_edge_matrix().

    use_degrees: bool
        Whether convert angle into degrees.
        Defaults to False.

    Returns:
    --------
    eam: numpy.matrix
        Edge crossing matrix.
    """
    n = len(pos)
    if em is None:
        em, ep = _get_edge_matrix(n)
    ne = int(n * (n - 1) / 2)  # number of edges
    eam = np.matrix(np.zeros(shape=(ne, ne), dtype=np.float16))
    for i in range(ne):
        for j in range(ne):
            if i == j:
                eam[i, j] = 0
            elif i < j:
                n1, n2 = ep[i]
                e1 = (pos[n1], pos[n2])
                n3, n4 = ep[j]
                e2 = (pos[n3], pos[n4])
                eam[i, j] = _compute_angle(e1, e2, ret_cos=False)
                if use_degrees:
                    eam[i, j] = np.degrees(eam[i, j])
                if is_debug and is_crossing:
                    print(e1)
                    print(e2)
                    print(ema[i, j])
                    print()
            else:
                eam[i, j] = eam[j, i]
    return eam


def _edge_crossing_matrix(pos, em=None, ep=None, is_debug=False):
    """
    Create an matrix storing whether two
    edges are crossing.

    Parameters:
    -----------
    pos: list
        Positions for nodes.
        Returned by _generate_node_pos

    em: numpy.matrix
        Edge matrix, returned by _get_edge_matrix.
        Defaults to None, in which case the matrix
        is created using _get_edge_matrix().

    em: dict
        Endpoint matrix, returned by _get_edge_matrix.
        Defaults to None, in which case the dictionary
        is created using _get_edge_matrix().

    Returns:
    --------
    ecm: numpy.matrix
        Edge crossing matrix.
    """
    n = len(pos)
    if em is None:
        em, ep = _get_edge_matrix(n)
    ne = int(n * (n - 1) / 2)  # number of edges
    ecm = np.matrix(np.zeros(shape=(ne, ne), dtype=np.int16))
    for i in range(ne):
        for j in range(ne):
            if i == j:
                ecm[i, j] == -1
            elif i < j:
                n1, n2 = ep[i]
                e1 = (pos[n1], pos[n2])
                n3, n4 = ep[j]
                e2 = (pos[n3], pos[n4])
                is_crossing = _is_crossing(e1, e2, inc_endpoint=False)
                if is_debug and is_crossing:
                    print(e1)
                    print(e2)
                    print(is_crossing)
                    print()
                if is_crossing:
                    ecm[i, j] = 1
            else:
                ecm[i, j] = ecm[j, i]
    return ecm


def _get_edge_matrix(n):
    """
    Create an edge matrix.
    Use two node id to determine an edge id.

    Parameters:
    -----------
    n: int
        Number of nodes/vertices.

    Returns:
    em: numpy.matrix
        Edge matrix.

    ep: dict
        Endpoint matrix.
        {edge_id: (node1, node2)}
    """
    em = np.matrix(np.zeros(shape=(n, n), dtype=np.int16))
    ep = {}
    c = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                em[i, j] = -1
                continue

            if i < j:
                em[i, j] = c
                c += 1
            else:
                em[i, j] = em[j, i]
            if em[i, j] not in ep:
                ep[em[i, j]] = (i, j)
    return em, ep


def _rearrange_matrix(a):
    """
    Rearrange adjacency matrix according to
    edge degrees.

    Parameters:
    -----------
    a: numpy.matrix
        Adacency matrix.

    Returns:
    --------
    m: numpy.matrix
        Adacency matrix.
    """
    n_nodes = len(a)
    if n_nodes == 1:
        return a
    for i in range(n_nodes):
        for j in range(n_nodes):
            a[i, j] = int(a[i, j])
    degrees = _calculate_degree(a)
    node_id = list(range(n_nodes))
    sorted_list = sorted(node_id, key=lambda x: degrees[x], reverse=True)
    m = np.matrix(np.zeros(shape=(n_nodes, n_nodes), dtype=np.int16))
    for i in range(n_nodes):
        for j in range(n_nodes):
            m[i, j] = a[sorted_list[i], sorted_list[j]]
    return m


def _calculate_degree(m):
    """
    Calculate the degree of each edge.

    Parameters:
    -----------
    m: numpy.matrix
        Adjacency matrix.

    Returns:
    --------
    d: list
        Degree of each edge.
    """
    d = []
    l = len(m)
    for i in range(l):
        connected_nodes = (m[i, :] | m[:, i].T).sum()
        out_degree = m[i, :].sum()
        in_degree = m[:, i].sum()
        d.append(100 * connected_nodes + 10 * out_degree + in_degree)
    return d


def _generate_node_pos(n, w=1, r=0.1, center=(0, 0), dist=5):
    """
    Generate node positions.

    Parameters:
    -----------
    g: int
        Number of nodes.

    w: float
        Width of the drawing.
        Default value is 1.

    r: float
        Ratio of radius of the node to the width
        of the drawing.
        Defaults to None, in which case
        the value is set to 1/10 of the width.

    center: list of float
        Coordinate of the center of the drawing.
        Default value is (0, 0).

    dist: float
        Distance between two nodes in terms of
        one tenth of the width.
        Default is five to six units.

    Returns:
    --------
    pos: list
        Nodes position coordinates.

    h: float
        Height of the drawing.
    """
    pos = []
    r *= w
    bs = w / 10  # block size
    ni = dist / 2  # interval between nodes
    dist = dist / 10 * w
    ox, oy = center
    if n == 1:
        pos.append(center)
        h = w
    elif n == 2:
        pos.append((ox, oy + 3 * bs))
        pos.append((ox, oy - 3 * bs))
        h = dist + 2 * r
    elif n == 3:
        h = dist + 2 * r
        pos.append((ox - ni * bs, oy - ni * bs))
        pos.append((ox, oy + ni * bs))
        pos.append((ox + ni * bs, oy - ni * bs))
    elif n == 4:
        h = dist + 2 * r
        pos.append((ox - ni * bs, oy + ni * bs))
        pos.append((ox + ni * bs, oy + ni * bs))
        pos.append((ox + ni * bs, oy - ni * bs))
        pos.append((ox - ni * bs, oy - ni * bs))
    elif (n % 2) == 1:
        n_oneside = int((n - 1) / 2)
        if (n_oneside % 2) == 0:
            top_y = oy + dist / 2 + ((n_oneside) / 2 - 1) * dist

            # add top left node
            pos.append((ox - dist / 2, top_y))

            # add top node
            pos.append((ox,
                        top_y + dist / 2 * math.sqrt(3)))

            # add right nodes
            for i in range(n_oneside):
                pos.append((ox + dist / 2,
                            top_y - i * dist))

            # add left nodes
            for i in range(n_oneside - 1):
                pos.append((ox - dist / 2,
                            top_y - (n_oneside - 1) * dist + i * dist))
            h = dist * (n_oneside - 1) + 2 * r + dist / 2 * math.sqrt(3)
        else:
            n_on_one_side = int(n_oneside)
            top_y = oy + (n_on_one_side - 1) / 2 * dist

            # add top left node
            pos.append((ox - dist / 2, top_y))

            # add top node
            pos.append((ox,
                        top_y + dist / 2 * math.sqrt(3)))

            # add right nodes
            for i in range(n_on_one_side):
                pos.append((ox + dist / 2,
                            top_y - i * dist))

            # add left nodes
            for i in range(n_oneside - 1):
                pos.append((ox - dist / 2,
                            top_y - (n_on_one_side - 1) * dist + i * dist))
            h = dist * (n_oneside - 1) + 2 * r + dist / 2 * math.sqrt(3)
    else:
        n_oneside = int(n / 2)
        if ((n_oneside - 1) % 2) == 0:
            top_y = oy + dist / 2 + ((n_oneside - 1) / 2 - 1) * dist

            # add top left node
            pos.append((ox - dist / 2, top_y))

            # add top node
            pos.append((ox,
                        top_y + dist / 2 * math.sqrt(3)))

            # add right nodes
            for i in range(n_oneside - 1):
                pos.append((ox + dist / 2,
                            top_y - i * dist))

            # add bottom node
            pos.append((ox,
                        top_y - (n_oneside - 2) * dist -
                        dist / 2 * math.sqrt(3)))

            # add left nodes
            for i in range(n_oneside - 2):
                pos.append((ox - dist / 2,
                            top_y - (n_oneside - 2) * dist + i * dist))
            h = dist * (n_oneside - 1) + 2 * (r + dist / 2 * math.sqrt(3))
        else:
            n_on_one_side = int(n_oneside - 1)
            top_y = oy + (n_on_one_side - 1) / 2 * dist

            # add top left node
            pos.append((ox - dist / 2, top_y))

            # add top node
            pos.append((ox,
                        top_y + dist / 2 * math.sqrt(3)))

            # add right nodes
            for i in range(n_on_one_side):
                pos.append((ox + dist / 2,
                            top_y - i * dist))

            # add bottom node
            pos.append((ox,
                        top_y - (n_on_one_side - 1) * dist -
                        dist / 2 * math.sqrt(3)))

            # add left nodes
            for i in range(n_oneside - 2):
                pos.append((ox - dist / 2,
                            top_y - (n_on_one_side - 1) * dist + i * dist))
            h = dist * (n_oneside - 1) + 2 * (r + dist / 2 * math.sqrt(3))
    return pos, h


def _is_crossing(e1, e2, inc_endpoint=True, is_debug=False):
    """
    Determine if two edges are crossing.
    Determine if line segments are crossing.

    Parameters:
    -----------
    e1, e2: list or tuple
        List of end points for edges.
        e = ((xy1, xy2), (xy3, xy4))

    inc_endpoint: bool
        Whether consider endpoint.
        Defaults to True.

    is_debug: bool
        Whether in debug mode.
        If true, display debug info.
        Defualts to False.

    Returns:
    --------
    crossing: bool
        Whether the two edge are crossing.
    """
    p1, p2 = e1
    p3, p4 = e2
    k1, b1 = _get_line_para(p1, p2)
    k2, b2 = _get_line_para(p3, p4)
    if is_debug:
        print('line1:')
        print('   {} --- {}'.format(p1, p2))
        print('   k = {}, b = {}'.format(k1, b1))
        print()
        print('line2:')
        print('   {} --- {}'.format(p3, p4))
        print('   k = {}, b = {}'.format(k2, b2))
    # both are parallel to each other
    if k1 == k2:
        if is_debug:
            print()
            print('Parallel lines.')
        crossing = False
    else:
        ix, iy = _get_intxn(k1, b1, k2, b2)
        if is_debug:
            print()
            print('Intersection: {}'.format((ix, iy)))

        intxn_on_line1 = _is_on_line((ix, iy), k1, b1,
                                     ep1=p1, ep2=p2,
                                     inc_endpoint=inc_endpoint)
        intxn_on_line2 = _is_on_line((ix, iy), k2, b2,
                                     ep1=p3, ep2=p4,
                                     inc_endpoint=inc_endpoint)
        crossing = intxn_on_line1 and intxn_on_line2
        if is_debug:
            print('Intersection on line1: {}'.format(intxn_on_line1))
            print('Intersection on line2: {}'.format(intxn_on_line2))
            print()
    return crossing


def _is_on_line(p, k, b, ep1=None, ep2=None,
                inc_endpoint=True,
                is_debug=False):
    """
    Check if a point is on the line,
    or a segment.

    p: list
        Coordinates of the point to be checked.

    k, b: float
        Slope and intercept of a line.
        Returned by _get_line_para().

    p1, p2: list
        Coordinates of two end points
        of the line.
        Defualts to None, in which case the
        function checks whether the point is
        on the line.

    inc_endpoint: bool
        Whether include endpoint of the
    Returns:
    --------
    online: bool
        Whether the point is on the line/segment.
    """
    x, y = p
    if is_debug:
        print()
        print('Point: ({}, {})'.format(x, y))
        print('Line: k = {}, b = {}'.format(k, b))
        print('Include endpoint: {}'.format(inc_endpoint))
    if ep1 is None:
        if is_debug:
            print('Test for line.')
        if k is None:
            online = (b == x)
        else:
            online = (k * x + b) == y
    else:
        if is_debug:
            print('Test for segment.')
            print('End points: {}, {}'.format(ep1, ep2))
        if k is None:
            uy = max(ep1[1], ep2[1])
            ly = min(ep1[1], ep2[1])
            if inc_endpoint:
                online = (x == b) and \
                         (y <= uy) and \
                         (y >= ly)
            else:
                online = (x == b) and \
                         (y < uy) and \
                         (y > ly)
            if is_debug:
                print('Line parellel to y axis.')
        else:
            on_line = abs(x * k + b - y) < 0.0001
            if is_debug:
                print('Slope is not zero.')
                print('Point is on line: {}'.format(on_line))
            if k == 0:
                lx = min(ep1[0], ep2[0])
                rx = max(ep1[0], ep2[0])
                if inc_endpoint:
                    online = on_line and \
                             (x >= lx) and \
                             (x <= rx)
                else:
                    online = on_line and \
                             (x > lx) and \
                             (x < rx)
                if is_debug:
                    print('Line parellel to x axis.')
            else:
                lx = min(ep1[0], ep2[0])
                rx = max(ep1[0], ep2[0])
                uy = max(ep1[1], ep2[1])
                ly = min(ep1[1], ep2[1])
                if inc_endpoint:
                    online = on_line and \
                             (x >= lx) and (x <= rx) and \
                             (y <= uy) and (y >= ly)
                else:
                    online = on_line and \
                             (x > lx) and (x < rx) and \
                             (y < uy) and (y > ly)
        return online


def _get_intxn(k1, b1, k2, b2):
    """
    Get the intersection of two line.
    Two lines are assumed to be not
    parallel to each other.

    Parameters:
    -----------
    k1, k2: float
        Slopes of the two lines.

    b1, b2: float
        Intercepts of the two lines.

    Returns:
    --------
    (x, y): tuple of floats
        Coordinates of the intersection.
    """
    # if either of the two lines
    # are parallel to y axis
    if k1 is None:
        x = b1
        y = k2 * x + b2
    elif k2 is None:
        x = b2
        y = k1 * x + b1
    else:
        x = (b2 - b1) / (k1 - k2)
        y = k1 * x + b1
    return (x, y)


def _get_line_para(xy1, xy2):
    """
    Calculate the slope and intercept of a line
    defined by two points.

    Parameters:
    -----------
    xy1, xy2: list or tuple
        Coordinates of two points.

    Returns:
    --------
    k: float
        Slope. Return None if the line is
        parallel to y axis, when slope does
        not exist.

    b: float
        Intercept when k exists, or x value
        when k does not exist.
    """
    x1, y1 = xy1
    x2, y2 = xy2

    # if k doesn't exist
    if x2 - x1 == 0:
        k = None
        b = x1
        return k, b

    # calculate slope
    k = (y2 - y1) / (x2 - x1)
    b = y1 - k * x1
    return k, b


def _patch_edge(xy1, xy2, dir1, dir2, radius):
    """
    Create an arrow patch representing an edge
    from node position xy1 to node position xy2.

    Parameters:
    -----------
    xy1, xy2: tuple
        Coordinates of the two nodes that
        the edge connects.

    dir1, dir2: int
        Whether there is an edge for either
        of the two directions between the two
        nodes. 0 - exist, 1 - non-exist.
        dir1 for edge from xy1 to xy2,
        dir2 for edge from xy2 to xy1.

    radius: float
        Radius of the nodes.

    Returns:
    --------
    edge: matplotlib.patches.FancyArrow
        Edge patch.
        Return an empty list if there is no
        edge between the two nodes.
    """
    # return None if there is no edge
    if dir1 == 0 and dir2 == 0:
        return []

    l = math.sqrt((xy1[0] - xy2[0]) ** 2 +
                  (xy1[1] - xy2[1]) ** 2)
    sin_alpha = (xy2[1] - xy1[1]) / l
    cos_alpha = (xy2[0] - xy1[0]) / l
    xoffset = radius * 1.4 * cos_alpha
    yoffset = radius * 1.4 * sin_alpha
    edge = []

    head_length = 1 / 4
    line_width = radius * 4
    head_width = radius / 1.1
    dx = xy2[0] - xy1[0] - xoffset * 2
    dy = xy2[1] - xy1[1] - yoffset * 2
    full_edge_length = math.sqrt(dx ** 2 + dy ** 2)
    head_length_width_ratio = 1.4
    if head_length * full_edge_length / head_width > head_length_width_ratio:
        head_length = head_width * head_length_width_ratio / full_edge_length
    # two edge
    if dir1 == 1 and dir2 == 1:
        # alpha_radians = math.asin(sin_alpha)
        alpha_radians = _get_angle(xy1, xy2)
        beta_radians = alpha_radians + math.pi / 2
        sin_beta = math.sin(beta_radians)
        cos_beta = math.cos(beta_radians)
        xoffset2 = radius * 0.35 * cos_beta
        yoffset2 = radius * 0.35 * sin_beta

        # create arrow patches
        dx1 = xy2[0] - xy1[0] - xoffset * 2
        dy1 = xy2[1] - xy1[1] - yoffset * 2
        e1 = plt.arrow(x=xy1[0] + xoffset + xoffset2,
                       y=xy1[1] + yoffset + yoffset2,
                       dx=dx1 * (1 - head_length),
                       dy=dy1 * (1 - head_length),
                       linewidth=line_width,
                       fc="k",
                       ec="k",
                       head_width=head_width,
                       head_length=full_edge_length *
                       head_length)

        dx2 = xy1[0] - xy2[0] + xoffset * 2
        dy2 = xy1[1] - xy2[1] + yoffset * 2
        e2 = plt.arrow(x=xy2[0] - xoffset - xoffset2,
                       y=xy2[1] - yoffset - yoffset2,
                       dx=dx2 * (1 - head_length),
                       dy=dy2 * (1 - head_length),
                       linewidth=line_width,
                       fc="k",
                       ec="k",
                       head_width=head_width,
                       head_length=full_edge_length *
                       head_length)
        edge = [e1, e2]
    elif dir1 == 1:
        dx1 = xy2[0] - xy1[0] - xoffset * 2
        dy1 = xy2[1] - xy1[1] - yoffset * 2
        e1 = plt.arrow(x=xy1[0] + xoffset,
                       y=xy1[1] + yoffset,
                       dx=dx1 * (1 - head_length),
                       dy=dy1 * (1 - head_length),
                       linewidth=line_width,
                       fc="k",
                       ec="k",
                       head_width=head_width,
                       head_length=full_edge_length *
                       head_length)
        edge = [e1]
    else:
        dx2 = xy1[0] - xy2[0] + xoffset * 2
        dy2 = xy1[1] - xy2[1] + yoffset * 2
        e2 = plt.arrow(x=xy2[0] - xoffset,
                       y=xy2[1] - yoffset,
                       dx=dx2 * (1 - head_length),
                       dy=dy2 * (1 - head_length),
                       linewidth=line_width,
                       fc="k",
                       ec="k",
                       head_width=head_width,
                       head_length=full_edge_length *
                       head_length)
        edge = [e2]
    return edge


def _get_angle(xy1, xy2):
    """
    Get the angle for the two nodes.

    Parameters:
    -----------
    xy1, xy2: list
        XY corrdinates.

    Returns:
    --------
    a: float
        Angle in radian.
    """
    l = math.sqrt((xy1[0] - xy2[0]) ** 2 +
                  (xy1[1] - xy2[1]) ** 2)
    sin_alpha = (xy2[1] - xy1[1]) / l
    if xy2[0] - xy1[0] > 0:
        a = sin_alpha
    else:
        a = math.pi - sin_alpha
    return a


def _patch_node(xy, radius,
                is_center=False,
                color='k',
                color_center='r'):
    """
    Create a circle pacth representing a node.

    Parameters:
    -----------
    xy: tuple
        Coordinates(x, y) of the center of node.

    radius: float
        Radius.

    is_center: bool
        Whether this node is a center node.
        A center node is a node where
        a daily motif starts and ends.

    color: matplotlib.colors
        Node color.
        Default color is black.

    color_center: matplotlib.colors
        Color for a center node.
        Default color is red.

    Returns:
    --------
    node: matplotlib.patches.Circle
        Node patch.
    """
    if is_center:
        c = color_center
    else:
        c = color
    node = plt.Circle(xy=xy,
                      radius=radius,
                      fc=c,
                      ec='none')
    return node


def _cmp_motif(m1, m2):
    """
    Determine the order of two motifs.

    Parameters:
    -----------
    m1, m2: list
        Motifs.

    Returns:
    --------
    int
        0 means equal.
        1 means m1 ranks before m2.
        -1 means m1 ranks after m2.
    """
    f1 = len(m1['data'])
    f2 = len(m2['data'])
    n1 = len(m1['graph'].nodes())
    n2 = len(m2['graph'].nodes())
    if n1 < n2:
        return 1
    elif n1 > n2:
        return -1
    else:
        if f1 > f2:
            return 1
        elif f1 < f2:
            return -1
        else:
            return 0


def _sort_motifs(motifs):
    """
    Sort motifs. Insertion sort.

    Parameters:
    -----------
    motifs: list
        Motifs.

    Returns:
    --------
    sorted_motifs: list
        Sorted motifs.
    """
    sorted_motifs = []
    for m in motifs:
        l = len(sorted_motifs)
        i = 0
        while (i < l) and (_cmp_motif(m, sorted_motifs[i]) < 1):
            i += 1
        sorted_motifs.insert(i, m)
    return sorted_motifs


def draw_motif(g,
               w=1,
               r=0.1,
               center=(0, 0),
               center_node=None,
               ax=None,
               is_debug=False):
    """
    Draw a motif in an aesthetically legible way.

    Parameters:
    -----------
    g: networkx.DiGraph
        A directed graph representing the motif.

    w: float
        Width of the plot.

    r: float
        Radius of the node in motifs.

    center_node: str
        Center node.

    ax: matplotlib.axes
        The axes on which the motif is drawn.

    Returns:
    --------
    ax: matplotlib.axes
        The axes on which the motif is drawn.

    h: float
        The height of the plot.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    # number of nodes
    n_node = len(g.nodes())
    # node positions, height of plot
    pos, h = _generate_node_pos(n_node, w=w, r=r, center=center)
    # adjacency matrix
    # m = _adjacency_matrix(g)
    am = nx.adjacency_matrix(g).todense()
    m = _rearrange_matrix(a=am)
    # assign node positions to nodes
    asgmt = _assign_node_pos(m, pos, is_debug=is_debug)
    if is_debug:
        print('final asgmt: {}'.format(asgmt))
    # draw motifs
    cpnt = []
    # nodes
    for p in pos:
        cpnt.append(_patch_node(xy=p, radius=w * r))
    # add node id
    if is_debug:
        c = 0
        for p in pos:
            ax.text(p[0], p[1], str(c),
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=20, color='w')
            c += 1

    # edges
    for i in range(n_node - 1):
        for j in range(i, n_node):
            cpnt.extend(_patch_edge(pos[i],
                                    pos[j],
                                    m[asgmt[i], asgmt[j]],
                                    m[asgmt[j], asgmt[i]],
                                    w * r))
    # edge id
    if is_debug:
        em, ep = _get_edge_matrix(len(am))
        for i in range(n_node - 1):
            for j in range(i, n_node):
                if (m[asgmt[i], asgmt[j]] == 1) or \
                   (m[asgmt[j], asgmt[i]] == 1):
                    d = em[i, j]
                    x = (pos[i][0] + pos[j][0]) / 2
                    y = (pos[i][1] + pos[j][1]) / 2
                    ax.text(x, y, str(d),
                            horizontalalignment='center',
                            verticalalignment='center',
                            fontsize=20, color='k')
    for p in cpnt:
        ax.add_patch(p)
    return ax, h


def plot_motifs(motifs,
                ax=None,
                y_scale=1,
                cmap=None,
                motif_id=None,
                txt=None):
    """
    Draw motifs.

    Parameters:
    -----------
    motifs: list of dictionaries
        Motifs.

    y_scale: float
        Scale for y axies (0, 1].
        Defaults to be 1.

    cmap: dictionary
        Colormap for different motifs.

    txt: str
        Text to display under the plot.
        Defaults to None.

    Returns:
    --------
    ax: matplotlib.axes
        Axis the motifs are drawn on.
    """
    n_motifs = len(motifs)  # number of motifs
    if n_motifs == 0:
        print(' No motif')
        return ax

    motifs = _sort_motifs(motifs)

    # calculate freqency
    freq = [len(d['data']) for d in motifs]
    if ax is None:
        # fig = plt.figure(figsize=(5 * 0.2 * n_motifs, 5))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    ax.set_xlim([0, n_motifs])
    ax.set_ylim([0, 1])

    # draw freqcency bar chart
    # calculate freq percentage
    sc = 1 / y_scale
    sum_freq = sum(freq)
    pct = [f / sum_freq for f in freq]
    scaled_pct = [p * sc for p in pct]
    w = 0.2 * 0.5  # bar width
    o = (0.2 - 0.1) / 2  # bar offset
    c = '#59A9C2'
    for i in range(n_motifs):
        p = patches.Rectangle((i * 0.2 + o, 0),  # lower left corner
                              w,                 # width
                              scaled_pct[i],     # height
                              color=c)           # color
        ax.add_patch(p)

    # plot motif
    w = 0.2
    x = 0.5
    y = 0.9
    max_h = 0
    for i in range(n_motifs):
        ax, h = draw_motif(ax=ax,
                           g=motifs[i]['graph'],
                           w=w,
                           center=(0.2 / 2 + i * 0.2, 0.9))
        if h > max_h:
            max_h = h

    # plot lines to seperate different number of nodes
    for i in range(n_motifs-1):
        if len(motifs[i]['graph'].nodes()) != \
           len(motifs[i+1]['graph'].nodes()):
            ax.plot((i * 0.2 + 0.2, i * 0.2 + 0.2),
                    (0, 1 + max_h * 0.5),
                    color='k',
                    linewidth=0.5,
                    linestyle='--')

    # configure figure
    xticks = np.arange(0.1, (n_motifs + 1) * 0.2, 0.2)
    xticklabels = np.arange(1, n_motifs + 1, 1)
    ax.set_xticks(xticks)
    if motif_id is not None:
        xticklabels = []
        for m in motifs:
            for i in motif_id:
                if nx.is_isomorphic(motif_id[i], m['graph']):
                    l = i
                    break
            xticklabels.append(l)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel('Motif ID')
    ax.set_xlim([-0.5, (n_motifs + 1) * 0.2 + 0.1])
    ax.set_ylim([0, 1 + max_h * 0.5 + 0.1])
    yticks = np.arange(0, 1.1, 0.1)
    yticklabels = ['{}%'.format(int(x * 100)) for x in yticks]
    yticks = [x * sc for x in yticks if x * sc <= 1]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels[0:len(yticks)])
    ax.set_ylabel('Motif frequency')
    fig.set_size_inches(5 * 0.2 * n_motifs + 0.7, 5 * (1 + max_h * 0.5) + 0.1)
    plt.axis('equal')
    ax.grid(False)
    ax.yaxis.grid(True)
    return ax
