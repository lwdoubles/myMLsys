import numpy as np
from .Node import ones_like

def find_topo_sort(node_list):
    visited = set()
    topo_order = []
    for node in node_list:
        depth_first_search(node, visited, topo_order)
    return topo_order


def depth_first_search(node, visited, topo_order):
    if node in visited:
        return
    visited.add(node)
    for n in node.inputs:
        depth_first_search(n, visited, topo_order)
    topo_order.append(node)


def sum_node_list(node_list):
    from operator import add
    from functools import reduce
    return reduce(add, node_list)


# 反向求导
def gradients(output_node, node_list):
    node_to_output_grads_list = {}
    node_to_output_grads_list[output_node] = [ones_like(output_node)]
    # a map from node to the gradient of that node
    node_to_output_grad = {}
    reverse_topo_order = reversed(find_topo_sort([output_node]))
    for node in reverse_topo_order:
        output_grad = sum_node_list(node_to_output_grads_list[node])
        node_to_output_grad[node] = output_grad

        input_grads_list = node.op.gradient(node, output_grad)
        for i in range(len(node.inputs)):
            if node.inputs[i] not in node_to_output_grads_list:
                node_to_output_grads_list[node.inputs[i]] = []
            node_to_output_grads_list[node.inputs[i]].append(input_grads_list[i])

    grad_node_list = [node_to_output_grad[node] for node in node_list]
    return grad_node_list
