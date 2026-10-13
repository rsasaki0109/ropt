"""Scalar incident-edge central differences for large pose graphs.

This remains numerical differentiation of the original stored-coordinate
objective. Only edges incident to a perturbed pose can change; evaluating them
avoids O(poses * edges) work and cancellation against unrelated constant costs.
"""
from pose_graph import objective


def gradient_inf_incident(poses, edges, anchors=(0,)):
    incident = {key: [] for key in poses}
    for edge in edges:
        incident[edge[0]].append(edge)
        if edge[1] != edge[0]:
            incident[edge[1]].append(edge)
    maximum, h = 0.0, 1e-6
    for key, value in poses.items():
        if key in anchors:
            continue
        selected = incident[key]
        local = {endpoint: poses[endpoint] for edge in selected for endpoint in edge[:2]}
        for coordinate in range(3):
            plus, minus = list(value), list(value)
            plus[coordinate] += h
            minus[coordinate] -= h
            local[key] = tuple(plus)
            positive = objective(local, selected)
            local[key] = tuple(minus)
            negative = objective(local, selected)
            maximum = max(maximum, abs((positive - negative) / (2 * h)))
    return maximum
