"""Reference camera roles selected by visible-Gaussian coverage."""
import math
import numpy as np


def split_reference_cameras(names, common, event_views):
    """Select a representative 5:1 split while retaining every edit event.

    ``common[k,l]`` counts Gaussians visible in both views. ``event_views``
    contains the available camera indices for each object/edit event.
    """
    counts = np.diag(common)
    distance = 1 - common / (counts[:, None] + counts[None, :] - common)
    np.fill_diagonal(distance, 0)
    size = math.ceil(len(names) / 6)

    def leaves_events(proposed):
        return all(set(views) - set(proposed) for views in event_views.values())

    selected = []
    best_distance = np.ones(len(names))
    for _ in range(size):
        choices = [j for j in range(len(names)) if j not in selected
                   and leaves_events(selected + [j])]
        if not choices:
            raise ValueError("Reference views cannot satisfy the event-availability constraint.")
        j = min(choices, key=lambda j: (np.minimum(best_distance, distance[:, j]).sum(), names[j]))
        selected.append(j)
        best_distance = np.minimum(best_distance, distance[:, j])
    while True:
        best = (float(distance[:, selected].min(1).sum()), None)
        for position in range(size):
            kept = selected[:position] + selected[position + 1:]
            base_distance = distance[:, kept].min(1) if kept else np.ones(len(names))
            for j in range(len(names)):
                if j in selected or not leaves_events(kept + [j]):
                    continue
                loss = float(np.minimum(base_distance, distance[:, j]).sum())
                if loss < best[0] - 1e-12:
                    best = (loss, (position, j))
        if best[1] is None:
            break
        position, j = best[1]
        selected[position] = j
    return ([name for index, name in enumerate(names) if index not in selected],
            [names[index] for index in sorted(selected)])
