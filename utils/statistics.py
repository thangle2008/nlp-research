from __future__ import absolute_import, print_function, division


def map_range(values, target_min, target_max):
    cur_min = min(values)
    cur_max = max(values)

    def map_value(v):
        # map to (0, 1)
        tmp = (v - cur_min) / (cur_max - cur_min)
        # map to target range
        return target_min + tmp * (target_max - target_min)

    return [map_value(v) for v in values]
