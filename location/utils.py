# -*- coding: utf-8 -*-
"""
    utils
    ~~~~~

    Utility functions for motif analysis.
"""

import pandas as pd
import numpy as np
import math
from copy import deepcopy, copy
from collections import Counter
from geopy.distance import vincenty
import geohash

from location import motif


def compute_gyration(data,
                     sr_col='stay_region',
                     k=None):
    """
    Compute the total or k-th radius of gyration.
    This follows the work of Pappalardo et al.
    (see http://www.nature.com/articles/ncomms9166)

    Parameters:
    -----------

    data: DataFrame
        Location data.

    sr_col: str
        Column name for stay region.
        Default is 'stay_region'.

    k: int
        k-th radius of gyration.
        Default is None, in this case return total radius of gyration.
        k-th radius of gyration is the radius gyration compuated up to
        the k-th most frequent visited locations.


    Returns:
    --------
    (radius of gyration): float
        Radius of gyration in meters.
        Return np.nan is k is greater than the number of different
        visited locations.
    """
    loc_data = data.copy()
    loc_data = pd.DataFrame(loc_data[sr_col].dropna())

    # get location data for corresponding k
    if k is not None:
        cnt_locs = Counter(loc_data[sr_col])
        # number of different visited locations
        num_visited_locations = len(cnt_locs)
        if k > num_visited_locations:
            return np.nan
        else:
            # k most frequent visited locations
            k_locations = cnt_locs.most_common()[:k]
            k_locations = [x[0] for x in k_locations]
            # compute gyration for the k most frequent locations
            loc_data = loc_data.loc[loc_data[sr_col].isin(k_locations)]

    # compute coordinates for visited locations/stay regions
    locs_hist_coord = [geohash.decode(x) for x in loc_data[sr_col]]
    loc_data['latitude'] = [x[0] for x in locs_hist_coord]
    loc_data['longitude'] = [x[1] for x in locs_hist_coord]

    # compute mass of locations
    r_cm = motif.get_geo_center(loc_data, lat_c='latitude', lon_c='longitude')
    r_cm = (r_cm['latitude'], r_cm['longitude'])

    # compute gyration of radius
    temp_sum = 0
    for _, r in loc_data.iterrows():
        p = (r.latitude, r.longitude)
        d = vincenty(p, r_cm).m
        temp_sum += d ** 2

    return math.sqrt(temp_sum / len(loc_data))


