# -*- coding: utf-8 -*-
"""
    Module for location features.
"""

import numpy as np
import pandas as pd
from location import motif
from geopy.distance import vincenty
import math
from collections import Counter
import pytz
import datetime


def gyrationradius(data,
                   k=None,
                   lat='latitude',
                   lon='longitude',
                   cluster='cluster'):
    """
    Compute the total or k-th radius of gyration.
    The radius of gyration is used to characterize the typical
    distance travelled by an individual.

    This follows the work of Pappalardo et al.
    (see http://www.nature.com/articles/ncomms9166)

    Parameters:
    -----------
    data: DataFrame
        Location data.

    k: int
        k-th radius of gyration.
        Default is None, in this case return total radius of gyration.
        k-th radius of gyration is the radius gyration compuated up to
        the k-th most frequent visited locations.

    lat, lon, cluster: str
        Columns of latitude, longitude, and
        cluster ids. The default valuesa are
        'latitude', 'longitude', and 'cluster'
        respectively.

    Returns:
    --------
    float
        Radius of gyration in meters.
        Return np.nan is k is greater than the number of different
        visited locations.
    """
    loc_data = data[[lat, lon, cluster]].dropna()
    if len(loc_data) <= 0:
        return np.nan

    # get location data for corresponding k
    if k is not None:
        cnt_locs = Counter(loc_data[cluster])
        # number of different visited locations
        num_visited_locations = len(cnt_locs)
        if k > num_visited_locations:
            return np.nan
        else:
            # k most frequent visited locations
            k_locations = cnt_locs.most_common()[:k]
            k_locations = [x[0] for x in k_locations]
            # compute gyration for the k most frequent locations
            loc_data = loc_data.loc[loc_data[cluster].isin(k_locations)]

    # compute mass of locations
    r_cm = motif.get_geo_center(loc_data, lat_c=lat, lon_c=lon)
    r_cm = (r_cm['latitude'], r_cm['longitude'])

    # compute gyration of radius
    temp_sum = 0
    for _, r in loc_data.iterrows():
        p = (r[lat], r[lon])
        d = vincenty(p, r_cm).m
        temp_sum += d ** 2

    return math.sqrt(temp_sum / len(loc_data))


def test_num_trips():
    df = pd.DataFrame(columns=['cluster'])
    n = lf.num_trips(df)
    assert np.isnan(n)

    df = pd.DataFrame([[1],
                       [1],
                       [1]],
                      columns=['cluster'])
    n = lf.num_trips(df)
    assert n == 0

    df = pd.DataFrame([[1],
                       [np.nan],
                       [2]],
                      columns=['cluster'])
    n = lf.num_trips(df)
    assert n == 1

    df = pd.DataFrame([[1],
                       [1],
                       [np.nan],
                       [2],
                       [1],
                       [np.nan]],
                      columns=['cluster'])
    n = lf.num_trips(df)
    assert n == 2


def test_max_dist():
    data = pd.DataFrame(columns=['latitude', 'longitude', 'cluster'])
    d = lf.max_dist(data)
    assert np.isnan(d)

    data = pd.DataFrame([[12.3, -45.6, 1],
                         [12.3, -45.6, 1]],
                        columns=['latitude', 'longitude', 'cluster'])
    d = lf.max_dist(data)
    assert d == pytest.approx(0, 0.000001)

    data = pd.DataFrame([[12.3, -45.6, 1],
                         [43.8, 72.9, 2],
                         [32.5, 12.9, 3]],
                        columns=['latitude', 'longitude', 'cluster'])
    d = lf.max_dist(data)
    assert d == pytest.approx(11233331.835309023, 0.00001)


def num_clusters(data, cluster_col='cluster'):
    """
    Compute the number of location clusters, which is
    the number of different places visited.

    Parameters:
    -----------
    data: DataFrame
        Location data.

    cluster_col: str
        Location cluster id.

    Returns:
    --------
    n: int
        Number of clusters.
    """
    data = data[[cluster_col]].dropna()
    if len(data) == 0:
        return 0
    else:
        n = len(np.unique(data))
        return n


def displacement(data,
                 lat='latitude',
                 lon='longitude',
                 cluster='cluster',
                 cluster_mapping=None):
    """
    Calculate the displacement of the location data,
    which is list of distances traveled from one location
    to another.

    Parameters:
    -----------
    data: dataframe
        Location data.

    cluster: str
        Column cluster ids.
        Default value is cluster.

    lat, lon: str
        Columns of latitude, and longitude.
        Default values are 'latitude', and
        'longitude' respectively.

    cluster_mapping: dict
        A mapping from cluster id to
        gps coordinates.
        Defaults to None, in which case
        use latitude and longitude given
        in location data.

    Returns:
    --------
    displace: list
        List of displacements in meters.
    """
    data = data.loc[~pd.isnull(data[cluster])]
    displace = []

    if len(data) <= 1:
        return displace

    data = data.reset_index()

    if cluster_mapping is None:
        prev_idx = 0
        prev_cluster = data.ix[0, cluster]
        loc_list = []

        # get location history
        for i in range(1, len(data)):
            curr_cluster = data.ix[i, cluster]

            # compute the coordinates of the center
            # of current location cluster
            if curr_cluster != prev_cluster:
                tmp_df = data.loc[(data.index >= prev_idx) &
                                  (data.index <= i - 1)]
                coord = motif.get_geo_center(df=tmp_df,
                                             lat_c=lat,
                                             lon_c=lon)
                loc_list.append((coord['latitude'],
                                 coord['longitude']))
                prev_idx = i
                prev_cluster = curr_cluster

        # handle last location
        tmp_df = data.ix[(data.index >= prev_idx) &
                         (data.index <= len(data) - 1)]
        coord = motif.get_geo_center(df=tmp_df,
                                     lat_c=lat,
                                     lon_c=lon)
        loc_list.append((coord['latitude'],
                         coord['longitude']))

        # compute displacements
        if len(loc_list) <= 1:
            return displace

        # compute the distance between different
        # consecutive locations
        for i in range(1, len(loc_list)):
            displace.append(vincenty(loc_list[i-1],
                                     loc_list[i]).m)

    # use cluster mapping instead of computing
    # the coordinates of cluster center
    else:
        prev_cluster = data.ix[0, cluster]
        loc_list = []

        # get location history
        for i in range(1, len(data)):
            curr_cluster = data.ix[i, cluster]
            if curr_cluster != prev_cluster:
                loc_list.append(cluster_mapping[prev_cluster])
                prev_cluster = curr_cluster

        # handle last location
        loc_list.append((cluster_mapping[prev_cluster]))

        # compute displacements
        if len(loc_list) <= 1:
            return displace

        for i in range(1, len(loc_list)):
            displace.append(vincenty(loc_list[i-1],
                                     loc_list[i]).m)

    return displace
