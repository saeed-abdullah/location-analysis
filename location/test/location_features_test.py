# -*- coding: utf-8 -*-
"""

    Unit testing location_features module

"""

import geohash
import numpy as np
import pandas as pd
import pytest
import location.location_features as lf


def test_gyrationradius():
    # test data
    data_stay_region = ['dr5xfdt',
                        'dr5xfdt',
                        'dr5xfdt',
                        'dr5rw5u',
                        'dr5rw5u',
                        'dr5rw5u',
                        'dr5rw5u',
                        'dr5rw5u',
                        'dr5rw5u',
                        'dr5rw5u']
    coordinates = [geohash.decode(x) for x in data_stay_region]
    lat = [c[0] for c in coordinates]
    lon = [c[1] for c in coordinates]
    df = pd.DataFrame()
    df['cluster'] = data_stay_region
    df['latitude'] = lat
    df['longitude'] = lon

    # expected result
    expected = 7935.926632803189

    # tolerance = 0.01 meter
    actual_value = lf.gyrationradius(df)
    assert actual_value == pytest.approx(expected, 0.01)

    # check when k is larger the number of different visited locations
    assert np.isnan(lf.gyrationradius(df, k=5))

    # add the last gps point for five more times
    add_df = pd.DataFrame()
    add_df['cluster'] = ['dr5rw5u'] * 5
    add_df['latitude'] = [lat[-1]] * 5
    add_df['longitude'] = [lon[-1]] * 5
    df = pd.concat([df, add_df])

    expected = 6927.0444113855365
    assert lf.gyrationradius(df) == pytest.approx(expected, 0.01)

    # test the k-th radius of gyration
    add_df = pd.DataFrame()
    add_df['cluster'] = ['dr5xg5g'] * 2
    coordinate = geohash.decode('dr5xg5g')
    add_df['latitude'] = coordinate[0]
    add_df['longitude'] = coordinate[1]
    df = pd.concat([df, add_df])
    assert lf.gyrationradius(df, k=2) == pytest.approx(expected, 0.01)


def num_trips(data,
              cluster_col='cluster'):
    """
    Compute the number of trips from one
    location to another.

    Parameters:
    -----------
    data: DataFrame
        location data.

    cluster_col: str
        Location cluster column.
        Default value is 'cluster'.

    Returns:
    --------
    n_trip: int
        Number of trips.
    """
    data = data.loc[~pd.isnull(data[cluster_col])]

    if len(data) == 0:
        return np.nan

    data = data.reset_index()

    # previous location
    p = data.ix[0, cluster_col]
    n_trip = 0

    for i in range(1, len(data)):

        # current location
        c = data.ix[i, cluster_col]
        if p == c:
            continue
        else:
            n_trip += 1
            p = c

    return n_trip


def max_dist(data,
             cluster_col='cluster',
             lat_col='latitude',
             lon_col='longitude',
             cluster_mapping=None):
    """
    Compute the maximum distance between two locations.

    Parameters:
    -----------
    data: DataFrame
        Location data.

    cluster_col: str
        Location cluster id column.

    lat_col, lon_col: str
        Latidue and longitude of the cluster
        locations.

    cluster_mapping: dict
        A dictionary storing the coordinates
        of location cluster locations.
        {location_cluster: coordinates}

    Returns:
    --------
    max_dist: float
        Maximum distance between two locations in meters.
    """
    data = data.loc[~pd.isnull(data[cluster_col])]

    if len(data) == 0:
        return np.nan

    locations = np.unique(data[cluster_col])
    if len(locations) == 1:
        return 0

    locations_coord = []
    # calculate coordinates
    if cluster_mapping is None:

        # compute the coordinates of each
        # of the location clusters
        for l in locations:
            df = data.loc[data[cluster_col] == l]
            coord = motif.get_geo_center(df=df,
                                         lat_c=lat_col,
                                         lon_c=lon_col)
            coord = (coord['latitude'], coord['longitude'])
            locations_coord.append(coord)
    else:
        # use the coordinates provided by the mapping
        for l in locations:
            locations_coord.append(cluster_mapping[l])

    # find maximum distance
    max_dist = 0
    for i in range(len(locations) - 1):
        for j in range(i + 1, len(locations)):
            d = vincenty(locations_coord[i], locations_coord[j]).m
            if d > max_dist:
                max_dist = d

    return max_dist


def test_num_clusters():
    df = pd.DataFrame([np.nan, np.nan],
                      columns=['cluster'])
    n = lf.num_clusters(df)
    assert n == 0

    df = pd.DataFrame([np.nan, 1, 1, 2, 4, 4, 9, 9, 4],
                      columns=['cluster'])
    n = lf.num_clusters(df)
    assert n == 4
