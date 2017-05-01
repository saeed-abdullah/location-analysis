# -*- coding: utf-8 -*-
"""

    Unit testing location_features module

"""

import geohash
import numpy as np
import pandas as pd
import pytest
import location.location_features as lf
from geopy.distance import vincenty


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


def test_displacement():
    df = pd.DataFrame(columns=['cluster'])
    assert len(lf.displacement(df)) == 0

    df['cluster'] = [np.nan, np.nan, 'dr5xejs', 'dr5xejs']
    df['latitude'] = [40.749562, 40.749563, 40.724195, 40.724248]
    df['longitude'] = [-73.710272, -73.706997, -73.690573, -73.690747]
    assert len(lf.displacement(df)) == 0

    df = pd.DataFrame(columns=['cluster'])
    df['cluster'] = [np.nan, np.nan,
                     'dr5xejs', 'dr5xejs',
                     np.nan, 'dr5xef2',
                     'dr5xef2']
    df['latitude'] = [40.749562, 40.749563,
                      40.724269, 40.724269,
                      40.724269, 40.706522,
                      40.706522]
    df['longitude'] = [-73.710272, -73.706997,
                       -73.690737, -73.690737,
                       -73.690737, -73.662139,
                       -73.662139]
    displace = lf.displacement(df)
    assert len(displace) == 1
    assert displace[0] == pytest.approx(3118.1779973248804, 0.0001)

    df = pd.DataFrame(columns=['cluster'])
    df['cluster'] = [np.nan, np.nan,
                     'dr5xejs', 'dr5xejs',
                     np.nan, 'dr5xef2',
                     'dr5xef2', 'dr5xejs']
    df['latitude'] = [40.749562, 40.749563,
                      40.724269, 40.724269,
                      40.724269, 40.706522,
                      40.706522, 40.724048]
    df['longitude'] = [-73.710272, -73.706997,
                       -73.690737, -73.690737,
                       -73.690737, -73.662139,
                       -73.662139, -73.690753]
    displace = lf.displacement(df)
    assert len(displace) == 2
    assert displace[0] == pytest.approx(3118.1779973248804, 0.0001)
    assert displace[1] == pytest.approx(3103.7813441942367, 0.0001)

    df = pd.DataFrame(columns=['cluster'])
    df['cluster'] = [np.nan, np.nan,
                     'dr5xejs', 'dr5xejs',
                     np.nan, 'dr5xef2',
                     'dr5xef2', 'dr5xejs']
    cluster_mapping = {'dr5xejs': geohash.decode('dr5xejs'),
                       'dr5xef2': geohash.decode('dr5xef2')}
    displace = lf.displacement(df, cluster_mapping=cluster_mapping)
    assert len(displace) == 2
    expected = vincenty(cluster_mapping['dr5xejs'],
                        cluster_mapping['dr5xef2']).m
    assert displace[0] == pytest.approx(expected, 0.0001)
    assert displace[1] == pytest.approx(expected, 0.0001)


def test_wait_time():
    df = pd.DataFrame(columns=['cluster', 'time'])
    wt, cwt = lf.wait_time(df, time_c='time')
    assert len(wt) == 0
    assert len(cwt) == 0

    df['cluster'] = ['dr5xejs']
    df['time'] = [pd.to_datetime('2015-04-14 07:46:43')]
    wt, cwt = lf.wait_time(df, time_c='time')
    assert len(wt) == 0

    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs', 'dr5xejs']
    df['time'] = [pd.to_datetime('2015-04-14 07:00:00'),
                  pd.to_datetime('2015-04-14 07:20:00')]
    wt, cwt = lf.wait_time(df, time_c='time')
    assert len(wt) == 1
    assert wt[0] == 1200
    assert len(cwt) == 1
    assert 'dr5xejs' in cwt
    assert cwt['dr5xejs'] == 1200

    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs', 'dr5xejs', 'dr5xef2']
    df['time'] = [pd.to_datetime('2015-04-14 07:00:00'),
                  pd.to_datetime('2015-04-14 07:20:00'),
                  pd.to_datetime('2015-04-14 07:40:00')]
    wt, cwt = lf.wait_time(df, time_c='time')
    assert len(wt) == 2
    assert wt[0] == 1800, 0.00001
    assert wt[1] == 600, 0.00001
    assert len(cwt) == 2
    assert ('dr5xejs' in cwt) and ('dr5xef2' in cwt)
    assert cwt['dr5xejs'] == 1800
    assert cwt['dr5xef2'] == 600

    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs', 'dr5xejs', np.nan, 'dr5xef2']
    df['time'] = [pd.to_datetime('2015-04-14 07:00:00'),
                  pd.to_datetime('2015-04-14 07:20:00'),
                  pd.to_datetime('2015-04-14 07:40:00'),
                  pd.to_datetime('2015-04-14 08:00:00')]
    wt, cwt = lf.wait_time(df, time_c='time')
    assert len(wt) == 2
    assert wt[0] == 1800
    assert wt[1] == 600
    assert len(cwt) == 2
    assert ('dr5xejs' in cwt) and ('dr5xef2' in cwt)
    assert cwt['dr5xejs'] == 1800
    assert cwt['dr5xef2'] == 600

    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs', 'dr5xejs', 'dr5xef2', 'dr5xef2']
    df['time'] = [pd.to_datetime('2015-04-14 07:00:00'),
                  pd.to_datetime('2015-04-14 07:20:00'),
                  pd.to_datetime('2015-04-14 07:40:00'),
                  pd.to_datetime('2015-04-14 08:00:00')]
    wt, cwt = lf.wait_time(df, time_c='time')
    assert len(wt) == 2
    assert wt[0] == 1800
    assert wt[1] == 1800
    assert len(cwt) == 2
    assert ('dr5xejs' in cwt) and ('dr5xef2' in cwt)
    assert cwt['dr5xejs'] == 1800
    assert cwt['dr5xef2'] == 1800

    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = [np.nan, 'dr5xejs',
                     'dr5xejs', 'dr5xef2',
                     'dr5xef2', np.nan]
    df['time'] = [pd.to_datetime('2015-04-14 06:52:00'),
                  pd.to_datetime('2015-04-14 07:00:00'),
                  pd.to_datetime('2015-04-14 07:20:00'),
                  pd.to_datetime('2015-04-14 07:40:00'),
                  pd.to_datetime('2015-04-14 08:00:00'),
                  pd.to_datetime('2015-04-14 08:10:00')]
    wt, cwt = lf.wait_time(df, time_c='time')
    assert len(wt) == 2
    assert wt[0] == 2040
    assert wt[1] == 2100
    assert len(cwt) == 2
    assert ('dr5xejs' in cwt) and ('dr5xef2' in cwt)
    assert cwt['dr5xejs'] == 2040
    assert cwt['dr5xef2'] == 2100

    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = [np.nan, 'dr5xejs',
                     'dr5xef2', 'dr5xejs',
                     'dr5xef2', np.nan]
    df['time'] = [pd.to_datetime('2015-04-14 06:52:00'),
                  pd.to_datetime('2015-04-14 07:00:00'),
                  pd.to_datetime('2015-04-14 07:20:00'),
                  pd.to_datetime('2015-04-14 07:40:00'),
                  pd.to_datetime('2015-04-14 08:00:00'),
                  pd.to_datetime('2015-04-14 08:10:00')]
    wt, cwt = lf.wait_time(df, time_c='time')
    assert len(wt) == 4
    assert wt[0] == 840
    assert wt[1] == 1200
    assert wt[2] == 1200
    assert wt[3] == 900
    assert len(cwt) == 2
    assert ('dr5xejs' in cwt) and ('dr5xef2' in cwt)
    assert cwt['dr5xejs'] == 2040
    assert cwt['dr5xef2'] == 2100

    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = [np.nan, 'dr5xejs',
                     'dr5xef2', 'dr5xejs',
                     'dr5xef2', np.nan]
    df['time'] = [pd.to_datetime('2015-04-14 06:52:00'),
                  pd.to_datetime('2015-04-14 07:00:00'),
                  pd.to_datetime('2015-04-14 07:20:00'),
                  pd.to_datetime('2015-04-14 07:40:00'),
                  pd.to_datetime('2015-04-14 08:00:00'),
                  pd.to_datetime('2015-04-14 08:10:00')]
    df = df.set_index('time')
    wt, cwt = lf.wait_time(df)
    assert len(wt) == 4
    assert wt[0] == 840
    assert wt[1] == 1200
    assert wt[2] == 1200
    assert wt[3] == 900
    assert len(cwt) == 2
    assert ('dr5xejs' in cwt) and ('dr5xef2' in cwt)
    assert cwt['dr5xejs'] == 2040
    assert cwt['dr5xef2'] == 2100


def test_entropy():
    df = pd.DataFrame(columns=['cluster', 'time'])
    assert np.isnan(lf.entropy(df, time_col='time'))

    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs']
    df['time'] = [pd.to_datetime('2015-04-14 07:46:43')]
    assert np.isnan(lf.entropy(df, time_col='time'))

    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs']
    df['time'] = [pd.to_datetime('2015-04-14 07:46:43')]
    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs', 'dr5xejs', 'dr5xef2']
    df['time'] = [pd.to_datetime('2015-04-14 07:00:00'),
                  pd.to_datetime('2015-04-14 07:20:00'),
                  pd.to_datetime('2015-04-14 07:40:00')]
    ent = lf.entropy(df, time_col='time')
    assert ent == pytest.approx(0.5623351446188083, 0.00001)

    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs']
    df['time'] = [pd.to_datetime('2015-04-14 07:46:43')]
    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs', 'dr5xejs', 'dr5xef2', 'dr5xefq']
    df['time'] = [pd.to_datetime('2015-04-14 07:00:00'),
                  pd.to_datetime('2015-04-14 07:20:00'),
                  pd.to_datetime('2015-04-14 07:40:00'),
                  pd.to_datetime('2015-04-14 08:00:00')]
    ent = lf.entropy(df, time_col='time')
    assert ent == pytest.approx(1.0114042647073516, 0.00001)
