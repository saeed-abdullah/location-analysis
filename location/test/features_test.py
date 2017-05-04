# -*- coding: utf-8 -*-
"""

    Unit testing location_features module

"""

import geohash
import numpy as np
import pandas as pd
import pytest
import location.features as lf
from geopy.distance import vincenty
import math


def test_gyration_radius():
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
    actual_value = lf.gyration_radius(df)
    assert actual_value == pytest.approx(expected, 0.01)

    # check when k is larger the number of different visited locations
    assert np.isnan(lf.gyration_radius(df, k=5))

    # add the last gps point for five more times
    add_df = pd.DataFrame()
    add_df['cluster'] = ['dr5rw5u'] * 5
    add_df['latitude'] = [lat[-1]] * 5
    add_df['longitude'] = [lon[-1]] * 5
    df = pd.concat([df, add_df])

    expected = 6927.0444113855365
    assert lf.gyration_radius(df) == pytest.approx(expected, 0.01)

    # test the k-th radius of gyration
    add_df = pd.DataFrame()
    add_df['cluster'] = ['dr5xg5g'] * 2
    coordinate = geohash.decode('dr5xg5g')
    add_df['latitude'] = coordinate[0]
    add_df['longitude'] = coordinate[1]
    df = pd.concat([df, add_df])
    assert lf.gyration_radius(df, k=2) == pytest.approx(expected, 0.01)


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


def test_max_dist_between_clusters():
    data = pd.DataFrame(columns=['latitude', 'longitude', 'cluster'])
    d = lf.max_dist_between_clusters(data)
    assert np.isnan(d)

    data = pd.DataFrame([[12.3, -45.6, 1],
                         [12.3, -45.6, 1]],
                        columns=['latitude', 'longitude', 'cluster'])
    d = lf.max_dist_between_clusters(data)
    assert d == pytest.approx(0, 0.000001)

    data = pd.DataFrame([[12.3, -45.6, 1],
                         [43.8, 72.9, 2],
                         [32.5, 12.9, 3]],
                        columns=['latitude', 'longitude', 'cluster'])
    d = lf.max_dist_between_clusters(data)
    assert d == pytest.approx(11233331.835309023, 0.00001)


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
    assert np.isnan(lf.entropy(df, time_c='time'))

    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs']
    df['time'] = [pd.to_datetime('2015-04-14 07:46:43')]
    assert np.isnan(lf.entropy(df, time_c='time'))

    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs']
    df['time'] = [pd.to_datetime('2015-04-14 07:46:43')]
    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs', 'dr5xejs', 'dr5xef2']
    df['time'] = [pd.to_datetime('2015-04-14 07:00:00'),
                  pd.to_datetime('2015-04-14 07:20:00'),
                  pd.to_datetime('2015-04-14 07:40:00')]
    ent = lf.entropy(df, time_c='time')
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
    ent = lf.entropy(df, time_c='time')
    assert ent == pytest.approx(1.0114042647073516, 0.00001)


def test_norm_entropy():
    df = pd.DataFrame(columns=['cluster', 'time'])
    assert np.isnan(lf.norm_entropy(df, time_c='time'))

    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs']
    df['time'] = [pd.to_datetime('2015-04-14 07:46:43')]
    assert np.isnan(lf.norm_entropy(df, time_c='time'))

    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs']
    df['time'] = [pd.to_datetime('2015-04-14 07:46:43')]
    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs', 'dr5xejs', 'dr5xef2']
    df['time'] = [pd.to_datetime('2015-04-14 07:00:00'),
                  pd.to_datetime('2015-04-14 07:20:00'),
                  pd.to_datetime('2015-04-14 07:40:00')]
    ent = lf.norm_entropy(df, time_c='time')
    assert ent == pytest.approx(0.56233514 / math.log(2), 0.00001)

    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs']
    df['time'] = [pd.to_datetime('2015-04-14 07:46:43')]
    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs', 'dr5xejs', 'dr5xef2', 'dr5xefq']
    df['time'] = [pd.to_datetime('2015-04-14 07:00:00'),
                  pd.to_datetime('2015-04-14 07:20:00'),
                  pd.to_datetime('2015-04-14 07:40:00'),
                  pd.to_datetime('2015-04-14 08:00:00')]
    ent = lf.norm_entropy(df, time_c='time')
    assert ent == pytest.approx(1.0114042 / math.log(3), 0.00001)


def test_loc_var():
    df = pd.DataFrame(columns=['latitude', 'longitude', 'cluster'])
    v = lf.loc_var(df)
    assert np.isnan(v)

    df['latitude'] = [12, 13, 14, 40]
    df['longitude'] = [24, 26, 29, 70]
    df['cluster'] = [np.nan, 1, 2, 1]
    v = lf.loc_var(df)
    assert v == pytest.approx(6.326348221044057, 0.00001)


def test_home_stay():
    df = pd.DataFrame(columns=['cluster', 'time'])
    hs = lf.home_stay(df, 'abc', time_c='time')
    assert np.isnan(hs)

    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs', np.nan]
    df['time'] = [pd.to_datetime('2015-04-14 07:46:43'),
                  pd.to_datetime('2015-04-14 07:56:43')]
    hs = lf.home_stay(df, 'abc', time_c='time')
    assert np.isnan(hs)

    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs']
    df['time'] = [pd.to_datetime('2015-04-14 07:46:43')]
    hs = lf.home_stay(df, 'dr5xejs', time_c='time')
    assert np.isnan(hs)

    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs', 'dr5xejs']
    df['time'] = [pd.to_datetime('2015-04-14 02:00:00'),
                  pd.to_datetime('2015-04-14 03:00:00')]
    hs = lf.home_stay(df, 'dr5xejs', time_c='time')
    assert hs == 3600

    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs']
    df['time'] = [pd.to_datetime('2015-04-14 07:46:43')]
    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs', 'dr5xejs', 'dr5xef2', 'dr5xefq']
    df['time'] = [pd.to_datetime('2015-04-14 07:00:00'),
                  pd.to_datetime('2015-04-14 07:20:00'),
                  pd.to_datetime('2015-04-14 07:40:00'),
                  pd.to_datetime('2015-04-14 08:00:00')]
    hs = lf.home_stay(df, 'dr5xef2', time_c='time')
    assert hs == 1200


def test_trans_time():
    df = pd.DataFrame(columns=['cluster', 'time'])
    tt = lf.trans_time(df, time_c='time')
    assert np.isnan(tt)

    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs', np.nan]
    df['time'] = [pd.to_datetime('2015-04-14 07:40:00'),
                  pd.to_datetime('2015-04-14 08:00:00')]
    tt = lf.trans_time(df, time_c='time')
    assert tt == 600

    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs', np.nan]
    df['time'] = [pd.to_datetime('2015-04-14 07:40:00'),
                  pd.to_datetime('2015-04-14 08:00:00')]
    df = df.set_index('time')
    tt = lf.trans_time(df)
    assert tt == 600


def test_total_dist():
    df = pd.DataFrame([(12.3, -45.6, 1),
                       (43.8, 72.9, 2),
                       (32.5, 12.9, 3)],
                      columns=['latitude', 'longitude', 'cluster'])
    assert lf.total_dist(df) == pytest.approx(16520745.44722021, 0.00001)

    df = pd.DataFrame([(12.3, -45.6, 1),
                       (43.8, 72.9, 2),
                       (32.5, 12.9, 3),
                       (50, 50, np.nan)],
                      columns=['latitude', 'longitude', 'cluster'])
    assert lf.total_dist(df) == pytest.approx(16520745.44722021, 0.00001)

    df = pd.DataFrame([(12.3, -45.6, 1),
                       (43.8, 72.9, 2),
                       (43.8, 72.9, 2),
                       (32.5, 12.9, 3),
                       (50, 50, np.nan)],
                      columns=['latitude', 'longitude', 'cluster'])
    assert lf.total_dist(df) == pytest.approx(16520745.44722021, 0.00001)

    cluster_map = {1: (12.3, -45.6),
                   2: (43.8, 72.9),
                   3: (32.5, 12.9)}
    df = pd.DataFrame([1, 2, 3],
                      columns=['cluster'])
    td = lf.total_dist(df, cluster_mapping=cluster_map)
    assert td == pytest.approx(16520745.44722021, 0.00001)
