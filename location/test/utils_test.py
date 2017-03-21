# -*- coding: utf-8 -*-
"""
    location.test.utils_test
    ~~~~~~~~~~~~~~~~~~~~~~~~

    Unit testing motif module

    :copyright: (c) 2016 by Saeed Abdullah.

"""

import numpy as np
import pandas as pd
import geohash
from geopy.distance import vincenty
import math

from location import utils
import pytest
from pytest import approx


def test_compute_gyration():
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
    df = pd.DataFrame()
    df['stay_region'] = data_stay_region

    # expected result
    expected = 7935.926632803189

    # tolerance = 0.01 meter
    assert utils.compute_gyration(df) == pytest.approx(expected, 0.01)

    # check when k is larger the number of different visited locations
    assert np.isnan(utils.compute_gyration(df, k=5))

    # add the last gps point for five more times
    add_df = pd.DataFrame()
    add_df['stay_region'] = ['dr5rw5u'] * 5
    df = pd.concat([df, add_df.copy()])

    expected = 6927.0444113855365
    assert utils.compute_gyration(df) == pytest.approx(expected, 0.01)

    # test the k-th radius of gyration
    add_df = pd.DataFrame()
    add_df['stay_region'] = ['dr5xg5g'] * 2
    df = pd.concat([df, add_df.copy()])
    assert utils.compute_gyration(df, k=2) == pytest.approx(expected, 0.01)


def test_compute_regularity():
    df = pd.DataFrame()
    timestamp = pd.Timestamp('2016-12-5 00:30:00')
    df['time'] = pd.date_range(timestamp, periods=2, freq='7d')
    df.loc[2, 'time'] = pd.Timestamp('2016-12-6 1:30:00')
    df['stay_region'] = ['dr5rw5u', 'dr5xg5g', 'dr5xg5g']
    df = df.set_index('time')

    reg = utils.compute_regularity(df)

    reg_computed1 = reg.loc[(reg.index.get_level_values('weekday') == 0) &
                            (reg.index.get_level_values('hour') == 0),
                            'regularity']

    assert reg_computed1.iloc[0] == pytest.approx(0.5)

    reg_computed2 = reg.loc[(reg.index.get_level_values('weekday') == 1) &
                            (reg.index.get_level_values('hour') == 1),
                            'regularity']

    assert reg_computed2.iloc[0] == pytest.approx(1)


def test_displacement():
    df = pd.DataFrame(columns=['cluster'])
    assert len(utils.displacement(df)) == 0

    df['cluster'] = [np.nan, np.nan, 'dr5xejs', 'dr5xejs']
    df['latitude'] = [40.749562, 40.749563, 40.724195, 40.724248]
    df['longitude'] = [-73.710272, -73.706997, -73.690573, -73.690747]
    assert len(utils.displacement(df)) == 0

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
    displace = utils.displacement(df)
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
    displace = utils.displacement(df)
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
    displace = utils.displacement(df, cluster_mapping=cluster_mapping)
    assert len(displace) == 2
    expected = vincenty(cluster_mapping['dr5xejs'],
                        cluster_mapping['dr5xef2']).m
    assert displace[0] == pytest.approx(expected, 0.0001)
    assert displace[1] == pytest.approx(expected, 0.0001)


def test_wait_time():
    df = pd.DataFrame(columns=['cluster', 'time'])
    wt, cwt = utils.wait_time(df, time_c='time')
    assert len(wt) == 0
    assert len(cwt) == 0

    df['cluster'] = ['dr5xejs']
    df['time'] = [pd.to_datetime('2015-04-14 07:46:43')]
    wt, cwt = utils.wait_time(df, time_c='time')
    assert len(wt) == 0

    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs', 'dr5xejs']
    df['time'] = [pd.to_datetime('2015-04-14 07:00:00'),
                  pd.to_datetime('2015-04-14 07:20:00')]
    wt, cwt = utils.wait_time(df, time_c='time')
    assert len(wt) == 1
    assert wt[0] == pytest.approx(20, 0.00001)
    assert len(cwt) == 1
    assert 'dr5xejs' in cwt
    assert cwt['dr5xejs'] == pytest.approx(20, 0.00001)

    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs', 'dr5xejs', 'dr5xef2']
    df['time'] = [pd.to_datetime('2015-04-14 07:00:00'),
                  pd.to_datetime('2015-04-14 07:20:00'),
                  pd.to_datetime('2015-04-14 07:40:00')]
    wt, cwt = utils.wait_time(df, time_c='time')
    assert len(wt) == 2
    assert wt[0] == pytest.approx(30, 0.00001)
    assert wt[1] == pytest.approx(10, 0.00001)
    assert len(cwt) == 2
    assert ('dr5xejs' in cwt) and ('dr5xef2' in cwt)
    assert cwt['dr5xejs'] == pytest.approx(30, 0.00001)
    assert cwt['dr5xef2'] == pytest.approx(10, 0.00001)

    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs', 'dr5xejs', np.nan, 'dr5xef2']
    df['time'] = [pd.to_datetime('2015-04-14 07:00:00'),
                  pd.to_datetime('2015-04-14 07:20:00'),
                  pd.to_datetime('2015-04-14 07:40:00'),
                  pd.to_datetime('2015-04-14 08:00:00')]
    wt, cwt = utils.wait_time(df, time_c='time')
    assert len(wt) == 2
    assert wt[0] == pytest.approx(30, 0.00001)
    assert wt[1] == pytest.approx(10, 0.00001)
    assert len(cwt) == 2
    assert ('dr5xejs' in cwt) and ('dr5xef2' in cwt)
    assert cwt['dr5xejs'] == pytest.approx(30, 0.00001)
    assert cwt['dr5xef2'] == pytest.approx(10, 0.00001)

    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs', 'dr5xejs', 'dr5xef2', 'dr5xef2']
    df['time'] = [pd.to_datetime('2015-04-14 07:00:00'),
                  pd.to_datetime('2015-04-14 07:20:00'),
                  pd.to_datetime('2015-04-14 07:40:00'),
                  pd.to_datetime('2015-04-14 08:00:00')]
    wt, cwt = utils.wait_time(df, time_c='time')
    assert len(wt) == 2
    assert wt[0] == pytest.approx(30, 0.00001)
    assert wt[1] == pytest.approx(30, 0.00001)
    assert len(cwt) == 2
    assert ('dr5xejs' in cwt) and ('dr5xef2' in cwt)
    assert cwt['dr5xejs'] == pytest.approx(30, 0.00001)
    assert cwt['dr5xef2'] == pytest.approx(30, 0.00001)

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
    wt, cwt = utils.wait_time(df, time_c='time')
    assert len(wt) == 2
    assert wt[0] == pytest.approx(34, 0.00001)
    assert wt[1] == pytest.approx(35, 0.00001)
    assert len(cwt) == 2
    assert ('dr5xejs' in cwt) and ('dr5xef2' in cwt)
    assert cwt['dr5xejs'] == pytest.approx(34, 0.00001)
    assert cwt['dr5xef2'] == pytest.approx(35, 0.00001)

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
    wt, cwt = utils.wait_time(df, time_c='time')
    assert len(wt) == 4
    assert wt[0] == pytest.approx(14, 0.00001)
    assert wt[1] == pytest.approx(20, 0.00001)
    assert wt[2] == pytest.approx(20, 0.00001)
    assert wt[3] == pytest.approx(15, 0.00001)
    assert len(cwt) == 2
    assert ('dr5xejs' in cwt) and ('dr5xef2' in cwt)
    assert cwt['dr5xejs'] == pytest.approx(34, 0.00001)
    assert cwt['dr5xef2'] == pytest.approx(35, 0.00001)

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
    wt, cwt = utils.wait_time(df)
    assert len(wt) == 4
    assert wt[0] == pytest.approx(14, 0.00001)
    assert wt[1] == pytest.approx(20, 0.00001)
    assert wt[2] == pytest.approx(20, 0.00001)
    assert wt[3] == pytest.approx(15, 0.00001)
    assert len(cwt) == 2
    assert ('dr5xejs' in cwt) and ('dr5xef2' in cwt)
    assert cwt['dr5xejs'] == pytest.approx(34, 0.00001)
    assert cwt['dr5xef2'] == pytest.approx(35, 0.00001)


def test_entropy():
    df = pd.DataFrame(columns=['cluster', 'time'])
    assert np.isnan(utils.entropy(df, time_col='time'))

    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs']
    df['time'] = [pd.to_datetime('2015-04-14 07:46:43')]
    assert np.isnan(utils.entropy(df, time_col='time'))

    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs']
    df['time'] = [pd.to_datetime('2015-04-14 07:46:43')]
    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs', 'dr5xejs', 'dr5xef2']
    df['time'] = [pd.to_datetime('2015-04-14 07:00:00'),
                  pd.to_datetime('2015-04-14 07:20:00'),
                  pd.to_datetime('2015-04-14 07:40:00')]
    ent = utils.entropy(df, time_col='time')
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
    ent = utils.entropy(df, time_col='time')
    assert ent == pytest.approx(1.0114042647073516, 0.00001)


def test_norm_entropy():
    df = pd.DataFrame(columns=['cluster', 'time'])
    assert np.isnan(utils.norm_entropy(df, time_col='time'))

    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs']
    df['time'] = [pd.to_datetime('2015-04-14 07:46:43')]
    assert np.isnan(utils.norm_entropy(df, time_col='time'))

    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs']
    df['time'] = [pd.to_datetime('2015-04-14 07:46:43')]
    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs', 'dr5xejs', 'dr5xef2']
    df['time'] = [pd.to_datetime('2015-04-14 07:00:00'),
                  pd.to_datetime('2015-04-14 07:20:00'),
                  pd.to_datetime('2015-04-14 07:40:00')]
    ent = utils.norm_entropy(df, time_col='time')
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
    ent = utils.norm_entropy(df, time_col='time')
    assert ent == pytest.approx(1.0114042 / math.log(3), 0.00001)


def test_home_stay():
    df = pd.DataFrame(columns=['cluster', 'time'])
    hs = utils.home_stay(df, 'abc', time_col='time')
    assert np.isnan(hs)

    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs', np.nan]
    df['time'] = [pd.to_datetime('2015-04-14 07:46:43'),
                  pd.to_datetime('2015-04-14 07:56:43')]
    hs = utils.home_stay(df, 'abc', time_col='time')
    assert np.isnan(hs)

    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs']
    df['time'] = [pd.to_datetime('2015-04-14 07:46:43')]
    hs = utils.home_stay(df, 'dr5xejs', time_col='time')
    assert np.isnan(hs)

    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs', 'dr5xejs']
    df['time'] = [pd.to_datetime('2015-04-14 02:00:00'),
                  pd.to_datetime('2015-04-14 03:00:00')]
    hs = utils.home_stay(df, 'dr5xejs', time_col='time')
    assert hs == pytest.approx(60, 0.00001)

    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs']
    df['time'] = [pd.to_datetime('2015-04-14 07:46:43')]
    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs', 'dr5xejs', 'dr5xef2', 'dr5xefq']
    df['time'] = [pd.to_datetime('2015-04-14 07:00:00'),
                  pd.to_datetime('2015-04-14 07:20:00'),
                  pd.to_datetime('2015-04-14 07:40:00'),
                  pd.to_datetime('2015-04-14 08:00:00')]
    hs = utils.home_stay(df, 'dr5xef2', time_col='time')
    assert hs == pytest.approx(20, 0.00001)


def test_trans_time():
    df = pd.DataFrame(columns=['cluster', 'time'])
    tt = utils.trans_time(df, time_col='time')
    assert np.isnan(tt)

    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs', np.nan]
    df['time'] = [pd.to_datetime('2015-04-14 07:40:00'),
                  pd.to_datetime('2015-04-14 08:00:00')]
    tt = utils.trans_time(df, time_col='time')
    assert tt == pytest.approx(10, 0.00001)

    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs', np.nan]
    df['time'] = [pd.to_datetime('2015-04-14 07:40:00'),
                  pd.to_datetime('2015-04-14 08:00:00')]
    df = df.set_index('time')
    tt = utils.trans_time(df)
    assert tt == pytest.approx(10, 0.00001)


def test_travel_dist():
    df = pd.DataFrame(columns=['latitude', 'longitude'])
    assert np.isnan(utils.travel_dist(df))

    df = pd.DataFrame([[20, 30]],
                      columns=['latitude', 'longitude'])
    assert utils.travel_dist(df) == pytest.approx(0, 0.001)

    df = pd.DataFrame()
    df['latitude'] = [12.3, 43.8, 32.5]
    df['longitude'] = [-45.6, 72.9, 12.9]
    assert utils.travel_dist(df) == pytest.approx(16520745.44722021, 0.00001)


def test_total_dist():
    df = pd.DataFrame([(12.3, -45.6),
                       (43.8, 72.9),
                       (32.5, 12.9)],
                      columns=['latitude', 'longitude'])
    assert utils.total_dist(df) == pytest.approx(16520745.44722021, 0.00001)

    cluster_map = {1: (12.3, -45.6),
                   2: (43.8, 72.9),
                   3: (32.5, 12.9)}
    df = pd.DataFrame([1, 2, 3],
                      columns=['cluster'])
    td = utils.total_dist(df, cluster_mapping=cluster_map)
    assert td == pytest.approx(16520745.44722021, 0.00001)
