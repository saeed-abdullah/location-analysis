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
import geohash
from io import StringIO


def test_gyration_radius():
    # test data
    cluster_col = ['dr5xfdt',
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
    df['cluster'] = cluster_col
    t = pd.to_datetime('2015-04-14 06:52:00')
    df['time'] = pd.date_range(start=t, periods=10, freq='1d')
    df = df.set_index('time')
    for idx, row in df.iterrows():
        lat, lon = geohash.decode(row['cluster'])
        df.loc[idx, 'latitude'] = lat
        df.loc[idx, 'longitude'] = lon

    # expected result
    expected = 7935.926632803189

    # tolerance = 0.01 meter
    actual_value = lf.gyration_radius(df)
    assert actual_value == pytest.approx(expected, 0.01)

    # check when k is larger the number of different visited locations
    assert np.isnan(lf.gyration_radius(df, k=5))

    # add the last gps point for five more times
    p = df.index[-1]
    for i in range(5):
        p += pd.to_timedelta('1d')
        lat, lon = geohash.decode('dr5rw5u')
        df.loc[p, 'cluster'] = 'dr5rw5u'
        df.loc[p, 'latitude'] = lat
        df.loc[p, 'longitude'] = lon

    expected = 6927.0444113855365
    assert lf.gyration_radius(df) == pytest.approx(expected, 0.01)

    # test the k-th radius of gyration
    p = df.index[-1]
    for i in range(2):
        p += pd.to_timedelta('1d')
        df.loc[p, 'cluster'] = 'dr5xg5g'
        lat, lon = geohash.decode('dr5xg5g')
        df.loc[p, 'latitude'] = lat
        df.loc[p, 'longitude'] = lon
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
    ent, nent = lf.entropy(df, time_c='time')
    assert np.isnan(ent)
    assert np.isnan(nent)

    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs']
    df['time'] = [pd.to_datetime('2015-04-14 07:46:43')]
    ent, nent = lf.entropy(df, time_c='time')
    assert np.isnan(ent)
    assert np.isnan(nent)

    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs']
    df['time'] = [pd.to_datetime('2015-04-14 07:46:43')]
    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs', 'dr5xejs', 'dr5xef2']
    df['time'] = [pd.to_datetime('2015-04-14 07:00:00'),
                  pd.to_datetime('2015-04-14 07:20:00'),
                  pd.to_datetime('2015-04-14 07:40:00')]
    ent, nent = lf.entropy(df, time_c='time')
    assert ent == pytest.approx(0.5623351446188083, 0.00001)
    assert nent == pytest.approx(0.56233514 / math.log(2), 0.00001)

    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs']
    df['time'] = [pd.to_datetime('2015-04-14 07:46:43')]
    df = pd.DataFrame(columns=['cluster', 'time'])
    df['cluster'] = ['dr5xejs', 'dr5xejs', 'dr5xef2', 'dr5xefq']
    df['time'] = [pd.to_datetime('2015-04-14 07:00:00'),
                  pd.to_datetime('2015-04-14 07:20:00'),
                  pd.to_datetime('2015-04-14 07:40:00'),
                  pd.to_datetime('2015-04-14 08:00:00')]
    ent, nent = lf.entropy(df, time_c='time')
    assert ent == pytest.approx(1.0114042647073516, 0.00001)
    assert nent == pytest.approx(1.0114042 / math.log(3), 0.00001)


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


def test_convert_geohash_to_gps():
    x = geohash.encode(30, 100)
    lat, lon = lf.convert_geohash_to_gps(x)
    lat_e, lon_e = geohash.decode(x)
    assert lat == pytest.approx(lat_e, 0.00001)
    assert lon == pytest.approx(lon_e, 0.00001)


def test_convert_and_append_geohash():
    df = pd.DataFrame()
    df['cluster'] = ['dr5rw5u', 'dr5rw5u', np.nan, 'dr5rw52']
    lat1, lon1 = geohash.decode('dr5rw5u')
    lat2, lon2 = geohash.decode('dr5rw52')
    d = [['dr5rw5u', lat1, lon1],
         ['dr5rw5u', lat1, lon1],
         [np.nan, np.nan, np.nan],
         ['dr5rw52', lat2, lon2]]
    df2 = pd.DataFrame(d, columns=['cluster',
                                   'latitude',
                                   'longitude'])
    assert df2.equals(lf.convert_and_append_geohash(df))


def test_to_daily():
    data = pd.DataFrame()
    df = lf.to_daily(data)
    assert len(df) == 0

    t = '2015-10-01 00:00:00-04:00'
    t = pd.to_datetime(t).tz_localize('UTC').tz_convert('America/New_York')
    t2 = '2015-10-02 00:00:00-04:00'
    t2 = pd.to_datetime(t2).tz_localize('UTC').tz_convert('America/New_York')
    d = [['2015-10-01 05:39:46-04:00', 'dr5xfdt'],
         ['2015-10-01 17:56:13-04:00', 'dr78psd'],
         ['2015-10-02 02:27:54-04:00', 'dr78psd']]
    data = pd.DataFrame(d, columns=['time', 'stay_region'])
    data['time'] = pd.to_datetime(data['time'])
    data = data.set_index('time')
    data = data.tz_localize('UTC')
    data = data.tz_convert('America/New_York')
    df = lf.to_daily(data)
    assert len(df) == 2
    assert t == df[0][0]
    assert t2 == df[1][0]
    assert len(df[0][1]) == 2
    assert len(df[1][1]) == 1

    t = '2015-10-01 03:00:00-04:00'
    t = pd.to_datetime(t).tz_localize('UTC').tz_convert('America/New_York')
    df = lf.to_daily(data, start_hour=3)
    assert len(df) == 1
    assert t == df[0][0]
    assert len(df[0][1]) == 3

    d = [['2015-10-01 05:39:46-04:00', 'dr5xfdt'],
         ['2015-10-01 17:56:13-04:00', 'dr78psd'],
         ['2015-10-02 02:27:54-04:00', 'dr78psd'],
         ['2015-10-05 02:27:54-04:00', 'dr78psd']]
    data = pd.DataFrame(d, columns=['time', 'stay_region'])
    data['time'] = pd.to_datetime(data['time'])
    data = data.set_index('time')
    data = data.tz_localize('UTC')
    data = data.tz_convert('America/New_York')
    t = ['2015-10-01 00:00:00-04:00',
         '2015-10-02 00:00:00-04:00',
         '2015-10-03 00:00:00-04:00',
         '2015-10-04 00:00:00-04:00',
         '2015-10-05 00:00:00-04:00']
    t = [pd.to_datetime(x).tz_localize('UTC').tz_convert('America/New_York')
         for x in t]
    df = lf.to_daily(data)
    assert len(df) == 5
    assert df[0][0] == t[0]
    assert df[1][0] == t[1]
    assert df[2][0] == t[2]
    assert df[3][0] == t[3]
    assert df[4][0] == t[4]
    assert len(df[0][1]) == 2
    assert len(df[1][1]) == 1
    assert len(df[2][1]) == 0
    assert len(df[3][1]) == 0
    assert len(df[4][1]) == 1


def test_to_weekly():
    data = pd.DataFrame()
    df = lf.to_weekly(data)
    assert len(df) == 0

    t = '2017-04-24 00:00:00-04:00'
    t = pd.to_datetime(t).tz_localize('UTC').tz_convert('America/New_York')
    d = [['2017-04-24 05:39:46-04:00', 'dr5xfdt'],
         ['2017-04-25 17:56:13-04:00', 'dr78psd'],
         ['2017-04-25 02:27:54-04:00', 'dr78psd']]
    data = pd.DataFrame(d, columns=['time', 'stay_region'])
    data['time'] = pd.to_datetime(data['time'])
    data = data.set_index('time')
    data = data.tz_localize('UTC')
    data = data.tz_convert('America/New_York')
    data = data.sort_index()
    weekly = lf.to_weekly(data)
    assert len(weekly) == 1
    assert weekly[0][0] == t
    assert len(weekly[0][1]) == 3

    t = ['2017-04-24 00:00:00-04:00',
         '2017-05-01 00:00:00-04:00',
         '2017-05-08 00:00:00-04:00',
         '2017-05-15 00:00:00-04:00']
    t = [pd.to_datetime(x).tz_localize('UTC').tz_convert('America/New_York')
         for x in t]
    d = [['2017-04-24 05:39:46-04:00', 'dr5xfdt'],
         ['2017-04-25 17:56:13-04:00', 'dr78psd'],
         ['2017-04-25 22:27:54-04:00', 'dr78psd'],
         ['2017-05-03 02:27:54-04:00', 'dr78psd'],
         ['2017-05-17 02:27:54-04:00', 'dr78psd']]
    data = pd.DataFrame(d, columns=['time', 'stay_region'])
    data['time'] = pd.to_datetime(data['time'])
    data = data.set_index('time')
    data = data.tz_localize('UTC')
    data = data.tz_convert('America/New_York')
    data = data.sort_index()
    weekly = lf.to_weekly(data)
    assert len(weekly) == 4
    assert weekly[0][0] == t[0]
    assert len(weekly[0][1]) == 3
    assert weekly[1][0] == t[1]
    assert len(weekly[1][1]) == 1
    assert weekly[2][0] == t[2]
    assert len(weekly[2][1]) == 0
    assert weekly[3][0] == t[3]
    assert len(weekly[3][1]) == 1


def test_load_locatoin_data():
    df = pd.DataFrame([['2017-05-17 02:27:54-04:00', 1, 2, 'a'],
                       ['2017-05-13 02:27:54-04:00', 3, 4, 'b']],
                      columns=['time', 'lat', 'lon', 'cluster'])
    f = StringIO()
    df.to_csv(f)
    f.seek(0)
    args = {'time_c': 'time',
            'lat_c': 'lat',
            'lon_c': 'lon',
            'cluster_c': 'cluster'}

    loaded = lf._load_location_data(f, **args)
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time', drop=False)
    df = df.sort_index()
    assert loaded.equals(df)
