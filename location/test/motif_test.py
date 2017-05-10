# -*- coding: utf-8 -*-
"""
    location.test.motif_test
    ~~~~~~~~~~~~~~~~~~~~~~~~

    Unit testing motif module

    :copyright: (c) 2016 by Saeed Abdullah.

"""

from io import StringIO
from unittest.mock import ANY, patch

import geopy
import pandas as pd
import numpy as np
import math
import networkx as nx
import pytest
from pytest import approx
import geohash
from geopy.distance import vincenty

from location import motif


def get_nearby_point(lon, lat, dist_m, bearing=0):
    """
   Generates a nearby point with given distance and bearing.

   Parameters
   ----------
   lon: float
       Longitude of the origin.
   lat: float
       Latitude of the origin.
   dist_m: float
       Distance in meters
   bearing: float
       Bearing in degrees.


   Returns
   -------
   p: geopy.Point
       A point with given distance and bearing from the origin.
    """

    origin = geopy.Point(longitude=lon, latitude=lat)
    dist = geopy.distance.GreatCircleDistance(meters=dist_m)
    return dist.destination(origin, bearing=bearing)


def test_compute_geo_hash():
    coords = [{'lat': 0, 'lon': 0},
              {'lat': -90, 'lon': 0},
              {'lat': 0, 'lon': -90},
              ]
    df = pd.DataFrame(coords)
    expected = ['s00000000000', 'h00000000000', 'd00000000000']

    l = motif.compute_geo_hash(df)
    assert l == expected

    l = motif.compute_geo_hash(df, precision=2)
    assert l == [x[:2] for x in expected]


def test_trim_geo_hash_precision():
    l = ['11', '22', '33']
    a = motif.trim_geo_hash_precision(pd.Series(l),
                                      precision=1)
    assert np.all(a == [x[1] for x in l])


def test_filter_out_rare_points():
    r = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]  # 10 elements

    l = motif.filter_out_rare_points(r)
    assert l == r

    l = motif.filter_out_rare_points(r, 10)
    expected = [np.nan] + r[1:]
    assert l == expected

    l = motif.filter_out_rare_points(r, 100)
    assert l == [np.nan] * len(r)


def test_get_primary_location():
    l = ['1', '2', '2', '-1', '-1', '-1', '3']
    c = motif.get_primary_location(pd.Series(l))

    assert c == '-1'

    # should throw a ValueError if pass a different aggr_f
    with pytest.raises(ValueError):
        motif.get_primary_location(pd.Series(l), aggr_f='not-count')


def test_generate_daily_nodes():
    h = list(range(48))
    start = pd.to_datetime('2016-11-16')
    t = pd.date_range(start, periods=len(h), freq='30min')
    df = pd.DataFrame({'geo_hash': h}, index=t)

    actual = motif.generate_daily_nodes(df)
    expected = pd.DataFrame({'time': t, 'node': h})

    assert actual[0][0] == start
    assert expected.equals(actual[0][1])

    # start and end date
    end = start + pd.to_timedelta('2D')
    actual = motif.generate_daily_nodes(df, start_date=start,
                                        end_date=end)
    expected = pd.DataFrame({'time': t, 'node': h})

    assert len(actual) == 2
    assert actual[0][0] == start
    assert expected.equals(actual[0][1])

    # next day
    assert actual[1][0] == start + pd.to_timedelta('1D')
    expected = pd.DataFrame({'time': t + pd.to_timedelta('1D'),
                             'node': [np.nan] * len(t)})
    assert expected.equals(actual[1][1])

    # shift start of the day
    shift = '2hr'
    actual = motif.generate_daily_nodes(df, shift_day_start=shift)

    assert len(actual) == 1
    # the first 4 records (from 2 hr shift) now belongs to
    # previous day
    assert actual[0][0] == start + pd.to_timedelta(shift)
    expected = pd.DataFrame({'time': t + pd.to_timedelta(shift),
                             'node': h[4:] + [np.nan] * 4})

    # rare point threshold
    actual = motif.generate_daily_nodes(df, rare_pt_pct_th=3.0)
    expected = pd.DataFrame({'time': t, 'node': [np.nan] * len(t)})
    assert len(actual) == 1
    assert actual[0][0] == start
    assert expected.equals(actual[0][1])

    # valid count per day threshold
    actual = motif.generate_daily_nodes(df, valid_day_th=49)
    assert len(actual) == 1
    assert actual[0][0] == start
    assert np.isnan(actual[0][1])

    # keywords to generate_nodes()
    node_args = {'valid_interval_th': 2}
    actual = motif.generate_daily_nodes(df, node_args=node_args)
    expected = pd.DataFrame({'time': t, 'node': [np.nan] * len(t)})
    assert len(actual) == 1
    assert actual[0][0] == start
    assert expected.equals(actual[0][1])


def test_generate_nodes():
    # hash
    h = list(range(48))
    start = pd.to_datetime('2016-11-16 14:00')
    t = pd.date_range(start, periods=len(h), freq='30min')
    series = pd.Series(h, index=t)

    actual = motif.generate_nodes(series, start_time=start)
    expected = pd.DataFrame({'time': t, 'node': h})
    assert actual.equals(expected)

    # higher threshold for record numbers
    actual = motif.generate_nodes(series, start_time=start,
                                  valid_interval_th=2)
    expected = pd.DataFrame({'node': [np.nan] * len(h), 'time': t})
    assert actual.equals(expected)

    # end_time
    end = start + pd.to_timedelta('2D')
    actual = motif.generate_nodes(series, start_time=start,
                                  end_time=end)
    t = pd.date_range(start, periods=len(h) * 2, freq='30min')
    # the second half should be nan
    expected = pd.DataFrame({'node': h + [np.nan] * len(h), 'time': t})
    assert actual.equals(expected)

    # time_interval
    t = pd.date_range(start, periods=len(h), freq='15min')
    series = pd.Series(h, index=t)
    # similar to range, get_df_slices stops before the last element
    # so, we need to add one more element
    end = t[-1] + pd.to_timedelta('15min')
    actual = motif.generate_nodes(series, start_time=start,
                                  end_time=end,
                                  time_interval='15min')
    expected = pd.DataFrame({'node': h, 'time': t})
    assert actual.equals(expected)

    # the time index does not need to be aligned exactly
    s = start + pd.to_timedelta('5min')
    t = pd.date_range(s, periods=len(h), freq='30min')
    series = pd.Series(h, index=t)
    actual = motif.generate_nodes(series, start_time=start)

    # note that it time index will align with start
    expected = pd.DataFrame({'node': h,
                             'time': t - pd.to_timedelta('5min')})
    assert actual.equals(expected)


def test_generate_graph():
    s = ['a', 'b', 'c']
    actual = motif.generate_graph(pd.Series(s))
    expected = ['a b', 'b c']

    assert expected == actual

    # adding NaN would not change it
    s.insert(2, np.nan)
    s.insert(0, np.nan)

    actual = motif.generate_graph(pd.Series(s))
    assert expected == actual


def test_get_geo_center():
    coords = [{'lat': -21.1333, 'lon': -175.2},
              {'lat': -8.53333, 'lon': 179.2167}]
    df = pd.DataFrame(coords)

    d = motif.get_geo_center(df, lat_c='lat', lon_c='lon')

    assert d['latitude'] == approx(-14.83331)
    assert d['longitude'] == approx(-177.9916)


def test_get_stay_point():
    coords = [(-76.48327, 42.44701),  # origin
              # within 200m radius from origin
              (-76.48443561343255, 42.448589090744434),
              (-76.48560525440746, 42.44752375175882),
              # new origin which is 400m away
              (-76.48713118353778, 42.44920446354337),
              # within 300m block from new origin
              (-76.49037416236197, 42.44795997470675),
              (-76.48985180197955, 42.44740312340601)]

    start = pd.to_datetime('2016-11-16 14:00:00')
    time = pd.date_range(start, periods=len(coords), freq='15min')

    l = []
    for c, t in zip(coords, time):
        l.append({'latitude': c[1], 'longitude': c[0], 'time': t})

    df = pd.DataFrame(l)
    df = df.set_index('time').sort_index()

    stay_points = motif.get_stay_point(df)
    expected = [0, 0, 0, 1, 1, 1]
    assert stay_points == expected

    # smaller distance threshold
    stay_points = motif.get_stay_point(df, dist_th=200)
    expected = [0, 0, 0, np.nan, np.nan, np.nan]
    assert stay_points == expected

    # different time threshold
    stay_points = motif.get_stay_point(df, time_th='31min')
    expected = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    assert stay_points == expected


def test_merge_neighboring_grid():

    #
    # +-------+--------+--------+
    # | 9q8x  |   9q8z |  9q9p  |
    # |       |        |        |
    # +-------------------------+
    # | 9q8w  |   9q8y |  9q9n  |
    # |       |        |        |
    # +-------------------------+
    # | 9q8t  |   9q8v |  9q9j  |
    # |       |        |        |
    # +-------+--------+--------+
    #

    # the center grid (9q8y) is the most common
    # so all the grid will have same geohash
    h = pd.Series(['9q8y', '9q8y', '9q8x', '9q8t'])

    actual = motif.merge_neighboring_grid(h)
    expected = ['9q8y'] * len(h)

    assert np.all(expected == actual)

    # sometimes the greedy approach does not result in
    # most optimum solution. For example, if we
    # start from a corner, we will get two merged
    # grid instead of one.

    h = h.append(pd.Series(['9q8t'] * 2))

    actual = motif.merge_neighboring_grid(h)
    expected = ['9q8t', '9q8t', '9q8x', '9q8t', '9q8t', '9q8t']

    assert np.all(expected == actual)


def test_get_stay_region():

    coords = [(-122.52065742012005, 37.707623920764846),  # in 9q8y
              (-122.51778693308258, 37.70276073065426),
              (-122.51919726220027, 37.70776235552409),
              (-122.17130080126853, 37.529837604742944),  # in 9q9j
              (-122.16544209504582, 37.53110242860379),
              (-122.87447180947525, 37.52898358762108),  # in 9q8t
              (-122.87446635765954, 37.528948605061274)]

    # reflects the grids
    stay_point = [0, 0, 0, 1, 1, 2, 2]

    l = []
    for c, s in zip(coords, stay_point):
        l.append({'lon': c[0], 'lat': c[1], 'stay_point': s})

    df = pd.DataFrame(l)
    actual = motif.get_stay_region(df, lat_c='lat',
                                   lon_c='lon', precision=4)
    expected = ['9q8y'] * len(coords)
    assert np.all(actual == expected)

    # add more points to 9q8t so that the merging starts
    # from there. It will result in two regions
    df = df.append({'lon': -122.87401872818019,
                    'lat': 37.527920601182295,
                    'stay_point': 2}, ignore_index=True)
    df = df.append({'lon': -122.87396151226133,
                    'lat': 37.53074696993352,
                    'stay_point': 2}, ignore_index=True)

    actual = motif.get_stay_region(df, lat_c='lat',
                                   lon_c='lon', precision=4)
    # see the grids in test_get_stay_point`
    expected = ['9q8t', '9q8t', '9q8t', '9q9j', '9q9j',
                '9q8t', '9q8t', '9q8t', '9q8t']
    assert np.all(actual == expected)


def test_save_nodes():
    h = list(range(48))
    start = pd.to_datetime('2016-11-16', utc=True).tz_convert('US/Eastern')
    t = pd.date_range(start, periods=len(h), freq='30min')

    node = pd.DataFrame({'time': t, 'node': h})

    # next day
    next_s = start + pd.to_timedelta('1D')
    next_node = pd.DataFrame({'time': t + pd.to_timedelta('1D'), 'node': h})

    nodes = [(start, node), (next_s, next_node)]

    expected = node.copy()
    expected['time'] = expected.time.map(lambda z: z.tz_convert('UTC'))
    expected['timestamp'] = start.tz_convert('UTC')
    expected['tz'] = start.tz

    # next day
    n = next_node.copy()
    n['time'] = n.time.map(lambda z: z.tz_convert('UTC'))
    n['timestamp'] = next_s.tz_convert('UTC')
    n['tz'] = next_s.tz

    expected = pd.concat([expected, n])

    f = StringIO()
    motif._save_nodes(nodes, f)

    assert f.getvalue() == expected.to_csv()


def test_load_nodes():
    h = list(range(48))
    start = pd.to_datetime('2016-11-16', utc=True).tz_convert('US/Eastern')
    t = pd.date_range(start, periods=len(h), freq='30min')

    node = pd.DataFrame({'time': t, 'node': h})

    expected = node.copy()
    expected['timestamp'] = start
    expected['tz'] = start.tz

    f = StringIO()
    expected.to_csv(f)
    f.seek(0)

    actual = motif._load_nodes(f)
    assert len(actual) == 1
    assert actual[0][0] == start
    # columns in same order
    assert node.equals(actual[0][1].sort_index(axis=1))


def test_compute_nodes():

    # We need at least 8 records for generate_daily_nodes
    # And, for each of this value, we should have a stay
    # point. This means, we need
    #    i) at least two subsequent within 300m of each other,
    #   ii) the timestamp between each pair should be >= 30 mins
    #   iii) each pair should be followed by another point which
    #   is suffciently far enough (> 300 m)

    coords = [(-76.48327, 42.44701),
              (-76.4761338923341, 42.44583908268239)]
    time = []
    longitudes = []
    latitudes = []
    start = pd.to_datetime('2016-11-16')
    for index, h in enumerate(range(8)):
        s = start + pd.to_timedelta('{0}h'.format(h))
        time.append(s)

        # next time stamp should be at least 30 mins away
        n = s + pd.to_timedelta('45min')
        time.append(n)

        c = coords[index % 2 == 0]
        # we need two records
        longitudes.extend([c[0]] * 2)
        latitudes.extend([c[1]] * 2)

    df = pd.DataFrame({'longitude': longitudes,
                       'latitude': latitudes}, index=time)

    stay, nodes = motif.compute_nodes(df, lon_c='longitude',
                                      lat_c='latitude')
    # each point repeated twice
    expected_points = [i//2 for i in range(len(time))]

    expected = df.copy()
    expected['stay_point'] = expected_points
    expected['stay_region'] = motif.compute_geo_hash(expected,
                                                     lat_c='latitude',
                                                     lon_c='longitude',
                                                     precision=7)

    # columns in same order
    expected = expected.sort_index(axis=1)
    assert expected.equals(stay.sort_index(axis=1))

    # expected nodes
    index = pd.date_range(start=start, periods=48, freq='30min')
    n = expected.stay_region.tolist() + [np.nan] * (48 - len(expected))
    expected_nodes = pd.DataFrame({'node': n, 'time': index})

    assert nodes[0][0] == start
    assert expected_nodes.equals(nodes[0][1])

    # check the stay_point_args
    with patch.object(motif, 'get_stay_point', return_value=1) as p:
        args = {'dist_th': 400, 'time_th': '60m'}
        motif.compute_nodes(df, lon_c='longitude', lat_c='latitude',
                            stay_point_args=args)
    p.assert_called_once_with(ANY, lat_c='latitude', lon_c='longitude',
                              dist_th=400,
                              time_th='60m')

    # check the stay_region_args
    with patch.object(motif, 'get_stay_region', return_value=1) as p:
        args = {'precision': 10}
        motif.compute_nodes(df, lon_c='longitude', lat_c='latitude',
                            stay_region_args=args)
    p.assert_called_once_with(ANY, lat_c='latitude', lon_c='longitude',
                              precision=10)

    # check the node_args
    with patch.object(motif, 'generate_nodes', return_value=[]) as p:
        args = {'end_time': 'end_time', 'time_interval': '80Min',
                'valid_interval_th': 12}
        motif.compute_nodes(df, lon_c='longitude', lat_c='latitude',
                            node_args=args)
    p.assert_called_once_with(ANY, start_time=start, end_time='end_time',
                              time_interval='80Min',
                              valid_interval_th=12)

    # check the daily_args
    with patch.object(motif, 'generate_daily_nodes') as p:
        args = {'geo_hash_preicion': 2, 'shift_day_start': 'shift',
                'rare_pt_pct_th': 10, 'valid_day_th': 20,
                'start_date': 'start', 'end_date': 'end'}
        motif.compute_nodes(df, lon_c='longitude', lat_c='latitude',
                            daily_args=args)

    p.assert_called_once_with(ANY, hash_c='stay_region', geo_hash_preicion=2,
                              shift_day_start='shift',
                              rare_pt_pct_th=10,
                              valid_day_th=20,
                              start_date='start',
                              end_date='end',
                              node_args=ANY)

    # check the stay_info_output
    with patch.object(pd.DataFrame, 'to_csv') as p:
        motif.compute_nodes(df, lon_c='longitude', lat_c='latitude',
                            stay_info_output='output')

    p.assert_called_once_with('output')

    # check the stay_info_output
    with patch.object(motif, '_save_nodes') as p:
        motif.compute_nodes(df, lon_c='longitude', lat_c='latitude',
                            node_output='node')

    p.assert_called_once_with(ANY, 'node')


def test_filter_inadequate_nodes():
    node = pd.DataFrame()
    timestamp = pd.Timestamp('2016-01-07 03:30:00-0500')
    node['time'] = pd.date_range(timestamp, periods=48, freq='30min')
    n = [np.nan] * 41
    n.extend(['dr5xg5g'] * 7)
    node['node'] = n.copy()
    nodes = [(timestamp, node.copy())]

    node = pd.DataFrame()
    timestamp = pd.Timestamp('2016-01-08 03:30:00-0500')
    node['time'] = pd.date_range(timestamp, periods=48, freq='30min')
    n = [np.nan] * 40
    n.extend(['dr5xg5g'] * 8)
    node['node'] = n.copy()
    nodes.append((timestamp, node.copy()))

    nodes = motif.filter_inadequate_nodes(nodes)

    assert len(nodes) == 1


def test_insert_home_location():
    timestamp = pd.Timestamp('2016-01-07 03:30:00-0500')
    df = pd.DataFrame()
    stay_region = ['dr5rw5u'] * 20
    stay_region.extend(['dr5xg57'] * 70)
    df['stay_region'] = stay_region
    df['time'] = pd.date_range(timestamp, periods=90, freq='15min')
    df = df.set_index('time')

    node = pd.DataFrame()
    node['time'] = pd.date_range(timestamp, periods=48, freq='30min')
    n = [np.nan] * 40
    n.extend(['dr5xg5g'] * 8)
    node['node'] = n
    nodes = [(timestamp, node.copy())]

    nodes = motif.insert_home_location(df, nodes)
    assert nodes[0][1].ix[0, 'node'] == 'dr5rw5u'

    nodes = motif.insert_home_location(df, nodes, home='dr5xg57')
    assert nodes[0][1].ix[0, 'node'] == 'dr5rw5u'

    nodes[0][1].ix[0, 'node'] = np.nan
    nodes = motif.insert_home_location(df, nodes, home='dr5xg57')
    assert nodes[0][1].ix[0, 'node'] == 'dr5xg57'


def test_filter_days_without_round_trip():
    # a day without round trip
    timestamp = pd.Timestamp('2016-01-07 03:30:00-0500')
    node = pd.DataFrame()
    node['time'] = pd.date_range(timestamp, periods=48, freq='30min')
    n = [np.nan] * 39
    n.append('dr5rw5u')
    n.extend(['dr5xg5g'] * 8)
    node['node'] = n.copy()
    nodes = [(timestamp, node.copy())]

    # a day with round trip
    timestamp = pd.Timestamp('2016-01-08 03:30:00-0500')
    node = pd.DataFrame()
    node['time'] = pd.date_range(timestamp, periods=48, freq='30min')
    n = ['dr5xg5g']
    n.extend(['dr5rw5u'] * 39)
    n.extend(['dr5xg5g'] * 8)
    node['node'] = n.copy()
    nodes.append((timestamp, node.copy()))

    nodes = motif.filter_days_without_round_trip(nodes)
    assert len(nodes) == 1
    assert nodes[0][0] == timestamp


def test_filter_out_travelling_day():
    timestamp = pd.Timestamp('2016-01-07 03:30:00-0500')
    df = pd.DataFrame()
    stay_region = ['dr5rw5u'] * 20
    stay_region.extend(['dr5xg57'] * 70)
    df['stay_region'] = stay_region
    df['time'] = pd.date_range(timestamp, periods=90, freq='15min')
    df = df.set_index('time')

    node = pd.DataFrame()
    node['time'] = pd.date_range(timestamp, periods=48, freq='30min')
    n = [np.nan] * 40
    n.extend(['dr5xg5g'] * 8)

    distance_pt = (42.447909, -76.477998, 8)
    distance_pt_geosh = geohash.encode(distance_pt[0], distance_pt[1], 8)
    n[20] = distance_pt_geosh
    th = vincenty(distance_pt, geohash.decode('dr5rw5u')).m

    node['node'] = n.copy()
    nodes = [(timestamp, node.copy())]

    filtered_nodes = motif.filter_out_travelling_day(df,
                                                     nodes,
                                                     trav_dist_th=th - 100)
    assert len(filtered_nodes) == 0
    filtered_nodes = motif.filter_out_travelling_day(df,
                                                     nodes,
                                                     trav_dist_th=th + 100)
    assert len(filtered_nodes) == 1
    assert filtered_nodes[0][0] == timestamp


def test_filter_weekday():
    # a Monday
    timestamp = pd.Timestamp('2016-12-12 03:30:00-0500')
    node = pd.DataFrame()
    node['time'] = pd.date_range(timestamp, periods=48, freq='30min')
    n = [np.nan] * 40
    n.extend(['dr5xg5g'] * 8)
    node['node'] = n.copy()
    nodes = [(timestamp, node.copy())]

    # a Tuesday
    timestamp = pd.Timestamp('2016-12-13 03:30:00-0500')
    node = pd.DataFrame()
    node['time'] = pd.date_range(timestamp, periods=48, freq='30min')
    n = ['dr5xg5g']
    n.extend([np.nan] * 39)
    n.extend(['dr5xg5g'] * 8)
    node['node'] = n.copy()
    nodes.append((timestamp, node.copy()))

    # a Saturday
    timestamp = pd.Timestamp('2016-12-24 03:30:00-0500')
    node = pd.DataFrame()
    node['time'] = pd.date_range(timestamp, periods=48, freq='30min')
    n = ['dr5xg5g']
    n.extend([np.nan] * 39)
    n.extend(['dr5xg5g'] * 8)
    node['node'] = n.copy()
    nodes.append((timestamp, node.copy()))

    filtered_nodes = motif.filter_weekday(nodes)
    assert len(filtered_nodes) == 2
    assert filtered_nodes[0][0].weekday() == 0
    assert filtered_nodes[1][0].weekday() == 1

    filtered_nodes = motif.filter_weekday(nodes, dayofweek=[0])
    assert len(filtered_nodes) == 1
    assert filtered_nodes[0][0] == pd.Timestamp('2016-12-12 03:30:00-0500')


def test_generate_motifs():
    # day 1
    node = pd.DataFrame()
    timestamp1 = pd.Timestamp('2016-01-07 03:30:00-0500')
    node['time'] = pd.date_range(timestamp1, periods=48, freq='30min')
    n = [np.nan, 'dr5rw5u']
    n.extend(['dr5xg57'] * 5)
    n.extend(['dr5xg5g'] * 4)
    n.append(np.nan)
    n.extend(['dr5rw5u'] * 36)
    node['node'] = n.copy()
    nodes = [(timestamp1, node.copy())]

    # day 2
    timestamp2 = pd.Timestamp('2016-01-08 03:30:00-0500')
    node = pd.DataFrame()
    node['time'] = pd.date_range(timestamp2, periods=48, freq='30min')
    n = [np.nan] * 20
    n.extend(['dr5rw5u'] * 20)
    n.extend(['dr5xg57'] * 8)
    node['node'] = n.copy()
    nodes.extend([(timestamp2, node.copy())])

    # day 3
    timestamp3 = pd.Timestamp('2016-01-09 03:30:00-0500')
    node = pd.DataFrame()
    node['time'] = pd.date_range(timestamp3, periods=48, freq='30min')
    n = [np.nan] * 20
    n.extend(['dr5xg57'] * 20)
    n.extend(['dr5xg5g'] * 8)
    node['node'] = n.copy()
    nodes.extend([(timestamp3, node.copy())])

    # location information
    df = pd.DataFrame()
    stay_region = ['dr5xg5g'] * 90
    df['stay_region'] = stay_region
    df['time'] = pd.date_range(timestamp1, periods=90, freq='15min')
    df = df.set_index('time')
    df = pd.concat([df, df, df])

    motifs = motif.generate_motifs(df, nodes,
                                   insert_home=False,
                                   round_trip=False)
    # sort motifs by frequency
    motifs.sort(key=lambda x: len(x['data']), reverse=True)

    # number of motifs
    assert len(motifs) == 2

    # graph isomorphism
    adjacency_matrix = [[0, 1], [0, 0]]
    expected_graph1 = nx.from_numpy_matrix(np.array(adjacency_matrix.copy()),
                                           create_using=nx.MultiDiGraph())
    assert nx.is_isomorphic(motifs[0]['graph'], expected_graph1)
    assert len(motifs[0]['data']) == 2
    assert timestamp2 in motifs[0]['data']
    assert timestamp3 in motifs[0]['data']

    adjacency_matrix = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    expected_graph2 = nx.from_numpy_matrix(np.array(adjacency_matrix.copy()),
                                           create_using=nx.MultiDiGraph())
    assert nx.is_isomorphic(motifs[1]['graph'], expected_graph2)
    assert len(motifs[1]['data']) == 1
    assert timestamp1 in motifs[1]['data']

    # test round_trip parameter
    motifs = motif.generate_motifs(df, nodes,
                                   insert_home=False,
                                   round_trip=True)
    assert len(motifs) == 1
    assert len(motifs[0]['data']) == 1
    assert motifs[0]['data'][0] == timestamp1
    assert nx.is_isomorphic(motifs[0]['graph'], expected_graph2)

    # test insert_home parameter
    motifs = motif.generate_motifs(df, nodes,
                                   insert_home=True,
                                   round_trip=True)
    assert len(motifs) == 1
    assert len(motifs[0]['data']) == 1
    assert motifs[0]['data'][0] == timestamp3
    adjacency_matrix = [[0, 1], [1, 0]]
    expected_graph3 = nx.from_numpy_matrix(np.array(adjacency_matrix.copy()),
                                           create_using=nx.MultiDiGraph())
    assert nx.is_isomorphic(motifs[0]['graph'], expected_graph3)


def test_get_home_location():
    timestamp = pd.Timestamp('2016-01-07 03:30:00-0500')
    df = pd.DataFrame()
    stay_region = ['dr5rw5u'] * 20
    stay_region.extend(['dr5xg57'] * 70)
    df['stay_region'] = stay_region
    df['time'] = pd.date_range(timestamp, periods=90, freq='15min')
    df = df.set_index('time')

    home = motif.get_home_location(df)
    assert home == 'dr5rw5u'

    timestamp = pd.Timestamp('2016-01-07 03:30:00-0500')
    df = pd.DataFrame()
    stay_region = [np.nan] * 20
    stay_region.extend(['dr5xg57'] * 50)
    stay_region.extend([np.nan] * 20)
    df['stay_region'] = stay_region
    df['time'] = pd.date_range(timestamp, periods=90, freq='15min')
    df = df.set_index('time')

    home = motif.get_home_location(df)
    assert home is None
