# -*- coding: utf-8 -*-
"""
    location.test.motif_test
    ~~~~~~~~~~~~~~~~~~~~~~~~

    Unit testing motif module

    :copyright: (c) 2016 by Saeed Abdullah.

"""

import geopy
import pandas as pd
import numpy as np
import pytest
from pytest import approx

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
    assert all(a.values == [x[1] for x in l])


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


@pytest.mark.xfail
def test_generate_daily_nodes():
    raise NotImplementedError


@pytest.mark.xfail
def test_generate_nodes():
    raise NotImplementedError


@pytest.mark.xfail
def test_generate_graph():
    raise NotImplementedError


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

    assert all(expected == actual)

    # sometimes the greedy approach does not result in
    # most optimum solution. For example, if we
    # start from a corner, we will get two merged
    # grid instead of one.

    h = h.append(pd.Series(['9q8t'] * 2))

    actual = motif.merge_neighboring_grid(h)
    expected = ['9q8t', '9q8t', '9q8x', '9q8t', '9q8t', '9q8t']

    assert all(expected == actual)


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
    assert all(actual == expected)

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
    assert all(actual == expected)
