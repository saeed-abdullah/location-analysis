# -*- coding: utf-8 -*-
"""
    location.test.motif_test
    ~~~~~~~~~~~~~~~~~~~~~~~~

    Unit testing motif module

    :copyright: (c) 2016 by Saeed Abdullah.

"""

import pandas as pd
import numpy as np
import pytest
from pytest import approx

from location import motif


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


@pytest.mark.xfail
def test_get_stay_point():
    raise NotImplementedError


@pytest.mark.xfail
def test_merge_neighboring_grid():
    raise NotImplementedError


@pytest.mark.xfail
def test_get_stay_region():
    raise NotImplementedError
