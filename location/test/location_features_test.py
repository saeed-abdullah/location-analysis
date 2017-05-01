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
