# -*- coding: utf-8 -*-
"""
    utils
    ~~~~~

    Utility functions for motif analysis.
"""

import pandas as pd
import numpy as np
import math
from copy import deepcopy
from collections import Counter
from geopy.distance import vincenty
import geohash

from location import motif


def approx_home_location(data, sr_col='stay_region'):
    """
    Calculate the home location based on the assumption that the most
    frequently visited location during 24:00 to 6:00 is the home location.

    Parameters:
    -----------
    data: Dataframe
        User location data

    sr_col: str
        Column name for stay region.
        Default is 'stay_region'.

    Returns:
    --------
    home: str
        Home location in geohash value.
    """
    loc_data = data.copy()
    loc_data = loc_data[pd.notnull(loc_data[sr_col])]
    loc_data['hour'] = [x.hour for x in loc_data.index]
    # get the location data in the target hours
    trg_data = loc_data.loc[loc_data['hour'].isin([0, 1, 2, 3, 4, 5])]
    # if no data during 0:00 from 5:00, use location data from 20:00 to 8:00
    if len(trg_data) == 0:
        trg_data = loc_data.loc[loc_data['hour'].isin([20, 21, 23, 6, 7])]

    # if there is still no data in corresponding time period,
    # select the most frequently visited locatoins as home location
    if len(trg_data) == 0:
        trg_data = loc_data.copy()

    # return home location, that is the most frequently visited locatoin
    home = Counter(trg_data[sr_col]).most_common()[0][0]

    return home


def compute_total_gyration(data, sr_col='stay_region'):
    """
    Computes the total radius of gyration.

    Parameters:
    -----------

    data: DataFrame
        Location data.

    sr_col: str
        Column name for stay region.
        Default is 'stay_region'.

    Returns:
    --------

    (radius of gyration): float
        Radius gyration in meter.
    """
    loc_data = data.copy()
    loc_data = loc_data[pd.notnull(loc_data[sr_col])]
    # different visited locations
    visited_locs = np.unique(loc_data[sr_col])

    # compute coordinates
    locs_hist_coord = [geohash.decode(x) for x in loc_data[sr_col]]
    loc_data['latitude'] = [x[0] for x in locs_hist_coord]
    loc_data['longitude'] = [x[1] for x in locs_hist_coord]

    # compute mass of locations
    r_cm = motif.get_geo_center(loc_data, lat_c='latitude', lon_c='longitude')
    r_cm = (r_cm['latitude'], r_cm['longitude'])

    # compute gyration of radius
    temp_sum = 0
    loc_cnt = Counter(loc_data[sr_col])
    N = sum(loc_cnt.values())

    for loc in visited_locs:
        temp_sum += loc_cnt[loc] * vincenty(geohash.decode(loc), r_cm).m ** 2

    return math.sqrt(1/N * temp_sum)


def compute_gyration(data,
                     sr_col='stay_region',
                     k=None,
                     remove_dist_pt=True,
                     home=None,
                     dist_pt_th=50000):
    """
    Compute radius of gyration.

    Parameters:
    -----------

    data: DataFrame
        Location data.

    sr_col: str
        Column name for stay region.
        Default is 'stay_region'.

    k: int
        k-th radius of gyration.
        Default is None, in this case return total radius of gyration.
        k-th radius of gyration is the radius gyration compuated up to
        the k-th most frequent visited locations.

    remove_dist_pt: bool
        Whether remove points that are far away from home location.
        Default is True, that is do remove far away points.

    home: str
        Geohash string for home location.
        Default is None. In this case, approximate home location from
        given location data.

    dist_pt_th: int
        Threshold in meter to filter out distant points.

    Returns:
    (radius of gyration): float
        Radius of gyration in meters.
        Return np.nan is k is greater than the number of different
        visited locations.
    """
    loc_data = data.copy()
    loc_data = loc_data[pd.notnull(loc_data[sr_col])]

    visited_locs = np.unique(loc_data[sr_col])

    # remove points that are not in the same city
    if remove_dist_pt:
        if home is None:
            home = approx_home_location(loc_data)
        home_coord = geohash.decode(home)
        visited_locs_filtered = []
        for x in visited_locs:
            dist = vincenty(home_coord, geohash.decode(x)).m
            if dist <= dist_pt_th:
                visited_locs_filtered.append(x)
        loc_data = loc_data.loc[loc_data[sr_col].isin(visited_locs_filtered)]

    # compute gyration of radius
    if k is None:
        return compute_total_gyration(loc_data, sr_col=sr_col)
    else:
        cnt_locs = Counter(loc_data[sr_col])
        # number of different visited locations
        num_visited_locations = len(cnt_locs)
        if k > num_visited_locations:
            return np.nan
        else:
            # k most frequent visited locations
            k_locations = cnt_locs.most_common()[:k]
            k_locations = [x[0] for x in k_locations]
            # compute gyration for the k most frequent locations
            loc_data = loc_data.loc[loc_data[sr_col].isin(k_locations)]
            return compute_total_gyration(loc_data, sr_col=sr_col)


def compute_rec_ratio(data, k):
    """
    Compute recurrent ratio.

    Parameters:
    -----------

    data: DataFrame
        Location data.

    k: int
        k-th radius of gyration.


    Returns:
    --------

    (recurrent ratio): float
        Recurrent ratio.
        If k is larger than the number of different visited
        locations, return np.nan.
    """
    total_raidus_gyration = compute_gyration(data)
    k_th_radius_gyration = compute_gyration(data, k=k)

    # if k_th radius gyration is nan, return nan
    if np.isnan(k_th_radius_gyration):
        return np.nan
    else:
        return k_th_radius_gyration / total_raidus_gyration


def compute_regularity(data, sr_col='stay_region'):
    """
    Calculate mobility regularity R(t), which is defined as the probability of
    finding the user in her/his most visited location at hourly interval in a
    week (Jiang S et al, 2010)
    [http://humnetlab.mit.edu/wordpress/wp-content/uploads/2010/10/ACM13_ShanJiangII.pdf]

    Parameters:
    -----------
    data: DataFrame
        Location data.

    sr_col: str
        Column name for stay region.
        Default is 'stay_region'.

    Returns:
    --------
    reg: DataFrame
        Mobility regularity in hourly interval in a week.
    """

    # a dictionary stores location visited for intervals,
    # there are 7x4 hourly intervals,
    # each key is a tuple of dayofweek and hour. (dayofweek, hour)
    # where dayofweek is in [0..6] standing for [Monday,..,Sunday],
    # hour is in [0..23], each value is a list of visted location
    # during that interval.
    # eg. {(5,13):[a,b,c,a]} means during 13:00 - 14:00 on Fridays the
    # participant visited location [a,b,c,a]
    loc_dict = {}

    # simliart to loc_dict except the value if the mobility regularity
    # eg. {(5,13):0.6} means during 13:00 - 14:00
    # on Fridays the regularity is 0.6
    reg_dict = {}

    # initialize reg_dict and reg_list
    for day in range(7):        # for each day of week
        for hour in range(24):    # for each hour of the day
            loc_dict[(day, hour)] = []
            reg_dict[(day, hour)] = 0

    # group locatino data based on hour and dayofweek
    loc_data = data.copy()
    loc_data['hour'] = [x.hour for x in loc_data.index]
    loc_data['dayofweek'] = [x.dayofweek for x in loc_data.index]
    grouped = loc_data.groupby(['dayofweek', 'hour'])

    # compute visited locations for each interval
    for index, group in grouped:

        # get day of week and
        dayofweek = index[0]
        hour = index[1]

        # add visited locations to corresponding interval
        visited_locations = group[sr_col].dropna()
        loc_dict[(dayofweek, hour)].extend(visited_locations)

    # compute regularity
    for key, value in loc_dict.items():

        # total number of visited locations with duplicates
        num_locations = len(value)

        # get most frequent visited location and calculate its frequency
        counter_locs = Counter(value)
        if len(counter_locs) > 0:
            num_most_freq = counter_locs.most_common()[0][1]
            reg_dict[(key[0], key[1])] = num_most_freq / num_locations

    # convert regulariy infomation in R to a list in time order
    # from 0:00 Monday to 23:00 Sunday
    reg = pd.DataFrame(columns=['weekday', 'hour', 'regularity'])
    weekday = -1
    for day in range(7):
        tmp = {}
        weekday += 1
        tmp['weekday'] = [weekday]*24
        tmp['hour'] = list(range(24))
        regs = []
        for hour in range(24):
            regs.append(reg_dict[day, hour])
        tmp['regularity'] = regs
        reg = pd.concat([reg, pd.DataFrame(tmp)])
    reg.hour = reg.hour.astype(int).tolist()
    reg.weekday = reg.weekday.astype(int).tolist()
    reg = reg.set_index(['weekday', 'hour'])
    reg = reg.sort_index()
    return reg
