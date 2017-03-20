# -*- coding: utf-8 -*-
"""
    utils
    ~~~~~

    Utility functions for motif analysis.
"""

import pandas as pd
import numpy as np
import math
from collections import Counter
from geopy.distance import vincenty
import geohash

from location import motif


def compute_gyration(data,
                     sr_col='stay_region',
                     k=None):
    """
    Compute the total or k-th radius of gyration.
    This follows the work of Pappalardo et al.
    (see http://www.nature.com/articles/ncomms9166)

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


    Returns:
    --------
    float
        Radius of gyration in meters.
        Return np.nan is k is greater than the number of different
        visited locations.
    """
    loc_data = data.copy()
    loc_data = pd.DataFrame(loc_data[sr_col].dropna())

    # get location data for corresponding k
    if k is not None:
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

    # compute coordinates for visited locations/stay regions
    locs_hist_coord = [geohash.decode(x) for x in loc_data[sr_col]]
    loc_data['latitude'] = [x[0] for x in locs_hist_coord]
    loc_data['longitude'] = [x[1] for x in locs_hist_coord]

    # compute mass of locations
    r_cm = motif.get_geo_center(loc_data, lat_c='latitude', lon_c='longitude')
    r_cm = (r_cm['latitude'], r_cm['longitude'])

    # compute gyration of radius
    temp_sum = 0
    for _, r in loc_data.iterrows():
        p = (r.latitude, r.longitude)
        d = vincenty(p, r_cm).m
        temp_sum += d ** 2

    return math.sqrt(temp_sum / len(loc_data))


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

    float
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
