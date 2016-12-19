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
