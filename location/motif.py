# -*- coding: utf-8 -*-
"""
    motif
    ~~~~~

    Module for generating and analyzing motifs.

    :copyright: (c) 2016 by Saeed Abdullah.
"""

import argparse
from collections import Counter
import json
import math

from copy import deepcopy
from collections import Counter
from geopy.distance import vincenty
import networkx as nx

# install anvil from https://github.com/saeed-abdullah/Anvil
from anvil.utils import get_df_slices
from anvil.api import convert_time_zone

from geopy import distance, point
import geohash
import pandas as pd
import numpy as np

from location import utils


def compute_geo_hash(df, lat_c='lat',
                     lon_c='lon', precision=12):
    """
    Computes geohash values.

    Parameters
    ----------
    df : DataFrame
    lat_c : str
        Name of column containing latitude values.
        Default is `lat`.
    lon_c : str
        Name of column containing longitude values.
        Default is `lon`.
    precision : int
        Precisoin of generated geohash function.
        Take a look at geohash length and precision
        here: https://en.wikipedia.org/wiki/Geohash

    Returns
    -------
    l : iterables
        List of geohash values corresponding to the
        rows of values in the given dataframe
    """

    # get geohash data
    l = []
    for _, row in df.iterrows():
        g = geohash.encode(latitude=row[lat_c],
                           longitude=row[lon_c],
                           precision=precision)
        l.append(g)

    return l


def trim_geo_hash_precision(hashed_values, precision=9):
    """
    Trims geo hash precision.

    Parameters
    ----------
    hashed_values : Series
        Contains geohashed values as strings.

    precision : int
        The desired precision. If the current
        precision is smaller, then nothing is done.
    """

    return hashed_values.map(lambda z: z[:precision])


def filter_out_rare_points(points, threshold_pct=0.5):
    """
    Filters out rare points.

    All points with occurrences <= threshold_pct is
    considerd as rare points. All rare points are
    replaced by pd.NaN

    Parameters
    ----------
    points : iterables
        Instances of points

    threshold_pct : float
        Threshold in percentage for rare points. Any
        point occuring less than given threshold is
        considerd as a rare point. Default is 0.5%

    Returns
    -------
    l : list
        List where rare points are marked as pd.NaN.
    """

    c = Counter(points)
    total = sum(c.values())
    l = []
    for p in points:
        v = c[p]
        if v/total * 100 <= threshold_pct:
            l.append(np.nan)
        else:
            l.append(p)

    return l


def get_primary_location(locations, aggr_f='count'):
    """
    Gets the primary location.

    Within any given duration there might be a number
    of locations visited by user. This function
    identifies the primary location from a given list
    of locations.

    How to define a primary location? In this case,
    we use duration of time spent by a user within
    a given duration.

    Parameters
    ----------
    locations : Series
        Series with geo hashed values.

    aggr_f : str
        Aggregating function. Default is 'count' which
        will result in counting number of rows. If the
        sampling rate is somewhat consistent then just
        counting the number of times a place has been
        recorded is a good approximation of the time
        spent by a user.


    Parameters
    ----------
    location : str
        Returns the primary location.
    """

    if aggr_f != 'count':
        err = 'Aggregate function {0} is not supported'
        raise ValueError(err.format(aggr_f))

    # sorted by size of each group
    g = locations.groupby(locations).size().sort_values(ascending=False)
    return g.index[0]  # most visited place


def generate_daily_nodes(df, hash_c='geo_hash',
                         geo_hash_preicion=None,
                         shift_day_start=None,
                         rare_pt_pct_th=0.5,
                         valid_day_th=8,
                         start_date=None,
                         end_date=None,
                         node_args=None):
    """
    Parameters
    ----------
    df : DataFrame
        DataFrame with sorted DateTimeIndex.

    hash_c : str
        Columns containing geo hash values.

    geo_hash_precision : int
        Desired precision of geo hashed values. See
        `trim_geo_hash_precision` for details. If `None`,
        no trimming will happen. Default is None.

    shift_day_start : str
        The duration by which the start of the day should
        be shifted. For example, in the original paper,
        the day starts at 3:30AM. Provided value should
        be in pd.tslib.TimeDelta (e.g., for 3:30AM start
        of the day, it should be '3.5H'). Default is None.

    rare_pt_pct_th : float
        Threshold for rare points in percentage. See
        `filter_out_rare_points` for more details. Default
        is 0.5%. If `None`, no filtering happens.

    valid_day_th : int
        Minimum number of intervals for a valid day. If
        a day has < valid_day_th intervals, it will be
        discarded. Default is 8.

    start_date : TimeStamp
        Start date to generate nodes. If None, the minimum
        day in the DateTimeIndex will be used.

    end_date : TimeStamp
        End date to generate nodes. If None, 1 + maximum day
        in the DateTimeIndex will be used.

    node_args : dict
        Arguments to pass to `generate_nodes` (e.g., time_interval).
        Default is None.

    Returns
    -------
    l : list
        A list containing (date, nodes) pairs. Where nodes
        are represented as a DataFrame returned by `generate_nodes`.


    Notes
    -----
        The start date and end date is used pd.date_range to
        generate list of days. So, it is important to have
        same timezone information for start_date and end_date
        as in the given DateTimeIndex.

    """
    df = df.copy().loc[:, [hash_c]]

    l = []

    if start_date is None:
        d = df.index.min()
        tz = d.tz  # timezone information

        start_date = pd.to_datetime(d.date()).tz_localize(tz)

    if end_date is None:
        d = df.index.max()
        tz = d.tz  # timezone information

        # maximum date + 1
        end_date = pd.to_datetime(d.date()).tz_localize(tz)
        end_date = end_date + pd.to_timedelta('1D')

    # shifting start of the day
    if shift_day_start is not None:
        shift_day_start = pd.to_timedelta(shift_day_start)
        start_date = start_date + shift_day_start
        end_date = end_date + shift_day_start

    if geo_hash_preicion is not None:
        df[hash_c] = trim_geo_hash_precision(df[hash_c], geo_hash_preicion)

    # remove rare points
    if rare_pt_pct_th is not None:
        df[hash_c] = filter_out_rare_points(df[hash_c],
                                            rare_pt_pct_th)

    # remove NA values (potentially resulting from removing rare points)
    df = df.dropna(subset=[hash_c])

    if node_args is None:
        node_args = {}

    days = pd.date_range(start=start_date, end=end_date, freq='1D')
    for index, rows in enumerate(get_df_slices(df, days)):
        d = days[index]

        nodes = generate_nodes(rows[hash_c], start_time=d, **node_args)

        if len(nodes) < valid_day_th:
            l.append((d, np.nan))
        else:
            l.append((d, nodes))

    return l


def generate_nodes(locations,
                   start_time,
                   end_time=None,
                   time_interval='30Min',
                   valid_interval_th=1):
    """
    Generates motif information from location data.

    This function follows the work of Schneider et al.
    (see http://rsif.royalsocietypublishing.org/content/10/84/20130246/)

    Parameters
    ----------

    locations : Series
        Series with sorted DateTimeIndex and geo hashed values
        over a given day.

    start_time : pandas.tslib.Timestamp
        Start time to generate time intervals.

    end_time : pandas.tslib.Timestamp
        End time for generating time invervals. If None,
        it is set to 24 hours after start_time. Default
        is None.

    time_interval : str
        The interval duration. The default is 30 mins
        (resulting in 48 intervals per day). The argument
        is passed to `pd.date_range` as the frequency string.

    valid_interval_th : int
        Minimum number of records for a valid interval.
        If an interval has < valid_interval_th rows, it will
        not be considered. Default is 1.

    Returns
    -------
    DataFrame
        It contains two columns: 'node' and 'time'. The node
        column contains the primary locations and the time
        column contains the start timestamp of each interval.
        If the interval is not valid (e.g., not having
        sufficient records), it will contain np.nan as value.
    """

    if end_time is None:
        end_time = start_time + pd.to_timedelta('1D')

    intervals = pd.date_range(start=start_time,
                              end=end_time, freq=time_interval)
    s = []

    for index, t in enumerate(get_df_slices(locations, intervals)):

        if len(t) < valid_interval_th:
            s.append({'node': np.nan, 'time': intervals[index]})
        else:
            s.append({'node': get_primary_location(t),
                      'time': intervals[index]})

    return pd.DataFrame(s)


def generate_graph(nodes):
    """
    Generate graphs from a given series of nodes.

    Parameters
    ----------
    nodes : Series
        An iterable of nodes. It can contain NaN values


    Returns
    -------
    list
        A list of strings where the edges are seperated by
        whitespace (e.g., ["a b", "b c"]). It can be parsed
        by networkx.parse_edgelist.


    ToDos
    -----
        1. Only considering non-consecutive nodes (e.g.,
        what to do if seperated by NaN).
    """

    nodes = nodes.dropna()
    l = []
    edge_format = "{0} {1}"

    for x1, x2 in zip(nodes.shift(), nodes):
        if not(pd.isnull(x1) or pd.isnull(x2)):
            if x1 != x2:
                l.append(edge_format.format(x1, x2))

    return l


def get_geo_center(df, lat_c='latitude', lon_c='longitude'):
    """
    Calculates center of given geo points.

    To avoid "wrap-around" issues (near international dateline),
    this method follows [1].

    Parameters
    ----------
    df: DataFrame

    lat_c: str
        Column name containing latitudes. Default is latitude.
    lon_c: str
        Column name containing longitudes. Default is longitude

    Returns
    -------
    d: dict
        Keys 'latitude' and 'longitude' indicate
        the coordinates of the center point.

    [1]: https://carto.com/blog/center-of-points/
    """

    lat = df[lat_c]
    lon = df[lon_c]

    angle = math.pi / 180

    center_lat = math.atan2(np.mean(np.sin(lat * angle)),
                            np.mean(np.cos(lat * angle))) * 180 / math.pi

    center_lon = math.atan2(np.mean(np.sin(lon * angle)),
                            np.mean(np.cos(lon * angle))) * 180 / math.pi

    return {'longitude': center_lon, 'latitude': center_lat}


def get_stay_point(df, lat_c='latitude',
                   lon_c='longitude', dist_th=300,
                   time_th='30m'):
    """
    Calculates stay points.

    Stay points are determined following [1]. That is,
    if a user stays within a given radius (e.g., dist_th)
    for a given period of time (e.g., time_th), then the
    center of these points is a stay point.

    In this implementation, given a set of points p_k (i <= k <= j),
    where these points are sorted ascendingly in terms of occurence
    (i.e., t_k <= t_(k+1)), we take a greedy approach to
    determine if these points result in a stay point. That is, these
    points are considered for stay points if:

        1. distance between first (p_i) and last point (p_j) is less
    than dist_th, and
        2. the time between the first and last point is larger than the
    threshold (e.g., t_j - t_i >= time_th).


    Note that this is a greedy algorithm as we implicitly consider
    the first point as the center of a cluster (with radius of dist_th)
    which might not be the optimal choice (similar case for time duration
    calculation). However, these issues should get resolved in the
    subsequent phase (i.e., by computing stay regions).

    Parameters
    ----------
    df: DataFrame
        DataFrame with sorted (ascending) DateTimeIndex

    lat_c: str
        Column name with latitude values

    lon_c: str
        Column name with longitude values

    dist_th: float
        Distance threshold in meters. Default is 300m.

    time_th: str or pd.timedelta
        Time threshold that will be parsed by pd.to_timedelta.
        Default is 30 minutes.


    Returns
    -------
    stay_points : list
        A list corresponding to the rows of df. Each item
        represents the sray point ids (so points within same
        stay points will have same id). If a point is a travel point
        (i.e., does not belog to any stay points), then it will
        have np.NaN.


    [1]: https://dl.acm.org/citation.cfm?id=1463477
    """

    index = 0
    stay_points_c = 0  # total stay points count
    stay_points = []

    max_len = len(df)
    time_th = pd.to_timedelta(time_th)

    while index < max_len:
        mem_c = 1  # current stay point members: just index

        # time diff between current and first members
        time_diff = pd.to_timedelta('0s')

        p_f = point.Point(latitude=df.iloc[index][lat_c],
                          longitude=df.iloc[index][lon_c])

        time_f = df.index[index]

        j = index + 1
        while j < max_len:
            p_s = point.Point(latitude=df.iloc[j][lat_c],
                              longitude=df.iloc[j][lon_c])

            d = distance.GreatCircleDistance(p_f, p_s).m
            if d <= dist_th:
                mem_c += 1  # new member
                time_diff = df.index[j] - time_f  # update total time spent
                j += 1
            else:
                # spatial constrain is not met
                break

        # Check if previous points met the time threshold constraint
        if time_diff >= time_th:
            # All these points share same stay point id
            stay_points.extend([stay_points_c] * mem_c)
            stay_points_c += 1
        else:
            # these are not valid stay points
            stay_points.extend([np.nan] * mem_c)

        # points up to j has been considered
        index = j

    return stay_points


def merge_neighboring_grid(geo_hash):
    """
    Merges neighboring grids using a greedy approach.

    This function sorts each grid by the number by
    frequency. A grid G is merged with K, if:
        1. G and K is neighboring grid (note that
    the definition of neighboring grid depends on
    the precision of geohash), and
        2. K has higher frequency than G.

    The resulting merged grid retains the geohash
    of grid with higher frequency (e.g., in previous
    example, the merged grid will have the same geohash
    of K).

    Note that this is a greedy approach, so there is
    no gurantee that this would result in optimal clustering.
    For example, there might be cases in which a more
    cascaded merging would be better (e.g., I -> G -> K, but
    in our approach I and K might not be merged). However, this
    serves to constraint the maximum size of the merged grid
    (e.g., 9 grids) at a time, which has been shown to be a
    better strategy by Zheng et al. (2010):
    http://portal.acm.org/citation.cfm?doid=1772690.1772795

    Parameters
    ----------
    geo_hash : Series
        A series with geo hashed values.


    Returns
    -------
    Series
        A new series with updated geo hash values
        after merging neighboring grids.
    """

    c = Counter(geo_hash.dropna())
    d = {}

    # sort by frequency
    for z, _ in c.most_common():
        # this check is necessary as we remove
        # items dynamically
        if z in c:
            d[z] = z

            # go through the potential merge options
            for n in geohash.neighbors(z):
                if n in c:
                    d[n] = z
                    del c[n]  # merged with grid z

    return geo_hash.map(d)


def get_stay_region(df, stay_point_c='stay_point',
                    lat_c='latitude', lon_c='longitude',
                    precision=7):

    """
    Calculates stay regions.

    There are three steps:

    1. First compute the center of stay points.
    2. Compute the geohash of stay point centers
    3. Merge neighboring grids.

    Parameters
    ----------

    df : DataFrame

    stay_point_c : str
        Column name with stay point ids. Default
        is `stay_point_id`.

    lat_c : str
        Column name with latitude values. Default
        is `latitude`.

    lon_c : str
        Column name for longitude values. Default
        is `longitude`.

    precision : int
        Geo hash precision. Default is 7.

    Returns
    -------
    Series
        Series with stay regions where a stay region
        is defined by a geohash value.
    """

    centers = {}
    for k, v in df.groupby(stay_point_c):
        # get stay point centers
        center = get_geo_center(v, lat_c=lat_c, lon_c=lon_c)

        # now convert to geo-hash grid
        h = geohash.encode(latitude=center['latitude'],
                           longitude=center['longitude'],
                           precision=precision)
        centers[k] = h

    # associate same stay point centers
    # to each record
    stay_points = df[stay_point_c].map(centers)

    # now convert the stay points to stay regions
    return merge_neighboring_grid(stay_points)


def _save_nodes(nodes, path):
    """
    Saves nodes from `generate_daily_nodes` as csv file.

    Parameters
    ----------
    nodes : iterables
        Nodes from `generate_daily_nodes`. It is a list of
        tuples (Timestamp, DataFrame).

    path : str
        Output path. It should have a .csv extension.

    Notes
    -----
        The data is saved as a DataFrame with following
        columns: 'timestamp', 'tz', 'node', 'time'. The
        first two columns correspond to the first element
        in the tuple. The 'node' and 'time' columns are
        retained as returned from `generate_daily_nodes`.
        All the timestamps are converted to UTC timezone.

        To retrieve the nodes, you should perform a group-by
        function on 'timestamp'. See `_load_nodes` for
        more details.
    """

    df = None
    for node in nodes:
        timestamp = node[0]
        tz = node[0].tz

        d = node[1].copy()
        d['timestamp'] = timestamp.tz_convert('UTC')
        d['tz'] = tz

        d.time = d.time.map(lambda z: z.tz_convert('UTC'))

        if df is None:
            df = d
        else:
            df = pd.concat([df, d])

    df.to_csv(path)


def _load_nodes(path, convert_tz=True, target_tz=None):
    """
    Load nodes from a given csv file.

    For data format, see `_save_nodes`.

    Parameters
    ----------
    path : str
        Input file path.

    convert_tz : bool
        If the 'time' column should be converted to
        timestamp using pd.to_datetime. Default is
        True — the 'time' column will be converted.

    target_tz : string, pytz.timezone, dateutil.tz.tzfile or None
        Timezone information to be used when converting
        'time column. Default is None, in that case the timezone
        information from the corresponding timestamp column
        will be used.

    Returns
    ------
    l : list
        A list containing tuples of (Timestamp, DataFrame)
        similar to the return value of `generate_daily_nodes`.
    """

    l = []
    df = pd.read_csv(path)

    for t, v in df.groupby('timestamp'):
        tz = v.loc[0, 'tz']
        timestamp = pd.Timestamp(t).tz_convert(tz)
        # skip tz and timestamp columns
        nodes = v.loc[:, ['time', 'node']].copy()

        # converting to datetime
        if convert_tz:
            nodes.time = pd.to_datetime(nodes.time)

            # convert to utc
            nodes.time = nodes.time.map(lambda z: z.tz_localize('utc'))

            # timezone information
            if target_tz is None:
                tz_c = tz
            else:
                tz_c = target_tz

            # convert to target timezone
            nodes.time = nodes.time.map(lambda z: z.tz_convert(tz_c))

        l.append((timestamp, nodes))

    return l


def compute_nodes(df,
                  lon_c='longitude',
                  lat_c='latitude',
                  stay_point_args=None,
                  stay_region_args=None,
                  node_args=None,
                  daily_args=None,
                  stay_info_output=None,
                  node_output=None):
    """
    Utility function for generating location motif

    Parameters
    ----------
    df : DataFrame
        DataFrame with sorted DateTimeIndex
    lon_c : str
        Column containing longitude values. Default is `longitude`.
    lat_c : str
        Column containing latitude values. Default is `latitude`.
    stay_point_args : dict
        Arguments to pass to `get_stay_point`. Default is `None`,
        default parameters will be used in that case.
    stay_region_args : dict
        Arguments to pass to `get_stay_region`. Default is `None`,
        default parameters will be used in that case.
    node_args : dict
        Arguments to pass to `generate_nodes`. Default is `None`,
        default parameters will be used in that case.
    daily_args : dict
        Arguments to pass to `generate_daily_nodes`. Default is `None`,
        default parameters will be used in that case.
    stay_into_output : Path
        The output path to which a dataframe with stay points and regions
        will be saved. See the Returns section for the format.
        Default is `None`, no output will be saved in that case.
    node_output : Path
        The output path to save generated daily nodes. Default is `None`,
        no output will be saved in that case.

    Returns
    -------
    (df, nodes) : (DataFrame, list)
        The data frame with stay points and regions. It has the same rows
        as the given parameters with lat_c, lon_c, 'stay_point', and
        'stay_region' columns. The second element of the returned tuple
        contains a list of daily nodes. See `generate_daily_nodes` for
        more details.
    """

    if stay_point_args is None:
        stay_point_args = {}
    if stay_region_args is None:
        stay_region_args = {}
    if node_args is None:
        node_args = {}
    if daily_args is None:
        daily_args = {}

    df = df.loc[:, [lon_c, lat_c]].copy()
    df['stay_point'] = get_stay_point(df,
                                      lon_c=lon_c,
                                      lat_c=lat_c,
                                      **stay_point_args)
    df['stay_region'] = get_stay_region(df,
                                        lon_c=lon_c,
                                        lat_c=lat_c,
                                        **stay_region_args)

    nodes = generate_daily_nodes(df.dropna(subset=['stay_region']),
                                 hash_c='stay_region',
                                 node_args=node_args,
                                 **daily_args)

    if stay_info_output is not None:
        df.to_csv(stay_info_output)

    if node_output is not None:
        _save_nodes(nodes, node_output)

    return df, nodes


def main():
    """
    Handles command line options.
    """

    parser = argparse.ArgumentParser()

    # command
    parser.add_argument('-g', '--generate', required=True, choices=['node'],
                        help="Generate motif or node")
    parser.add_argument('-f', '--file', help='File path')
    parser.add_argument('-c', '--config', help='JSON config file path')
    parser.add_argument('-tz', '--timezone', default='America/New_York',
                        help='Target timezone (default: America/New_York)')
    parser.add_argument('-tc', '--timecolumn', default='time',
                        help='Column with DateTime info (default: time)')

    args = parser.parse_args()

    if args.generate == 'node':
        df = pd.read_csv(args.file)
        df = convert_time_zone(df, args.timecolumn, to_timezone=args.timezone)

        with open(args.config) as f:
            params = json.load(f)

        compute_nodes(df, **params)

if __name__ == '__main__':
    main()
