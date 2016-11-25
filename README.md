![Travis](https://travis-ci.org/saeed-abdullah/location-analysis.svg?branch=master)

## Utility functions for analyzing location data ##

This repository contains a collection for analyzing patterns and trends
from gps data collected from smartphones.

In particular, we focus on generating `motif` from location data. For more details,
see:

* [A review of urban computing for mobile phone traces: current methods, challenges and opportunities][1]
* [Unravelling daily human mobility motifs][2]

### Node generation ###

To generate stay nodes, you can run `location.motif` as a script:

`python3 location.motif -g -f [CSV] -c [CONFIG]`

where `-f` points to a csv data file and `-c` points to a JSON config file.

#### Config file ####

You can use a JSON config file to provide the arguments for `location.motif.compute_nodes`
function. Here is an example of a config file:

```javascript
{
  "lat_c": "latitude",
  "lon_c": "longitude",
  "stay_region_args": {
    "precision": 7
  },
  "node_args": {
    "time_interval": "30Min"
  },
  "stay_point_args": {
    "dist_th": 300,
    "time_th": "30m"
  },
  "daily_args": {
    "shift_day_start": "3.5H",
    "rare_pt_pct_th": null
  }
}
```


[1]: http://dl.acm.org/citation.cfm?doid=2505821.2505828
[2]: http://rsif.royalsocietypublishing.org/content/10/84/20130246/


