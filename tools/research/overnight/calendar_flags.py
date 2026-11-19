"""Calendar and event flags for each NIGHT (keyed by the morning's trading date t: the hold is close t-1 -> open t).

Official release calendars (bls.gov, federalreserve.gov) were NOT reachable from this session, so:
- Jobs report (Employment Situation, 8:30 ET): BLS rule "third Friday after the reference week (the week that
  contains the 12th)", with January releases on Jan 1-3 moved a week later, July 3/4 Friday holidays moved to
  Thursday, and known shutdown delays (Oct 2013, Oct-Dec 2025). APPROXIMATE.
- FOMC statement days: scheduled meetings 2000-2026 typed from memory (not verified online). APPROXIMATE.
- CPI release days: 2018-2025 typed from memory (not verified online). APPROXIMATE, short sample.
A night is flagged for an event if the event's morning falls inside the hold (first trading day >= event date).
"""
import datetime as dt
import numpy as np, pandas as pd

FOMC = """
2000-02-02 2000-03-21 2000-05-16 2000-06-28 2000-08-22 2000-10-03 2000-11-15 2000-12-19
2001-01-31 2001-03-20 2001-05-15 2001-06-27 2001-08-21 2001-10-02 2001-11-06 2001-12-11
2002-01-30 2002-03-19 2002-05-07 2002-06-26 2002-08-13 2002-09-24 2002-11-06 2002-12-10
2003-01-29 2003-03-18 2003-05-06 2003-06-25 2003-08-12 2003-09-16 2003-10-28 2003-12-09
2004-01-28 2004-03-16 2004-05-04 2004-06-30 2004-08-10 2004-09-21 2004-11-10 2004-12-14
2005-02-02 2005-03-22 2005-05-03 2005-06-30 2005-08-09 2005-09-20 2005-11-01 2005-12-13
2006-01-31 2006-03-28 2006-05-10 2006-06-29 2006-08-08 2006-09-20 2006-10-25 2006-12-12
2007-01-31 2007-03-21 2007-05-09 2007-06-28 2007-08-07 2007-09-18 2007-10-31 2007-12-11
2008-01-30 2008-03-18 2008-04-30 2008-06-25 2008-08-05 2008-09-16 2008-10-29 2008-12-16
2009-01-28 2009-03-18 2009-04-29 2009-06-24 2009-08-12 2009-09-23 2009-11-04 2009-12-16
2010-01-27 2010-03-16 2010-04-28 2010-06-23 2010-08-10 2010-09-21 2010-11-03 2010-12-14
2011-01-26 2011-03-15 2011-04-27 2011-06-22 2011-08-09 2011-09-21 2011-11-02 2011-12-13
2012-01-25 2012-03-13 2012-04-25 2012-06-20 2012-08-01 2012-09-13 2012-10-24 2012-12-12
2013-01-30 2013-03-20 2013-05-01 2013-06-19 2013-07-31 2013-09-18 2013-10-30 2013-12-18
2014-01-29 2014-03-19 2014-04-30 2014-06-18 2014-07-30 2014-09-17 2014-10-29 2014-12-17
2015-01-28 2015-03-18 2015-04-29 2015-06-17 2015-07-29 2015-09-17 2015-10-28 2015-12-16
2016-01-27 2016-03-16 2016-04-27 2016-06-15 2016-07-27 2016-09-21 2016-11-02 2016-12-14
2017-02-01 2017-03-15 2017-05-03 2017-06-14 2017-07-26 2017-09-20 2017-11-01 2017-12-13
2018-01-31 2018-03-21 2018-05-02 2018-06-13 2018-08-01 2018-09-26 2018-11-08 2018-12-19
2019-01-30 2019-03-20 2019-05-01 2019-06-19 2019-07-31 2019-09-18 2019-10-30 2019-12-11
2020-01-29 2020-04-29 2020-06-10 2020-07-29 2020-09-16 2020-11-05 2020-12-16
2021-01-27 2021-03-17 2021-04-28 2021-06-16 2021-07-28 2021-09-22 2021-11-03 2021-12-15
2022-01-26 2022-03-16 2022-05-04 2022-06-15 2022-07-27 2022-09-21 2022-11-02 2022-12-14
2023-02-01 2023-03-22 2023-05-03 2023-06-14 2023-07-26 2023-09-20 2023-11-01 2023-12-13
2024-01-31 2024-03-20 2024-05-01 2024-06-12 2024-07-31 2024-09-18 2024-11-07 2024-12-18
2025-01-29 2025-03-19 2025-05-07 2025-06-18 2025-07-30 2025-09-17 2025-10-29 2025-12-10
2026-01-28 2026-03-18 2026-04-29 2026-06-17 2026-07-29 2026-09-16 2026-10-28 2026-12-09
"""

CPI = """
2018-01-12 2018-02-14 2018-03-13 2018-04-11 2018-05-10 2018-06-12 2018-07-12 2018-08-10 2018-09-13 2018-10-11 2018-11-14 2018-12-12
2019-01-11 2019-02-13 2019-03-12 2019-04-10 2019-05-10 2019-06-12 2019-07-11 2019-08-13 2019-09-12 2019-10-10 2019-11-13 2019-12-11
2020-01-14 2020-02-13 2020-03-11 2020-04-10 2020-05-12 2020-06-10 2020-07-14 2020-08-12 2020-09-11 2020-10-13 2020-11-12 2020-12-10
2021-01-13 2021-02-10 2021-03-10 2021-04-13 2021-05-12 2021-06-10 2021-07-13 2021-08-11 2021-09-14 2021-10-13 2021-11-10 2021-12-10
2022-01-12 2022-02-10 2022-03-10 2022-04-12 2022-05-11 2022-06-10 2022-07-13 2022-08-10 2022-09-13 2022-10-13 2022-11-10 2022-12-13
2023-01-12 2023-02-14 2023-03-14 2023-04-12 2023-05-10 2023-06-13 2023-07-12 2023-08-10 2023-09-13 2023-10-12 2023-11-14 2023-12-12
2024-01-11 2024-02-13 2024-03-12 2024-04-10 2024-05-15 2024-06-12 2024-07-11 2024-08-14 2024-09-11 2024-10-10 2024-11-13 2024-12-11
2025-01-15 2025-02-12 2025-03-12 2025-04-10 2025-05-13 2025-06-11 2025-07-15 2025-08-12 2025-09-11 2025-10-24 2025-12-18
"""


def _dates(block):
    return sorted(pd.Timestamp(x) for x in block.split())


def jobs_dates(y0=2000, y1=2026):
    out = []
    for y in range(y0, y1 + 1):
        for m in range(1, 13):
            d12 = dt.date(y, m, 12)
            sat = d12 + dt.timedelta(days=(5 - d12.weekday()) % 7)     # Saturday ending the reference week
            rel = sat + dt.timedelta(days=20)                           # third Friday after it
            if rel.month == 1 and rel.day <= 3:
                rel += dt.timedelta(days=7)
            if rel.month == 7 and rel.day in (3, 4):
                rel -= dt.timedelta(days=1)
            out.append(pd.Timestamp(rel))
    fix = {"2013-10-04": "2013-10-22", "2013-11-01": "2013-11-08", "2025-10-03": "2025-11-20",
           "2025-11-07": None, "2025-12-05": "2025-12-16"}
    res = []
    for d in out:
        k = d.strftime("%Y-%m-%d")
        if k in fix:
            if fix[k] is None:
                continue
            d = pd.Timestamp(fix[k])
        res.append(d)
    return sorted(res)


def first_trading_on_or_after(dates, idx):
    pos = idx.searchsorted(pd.DatetimeIndex(dates))
    pos = pos[pos < len(idx)]
    return idx[pos]


def flags(idx):
    """DataFrame of boolean flags per night (index = morning trading date)."""
    idx = pd.DatetimeIndex(idx)
    s = pd.Series(idx, index=idx)
    prev_gap = s.diff().dt.days.fillna(1)
    next_gap = (-s.diff(-1).dt.days).fillna(1)
    dow = idx.dayofweek
    F = pd.DataFrame(index=idx)
    F["mon"], F["tue"], F["wed"], F["thu"], F["fri"] = [dow == k for k in range(5)]
    F["weekend"] = (dow == 0) & (prev_gap >= 3)
    F["holiday_night"] = ((prev_gap >= 2) & ~((dow == 0) & (prev_gap == 3)))
    F["pre_holiday"] = (next_gap >= 2) & ~((dow == 4) & (next_gap == 3))
    ym = idx.to_period("M")
    first_k = s.groupby(ym).cumcount()
    last = s.groupby(ym).transform("max") == s
    F["tom"] = last.values | (first_k.values < 3)
    # monthly opex: third Friday (or the trading day before it if a holiday)
    opex = []
    for p in pd.period_range(idx[0].to_period("M"), idx[-1].to_period("M"), freq="M"):
        d1 = pd.Timestamp(p.start_time)
        fri = d1 + pd.Timedelta(days=(4 - d1.dayofweek) % 7 + 14)
        k = idx.searchsorted(fri, side="right") - 1
        if k >= 0 and idx[k].to_period("M") == p:
            opex.append(idx[k])
    F["opex_morning"] = idx.isin(opex)
    after = [idx[idx.searchsorted(d, side="right")] for d in opex if idx.searchsorted(d, side="right") < len(idx)]
    F["post_opex"] = idx.isin(after)
    F["jobs"] = idx.isin(first_trading_on_or_after(jobs_dates(), idx))
    F["fomc_day"] = idx.isin(first_trading_on_or_after(_dates(FOMC), idx))
    cpi = first_trading_on_or_after(_dates(CPI), idx)
    F["cpi"] = idx.isin(cpi)
    F["cpi_sample"] = (idx >= "2018-01-01") & (idx < "2026-01-01")
    return F
