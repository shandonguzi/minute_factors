import pandas as pd

def generate_monthly_date_pairs_corrected(start_date, end_date):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    start_of_months = pd.date_range(start=start_date.replace(day=1), end=end_date, freq='MS')
    date_pairs = []
    for date in start_of_months:
        month_start = date.strftime('%Y-%m-%d')
        month_end = date + pd.offsets.MonthEnd(1)
        if date.month == start_date.month and date.year == start_date.year:
            month_start = start_date.strftime('%Y-%m-%d')
        if date.month == end_date.month and date.year == end_date.year:
            month_end = end_date
        date_pairs.append((month_start, month_end.strftime('%Y-%m-%d')))

    return date_pairs