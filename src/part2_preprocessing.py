'''
PART 2: Pre-processing
- Take the time to understand the data before proceeding
- Load `pred_universe_raw.csv` into a dataframe and `arrest_events_raw.csv` into a dataframe
- Perform a full outer join/merge on 'person_id' into a new dataframe called `df_arrests`
- Create a column in `df_arrests` called `y` which equals 1 if the person was arrested for a felony crime in the 365 days after their arrest date in `df_arrests`. 
- - So if a person was arrested on 2016-09-11, you would check to see if there was a felony arrest for that person between 2016-09-12 and 2017-09-11.
- - Use a print statment to print this question and its answer: What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?
- Create a predictive feature for `df_arrests` that is called `current_charge_felony` which will equal one if the current arrest was for a felony charge, and 0 otherwise. 
- - Use a print statment to print this question and its answer: What share of current charges are felonies?
- Create a predictive feature for `df_arrests` that is called `num_fel_arrests_last_year` which is the total number arrests in the one year prior to the current charge. 
- - So if someone was arrested on 2016-09-11, then you would check to see if there was a felony arrest for that person between 2015-09-11 and 2016-09-10.
- - Use a print statment to print this question and its answer: What is the average number of felony arrests in the last year?
- Print the mean of 'num_fel_arrests_last_year' -> pred_universe['num_fel_arrests_last_year'].mean()
- Print pred_universe.head()
- Return `df_arrests` for use in main.py for PART 3; if you can't figure this out, save as a .csv in `data/` and read into PART 3 in main.py
'''

import pandas as pd

def preprocess_data():
    # load the datasets
    pred_universe_raw = pd.read_csv('./data/pred_universe_raw.csv')
    arrest_events_raw = pd.read_csv('./data/arrest_events_raw.csv')

    # converts date variables to datetime format
    arrest_events_raw['arrest_date_event'] = pd.to_datetime(arrest_events_raw['arrest_date_event'])
    pred_universe_raw['arrest_date_univ'] = pd.to_datetime(pred_universe_raw['arrest_date_univ'])

    # perform a full outer join on 'person_id'
    df_arrests = pd.merge(pred_universe_raw, arrest_events_raw, on='person_id', how='outer', suffixes=('_univ', '_event'))

    # creates the 'y' column for rearrest for felony within a year
    def rearrest_felony(row):
        if pd.isna(row['arrest_date_event']) or pd.isna(row['arrest_date_univ']):
            return 0  # no rearrest within 365 days there are no dates

        # defines time window (365 days/year after the current arrest date)
        start_date = row['arrest_date_univ'] + pd.Timedelta(days=1)
        end_date = row['arrest_date_univ'] + pd.Timedelta(days=365)

        # check for felony rearrest within the year time period
        felony_rearrests = arrest_events_raw[
            (arrest_events_raw['person_id'] == row['person_id']) & 
            (arrest_events_raw['arrest_date_event'] >= start_date) & 
            (arrest_events_raw['arrest_date_event'] <= end_date) & 
            (arrest_events_raw['charge_degree'] == 'felony')
        ]
        if (len(felony_rearrests) > 0):
            return 1
        else:
            return 0

    # applies the rearrest_felony function across the dataframe
    df_arrests['y'] = df_arrests.apply(rearrest_felony, axis=1)

    # prints share of arrestees who were rearrested for a felony in the next year
    rearrest_share = df_arrests['y'].mean()
    print("What share of arrestees were rearrested for a felony crime in the next year? " , rearrest_share)

    # creates the 'current_charge_felony' column (1 if felony charge, 0 otherwise)
    df_arrests['current_charge_felony'] = df_arrests['charge_degree'].apply(lambda x: 1 if x == 'felony' else 0)

    # prints the share of current charges that are felonies
    felony_share = df_arrests['current_charge_felony'].mean()
    print("What share of current charges are felonies?", felony_share)

    # creates 'num_fel_arrests_last_year' column
    def felony_arrests_last_year(row):
        if pd.isna(row['arrest_date_event']):
            return 0  # no arrest date exists

        # defines time window (year before the current arrest date)
        start_date = row['arrest_date_event'] - pd.Timedelta(days=365)
        end_date = row['arrest_date_event'] - pd.Timedelta(days=1)

        # counts felony arrests within this time window
        felony_arrests = arrest_events_raw[
            (arrest_events_raw['person_id'] == row['person_id']) & 
            (arrest_events_raw['arrest_date_event'] >= start_date) & 
            (arrest_events_raw['arrest_date_event'] <= end_date) & 
            (arrest_events_raw['charge_degree'] == 'felony')
        ]

        return len(felony_arrests)

    df_arrests['num_fel_arrests_last_year'] = df_arrests.apply(felony_arrests_last_year, axis=1)

    # print the average number of felony arrests in the last year
    avg_felony_arrests = df_arrests['num_fel_arrests_last_year'].mean()
    print("What is the average number of felony arrests in the last year? " , avg_felony_arrests)

    # print the mean of 'num_fel_arrests_last_year'
    print("Mean of 'num_fel_arrests_last_year': ", df_arrests['num_fel_arrests_last_year'].mean())

    # print the first few rows of the dataframe
    print(df_arrests.head())

    # saves resulting dataframe as CSV in data folder
    df_arrests.to_csv('./data/df_arrests.csv', index=False)

    # returns the df_arrests dataframe for use in main.py
    return df_arrests







