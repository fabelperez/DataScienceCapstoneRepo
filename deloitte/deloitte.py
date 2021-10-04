#!/home/fperez/miniconda3/envs/deloitte/bin/python
"""
This module contains the scripts and calls used to examine the 
Pike's Peek 10K datasets for the Deloitte Data Exercise.
"""

import string
import re
import pandas as pd
import numpy as np
from datetime import datetime

def try_parsing_date(text):
    """
    Parsing out date and time values from different variations of time entries
    """
    for fmt in ('%H:%M:%S', '%M:%S', ':%S'):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError('no valid date format found')

def rreplace(s, old=' ', new=', ', occurrence=1):
    li = s.rsplit(old, occurrence)
    return new.join(li)

def division_parser(x):
    """
    Parsing out actual divisions number based on """
    if (pd.isnull(x) or x<0):
        return np.nan
    elif (x>0) and (x<=14):
        return 1
    elif (x>=15) and (x<=19):
        return 2
    else:
        return int(x/10)


us_state_to_abbrev = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
    "District of Columbia": "DC",
    "American Samoa": "AS",
    "Guam": "GU",
    "Northern Mariana Islands": "MP",
    "Puerto Rico": "PR",
    "United States Minor Outlying Islands": "UM",
    "U.S. Virgin Islands": "VI",
}

# invert the dictionary
abbrev_to_us_state = dict(map(reversed, us_state_to_abbrev.items()))


class Deloitte:
    """
    A class that contains all of the scripts used for the exercise
    """

    def __init__(self, file_path):
        self.file_path = file_path 
        self.f_df = None
        self.m_df = None
        self.all_df = None



    def clean_data(self):
        raw_data = pd.read_csv(\
                'data/raw/{}'.format(self.file_path),\
                encoding='latin-1', sep='\t')
        
        # renaming column names
        raw_data.columns = map(str.lower, raw_data.columns)
        raw_data.rename(columns=\
                {'div/tot':'div_total', 'ag':'age',\
                'gun tim':'gun_time', 'net tim':'net_time'},\
                inplace=True\
                )
        clean_cols = raw_data.columns.tolist()
        clean_cols.pop(3) # removes `name` column
        clean_cols.pop(4) # removes `hometown` column

        # cleaning special symbols from columns to normalize data
        for col in clean_cols:
            raw_data[col].replace(\
                    to_replace='[#*^a-zA-Z ]',\
                    value='',\
                    regex=True,\
                    inplace=True)
        raw_data['hometown'].replace(to_replace='[,.]', value='', regex=True, inplace=True)

        # Separating Hometown from the State
        raw_data['hometown'] = raw_data['hometown'].map(rreplace)
        raw_data[['city', 'state']] = raw_data.hometown.str.split(',', expand=True)
        raw_data['state'].replace(to_replace=' ', value='', regex=True, inplace=True)

        # Changing abbreviated state names to full names
        raw_data['state'] = raw_data['state'].map(abbrev_to_us_state)

        # Normalizing/fixing timed features
        time_cols = ['gun_time', 'net_time', 'pace']
        for col in time_cols:
            # Applies function to all rows
            raw_data[col] = raw_data[col].map(try_parsing_date)
            # Removes the default date
            raw_data[col] = raw_data[col] -datetime(1900, 1, 1)
            # Finding total time in seconds
            raw_data[col] = raw_data[col].dt.total_seconds()


        return raw_data
