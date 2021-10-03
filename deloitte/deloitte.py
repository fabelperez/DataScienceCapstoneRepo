#!/home/fperez/miniconda3/envs/deloitte/bin/python
"""
This module contains the scripts and calls used to examine the 
Pike's Peek 10K datasets for the Deloitte Data Exercise.
"""

import string
import re
import pandas as pd
import numpy as np
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
        raw_data = pd.read_csv(
                'data/raw/{}'.format(self.file_path),\
                sep='\t')
        
        # Renaming column names
        raw_data.columns = map(str.lower, raw_data.columns)
        raw_data.rename(columns=\
                {'div/tot':'div_total', 'ag':'age',\
                'gun tim':'gun_time', 'net tim':'net_time'},\
                inplace=True\
                )
        clean_cols = raw_data.columns.tolist()
        clean_cols.pop(3) # Removes `name` column
        clean_cols.pop(4) # Removes `hometown` column

        for col in clean_cols:
            raw_data[col].replace(\
                    to_replace='[#*^a-zA-Z ]',\
                    value='',\
                    regex=True, 
                    inplace=True)
        raw_data['hometown'].replace(to_replace='[,.]', value='', regex=True, inplace=True)

        return raw_data
