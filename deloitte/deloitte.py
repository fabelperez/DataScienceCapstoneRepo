#!/home/fperez/miniconda3/envs/deloitte/bin/python
"""
This module contains the scripts and calls used to examine the 
Pike's Peek 10K datasets for the Deloitte Data Exercise.
"""

import string
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
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
    def __init__(self):
        self.file_path = None 
        self.f_df = None
        self.m_df = None
        self.all_df = None

    def clean_data(self):
        raw_data = pd.read_csv('data/raw/{}'.format(self.file_path),encoding='latin-1', sep='\t')
        
        # renaming column names
        raw_data.columns = map(str.lower, raw_data.columns)
        raw_data.rename(columns={'div/tot':'div_total', 'ag':'age','gun tim':'gun_time', 'net tim':'net_time'},\
                inplace=True)
        clean_cols = raw_data.columns.tolist()
        clean_cols.pop(3) # removes `name` column
        clean_cols.pop(4) # removes `hometown` column

        # cleaning special symbols from columns to normalize data
        for col in clean_cols:
            raw_data[col].replace(to_replace='[#*^a-zA-Z ]',value='',regex=True, inplace=True)
        raw_data['hometown'].replace(to_replace='[,.]', value='', regex=True, inplace=True)

        # Separating Hometown from the State
        raw_data['hometown'] = raw_data['hometown'].map(rreplace)
        raw_data[['city', 'state']] = raw_data.hometown.str.split(',', expand=True)
        raw_data['state'].replace(to_replace=' ', value='', regex=True, inplace=True)

        # Changing abbreviated state names to full names
        raw_data['state'] = raw_data['state'].map(abbrev_to_us_state)

        # Adding in missing values
        missing_states={'Ellicott City':'Maryland','Fredericksburg':'Virginia','North Potomac':'Maryland',\
                'Silver Spring':'Maryland','Washington':'District of Columbia'}
        subset = raw_data.loc[raw_data['city'].isin(missing_states.keys()),'city']
        raw_data.loc[subset.index,'state']=raw_data.loc[subset.index,'city'].map(missing_states)

        # Normalizing/fixing timed features
        time_cols = ['gun_time', 'net_time', 'pace']
        for col in time_cols:
            # Applies function to all rows
            raw_data[col] = raw_data[col].map(try_parsing_date)
            # Removes the default date
            raw_data[col] = raw_data[col] -datetime(1900, 1, 1)
            # Finding total time in seconds
            raw_data[col] = raw_data[col].dt.total_seconds()
        
        raw_data['diff_time'] = raw_data['gun_time']-raw_data['net_time']
        raw_data['division_new'] = raw_data['age'].map(division_parser)

        # Adding Gender Column
        gender = self.file_path.split('.')[0].split('_')[-1].lower()
        raw_data['gender'] = gender

        # Normalizing string values (lower)
        cols_lower = ['name', 'hometown','city','state']
        for col in cols_lower:
            raw_data[col] = raw_data[col].str.lower()
        if gender=='females':
            self.f_df = raw_data
        elif gender=='males':
            self.m_df = raw_data
        return raw_data


    def combine(self):
        if all([self.f_df.empty==False, self.m_df.empty==False]):
            self.all_df = pd.concat(\
                    [self.f_df, self.m_df],\
                    axis=0).reset_index(drop=True)
        else:
            print("Please load both Female and Male Data Sets")


    def vis_q1(self):
        # Visualizing Violin Boxplot (All three: Females, Males, Combined)
        figure(figsize=(10,10), dpi=80)
        colors = ['red','blue','green'] 
        net_array = np.array([self.f_df['net_time'], self.m_df['net_time'], self.all_df['net_time']])
        vp = plt.violinplot(net_array, vert=False, showextrema=True, showmeans=True)
        med1,med2,med3 = np.quantile(net_array[0],.50), np.quantile(net_array[1],.50), np.quantile(net_array[2],.50)
        plt.scatter([med1,med2,med3], [1,2,3], marker='o', color='white', s=50, zorder=3) 
        q11,q21,q31 = np.quantile(net_array[0],.25), np.quantile(net_array[1],.25), np.quantile(net_array[2],.25)
        q13,q23,q33 = np.quantile(net_array[0],.75), np.quantile(net_array[1],.75), np.quantile(net_array[2],.75)

        whiskers_min = np.array([q11,q21,q31]) - (np.array([q13,q23,q33]) - np.array([q11,q21,q31])) * 1.5
        whiskers_max = np.array([q13,q23,q33]) + (np.array([q13,q23,q33]) - np.array([q11,q21,q31])) * 1.5

        plt.hlines([1,2,3],[q11,q21,q31], [q13,q23,q33], color='k', linestyle='-', lw=10)
        plt.hlines([1,2,3], whiskers_min, whiskers_max, color='aqua', linestyle='dashed', lw=3)
        plt.yticks([1,2,3],['Females','Males','Combined'])

        for i in range(len(vp['bodies'])):
            vp['bodies'][i].set(facecolor=colors[i])
            vp['bodies'][i].set(edgecolor='black')
            vp['bodies'][i].set(alpha=0.2)
            
        plt.title('Density Plot of `net_times` by Gender and Combined', size=17)
        plt.grid()
        plt.xlabel('`net_time` in seconds (s)', size=15)
        plt.ylabel('Participants/Racers', size=15)
        plt.show()

        # Visualizing Scatterplot & Stats for Females
        figure(figsize=(10,10), dpi=80)
        subset = self.f_df[self.f_df.division_new.notnull()]
        subset = subset[subset.division_new>0]
        colors = subset['division_new']
        scatter = plt.scatter(subset['age'], subset['net_time'], c=colors, alpha=0.35, label=set(colors))
        mu,mi,med,mx = subset['net_time'].describe().loc[['mean','min','50%','max']]
        
        mod = subset['net_time'].mode()
        meanli = plt.axhline(y=mu,color='red', linewidth=3, linestyle='-', label='Mean: {0:.3f} Sec'.format(mu))
        medli = plt.axhline(y=med,color='k', linewidth=4, linestyle='dotted', label='Median: {0:.3f} Sec'.format(med))
        
        modli=[]
        for i in range(len(mod)):
            modli.append(plt.axhline(y=mod[i],color='orange', linewidth=3, 
                            linestyle='--', label='Mode: {0:.3f} Sec'.format(mod[i])))
        
        minli = plt.axhline(y=mi,color='aqua', linewidth=3, 
                            linestyle='--', label='Minimum: {0:.3f} Sec'.format(mi))
        maxli = plt.axhline(y=mx,color='purple', linewidth=3, 
                            linestyle='--', label='Max: {0:.3f} Sec'.format(mx))
        
        extra = plt.axhline(y=mi,linewidth=0, label='Range: {0:.3f}'.format(mx-mi))
        
        
        first_legend = plt.legend(*scatter.legend_elements(), title='Divisions',\
                                 loc='upper right')
        plt.gca().add_artist(first_legend)
        
        plt.legend(handles=[meanli,medli,minli,maxli,extra].append(modli), bbox_to_anchor=(1.04,0.5), 
                   loc='center left',borderaxespad=0, prop={'size':16})
        plt.grid()
        plt.title('Scatterplot of `net_times` FEMALES', size=17) 
        plt.ylabel('`net_time` in seconds', size=17)
        plt.xlabel('Participants/Racers', size=17)
        plt.show()

        # Visualizing Scatterplot & Stats for Males
        figure(figsize=(10,10), dpi=80)
        subset = self.m_df[self.m_df.division_new.notnull()]
        subset = subset[subset.division_new>0]
        colors = subset['division_new']
        scatter = plt.scatter(subset['age'], subset['net_time'],
                    c=colors, alpha=0.35, label=set(colors))
        mu,mi,med,mx =\
            subset['net_time'].describe().loc[['mean','min','50%','max']]
        mod = subset['net_time'].mode()[0]
        
        mod = subset['net_time'].mode()
        meanli = plt.axhline(y=mu,color='red', linewidth=5,
                            linestyle='-', label='Mean: {0:.3f} Sec'.format(mu))
        medli = plt.axhline(y=med,color='k', linewidth=5,
                            linestyle='dotted', label='Median: {0:.3f} Sec'.format(med))
        
        #modeli = plt.axhline(y=mod,color='orange', linewidth=4,
        #                    linestyle='--', label='Mode: {0:.3f} Sec'.format(mod))
        modli=[]
        for i in range(len(mod)):
            modli.append(plt.axhline(y=mod[i],color='orange', linewidth=2,
                            linestyle='--', label='Mode: {0:.3f} Sec'.format(mod[i])))
        
        minli = plt.axhline(y=mi,color='aqua', linewidth=3,
                            linestyle='--', label='Minimum: {0:.3f} Sec'.format(mi))
        maxli = plt.axhline(y=mx,color='purple', linewidth=3,
                            linestyle='--', label='Max: {0:.3f} Sec'.format(mx))
        extra = plt.axhline(y=mi,linewidth=0, label='Range: {0:.3f}'.format(mx-mi))
        
        first_legend = plt.legend(*scatter.legend_elements(), title='Divisions',\
                                 loc='upper right')
        plt.gca().add_artist(first_legend)
        
        plt.legend(handles=[meanli,medli,minli,maxli,extra].append(modli), bbox_to_anchor=(1.04,0.5),
                   loc='center left',borderaxespad=0, prop={'size':16})
        plt.grid()
        plt.title('Scatterplot of `net_times` MALES', size=17)
        plt.ylabel('`net_time` in seconds (s)', size=17)
        plt.xlabel('Participants/Racers', size=17)
        plt.show()


    # Correlation Plots between Gun and Net time
    def vis_q2(self):

        # Combined Gender Correlation Data Plot
        figure(figsize=(10,10), dpi=80)
        subset = self.all_df[self.all_df.division_new.notnull()]
        subset = subset[subset.division_new>0].sort_values('net_time')
        colors = subset['division_new']
        
        scatter = plt.scatter(subset['net_time'], subset['gun_time'],\
                   c=colors, alpha=1, s=15, label=set(colors))
        plt.plot(subset['net_time'], subset['gun_time'],\
                linewidth=.5, color='0.85')
        first_legend = plt.legend(*scatter.legend_elements(), title='Divisions',\
                                 loc='lower right')
        plt.gca().add_artist(first_legend)
        
        plt.grid()
        plt.title('`gun_time` vs `net_time` comparison, Both',size=17)
        plt.xlabel('`net_time` in seconds (s)', size=15)
        plt.ylabel('`gun_time` in seconds (s)',size=15)
        plt.show()

        # Female Correlation Data plot
        figure(figsize=(10,10), dpi=80)
        subset = self.f_df[self.f_df.division_new.notnull()]
        subset = subset[subset.division_new>0]
        colors = subset['division_new']
        
        plt.scatter(subset['net_time'], subset['gun_time'],\
                   c=colors, alpha=1, s=15, label=set(colors))
        plt.plot(subset['net_time'], subset['gun_time'],\
                linewidth=.5, color='0.85')
        first_legend = plt.legend(*scatter.legend_elements(), title='Divisions',\
                                 loc='lower right')
        plt.gca().add_artist(first_legend)
        
        plt.grid()
        plt.title('`gun_time` vs `net_time` comparison, Female', size=17)
        plt.xlabel('`net_time` in seconds (s)', size=15)
        plt.ylabel('`gun_time` in seconds (s)', size=15)
        plt.show()

        # Male Correlation Data Plot
        figure(figsize=(10,10), dpi=80)
        subset = self.m_df[self.m_df.division_new.notnull()]
        subset = subset[subset.division_new>0]
        colors = subset['division_new']
        
        plt.scatter(subset['net_time'], subset['gun_time'],\
                   c=colors, alpha=1, s=15, label=set(colors))
        plt.plot(subset['net_time'], subset['gun_time'],\
                linewidth=.5, color='0.85')
        first_legend = plt.legend(*scatter.legend_elements(), title='Divisions',\
                                 loc='lower right')
        plt.gca().add_artist(first_legend)
        
        plt.grid()
        plt.title('`gun_time` vs `net_time` comparison, Male',size=17)
        plt.xlabel('`net_time` in seconds (s)',size=15)
        plt.ylabel('`gun_time` in seconds (s)',size=15)
        plt.show()

    def vis_q3(self):
        figure(figsize=(10,10), dpi=80)
        chrisdoe = self.all_df.loc[self.all_df.name=='chris doe',:]
        cd_nt = int(chrisdoe.net_time)
        net_array =\
            np.array(self.all_df.loc[self.all_df.division_new==float(chrisdoe.division_new),'net_time'])
        colors = ['blue']
        
        # Box plot generation
        bp = plt.boxplot(net_array, patch_artist=True, notch=True)
        q1, q50,q3,q10= np.quantile(net_array, [.25,.50,.75,.1])
        mu = net_array.mean()
        meanli = plt.axhline(y=mu,color='red', linewidth=2,
                            linestyle='-', label='Mean: {0:.3f} sec'.format(mu))
        q1li = plt.axhline(y=q1,color='aqua', linewidth=2,
                            linestyle='--', label='Q1: {0:.3f} sec'.format(q1))
        medli = plt.axhline(y=q50,color='red', linewidth=2,
                            linestyle='dotted', label='Median: {0:.3f} sec'.format(q50))
        q3li = plt.axhline(y=q3,color='purple', linewidth=2,
                            linestyle='--', label='Q3: {0:.3f} sec'.format(q3))
        q10li = plt.axhline(y=q10,linewidth=6,\
                            linestyle=':',label='Q 10%: {0:.3f} sec'.format(q10))
        
        cdli = plt.axhline(y=cd_nt, color='orange', linewidth=7,linestyle='-.',\
                          label='Chris Doe `net_time`: {} sec\nSeparation from Top 10%: {} sec'\
                           .format(int(chrisdoe.net_time), int(chrisdoe.net_time)-q10))
        plt.legend(handles=[meanli,q1li,medli,q3li, q10li, cdli], bbox_to_anchor=(1.04,0.5),\
                   loc='center left', borderaxespad=0, prop={'size':17})
        
        plt.xticks([1],['Males'], size=13)
        plt.title('Box Plot of `net_times` of Division {}, Find Chris Doe:'\
                  .format(int(chrisdoe.division_new)), size=20)
        plt.ylabel('`net_time` in seconds (s)', size=17)
        plt.xlabel('Participants/Racers', size=17)
        plt.grid()
        plt.show()

    def vis_q4(self):
        # Building Stacked histogram for average time difference per state
        import seaborn as sns
        subset = self.all_df[self.all_df.division_new.notnull()]
        subset = self.all_df[self.all_df.state.notnull()]
        subset = subset[subset.division_new>0].sort_values('net_time')
        color = ['blue','orange','green','red','purple',\
                 'pink','gray', 'olive','cyan','lightseagreen','violet',\
                'darkred', 'cornflowerblue','navy','darkcyan'] #list(np.random.choice(range(256), size=15))
        
        color = color[:len(set(subset.state))]
        
        agg_stacked =\
            subset.groupby(['division_new','state'])['diff_time'].mean().unstack().fillna(0)
        
        fig, ax = plt.subplots(figsize=(15,20), dpi=80)
        bottom = np.zeros(len(agg_stacked))
        
        for i, col in enumerate(agg_stacked.columns):
            ax.bar(agg_stacked.index, agg_stacked[col],bottom=bottom,label=col,\
                  width=.9, color=color[i]) 
            bottom+=np.array(agg_stacked[col])
        
        avg_totals = agg_stacked.sum(axis=1)
        y_offset=5
        
        for i, total in enumerate(avg_totals):
            ax.text(avg_totals.index[i], total+y_offset, round(total),\
                    ha='center', weight='bold', size=17)
        
        y_offset=-15
        for i, bar in enumerate(ax.patches):
            if bar.get_height()>0:
                if bar.get_height()<50:
                # Putting the text in the middle of each bar
                    ax.text(\
                        bar.get_x()+ bar.get_width(),\
                        bar.get_height() + bar.get_y(),\
                        round(bar.get_height()),
                        ha='right', color='w', weight='bold', size=14)
                else:
                    ax.text(\
                        bar.get_x() +bar.get_width()/2,\
                        bar.get_height() + bar.get_y() +y_offset,\
                        round(bar.get_height()),
                        ha='right', color='w', weight='bold', size=14)
            
        plt.grid()
        ax.set_ylim([0,1800])
        ax.set_title('Average difference in Time by State', size=20)
        ax.set_xlabel('Divisions by Age' , size=17)
        ax.set_ylabel('Difference in Time (`gun_time` - `net_time`) in Seconds (s)', size=17)
        ax.legend(prop={'size':17})

        # Stacked bar graph with counts of each state
        subset = self.all_df[self.all_df.division_new.notnull()]
        subset = self.all_df[self.all_df.state.notnull()]
        subset = subset[subset.division_new>0].sort_values('net_time')
        
        color = ['blue','orange','green','red','purple',\
                 'pink','gray', 'olive','cyan','lightseagreen','violet',\
                'darkred', 'cornflowerblue','navy','darkcyan'] #list(np.random.choice(range(256), size=15))
        
        color = color[:len(set(subset.state))]
        
        agg_stacked =\
            subset.groupby(['division_new','state'])['diff_time'].count().unstack().fillna(0)
        
        fig, ax = plt.subplots(figsize=(15,20), dpi=80)
        bottom = np.zeros(len(agg_stacked))
        
        for i, col in enumerate(agg_stacked.columns):
            ax.bar(agg_stacked.index, agg_stacked[col],bottom=bottom,label=col,\
                  width=.9, color=color[i]) 
            bottom+=np.array(agg_stacked[col])
        
        avg_totals = agg_stacked.sum(axis=1)
        y_offset=1
        
        for i, total in enumerate(avg_totals):
            ax.text(avg_totals.index[i], total+y_offset, round(total),\
                    ha='center', weight='bold', size=17)
        
        y_offset=-15
        for i, bar in enumerate(ax.patches):
            if bar.get_height()>0:
                if bar.get_height()<50:
                # Putting the text in the middle of each bar
                    ax.text(\
                        bar.get_x()+ bar.get_width(),\
                        bar.get_height() + bar.get_y() -.75,\
                        round(bar.get_height()),
                        ha='right', color='w', weight='bold', size=14)
                else:
                    ax.text(\
                        bar.get_x() +bar.get_width()/2,\
                        bar.get_height() + bar.get_y() -5,\
                        round(bar.get_height()),
                        ha='right', color='w', weight='bold', size=14)
            
        plt.grid()
        ax.set_ylim([0,1000])
        ax.set_title('Counts of Participants by State', size=20)
        ax.set_xlabel('Divisions by Age' , size=20)
        ax.set_ylabel('Difference in Time (`gun_time` - `net_time`) in Seconds (s)', size=20)
        ax.legend(prop={'size':17})


        # Average difference by state (10 % percentile)
        subset = self.all_df[self.all_df.division_new.notnull()]
        subset = self.all_df[self.all_df.state.notnull()]
        subset = subset[subset.division_new>0].sort_values('net_time')
        q10= np.quantile(subset.diff_time, .1)
        subset = subset[subset.diff_time<q10]
        color = ['blue','orange','green','red','purple',\
                 'pink','gray', 'olive','cyan','lightseagreen','violet',\
                'darkred', 'cornflowerblue','navy','darkcyan'] #list(np.random.choice(range(256), size=15))
        
        color = color[:len(set(subset.state))]
        
        agg_stacked =\
            subset.groupby(['division_new','state'])['diff_time'].mean().unstack().fillna(0)
        
        fig, ax = plt.subplots(figsize=(15,20), dpi=80)
        bottom = np.zeros(len(agg_stacked))
        
        for i, col in enumerate(agg_stacked.columns):
            ax.bar(agg_stacked.index, agg_stacked[col],bottom=bottom,label=col,\
                  width=.9, color=color[i]) 
            bottom+=np.array(agg_stacked[col])
        
        avg_totals = agg_stacked.sum(axis=1)
        y_offset=1
        
        for i, total in enumerate(avg_totals):
            ax.text(avg_totals.index[i], total+y_offset, round(total),\
                    ha='center', weight='bold', size=17)
        
        y_offset=-15
        for i, bar in enumerate(ax.patches):
            if bar.get_height()>0:
                if bar.get_height()<50:
                # Putting the text in the middle of each bar
                    ax.text(\
                        bar.get_x()+ bar.get_width(),\
                        bar.get_height() + bar.get_y() -.75,\
                        round(bar.get_height()),
                        ha='right', color='w', weight='bold', size=14)
                else:
                    ax.text(\
                        bar.get_x() +bar.get_width()/2,\
                        bar.get_height() + bar.get_y() -5,\
                        round(bar.get_height()),
                        ha='right', color='w', weight='bold', size=14)
            
        plt.grid()
        ax.set_ylim([0,40])
        ax.set_title('Average difference in Time by State (Top 10%)', size=20)
        ax.set_xlabel('Divisions by Age' , size=20)
        ax.set_ylabel('Difference in Time (`gun_time` - `net_time`) in Seconds (s)', size=20)
        ax.legend(prop={'size':17})

        # Counts of participants by state (Top 10%)
        subset = self.all_df[self.all_df.division_new.notnull()]
        subset = self.all_df[self.all_df.state.notnull()]
        subset = subset[subset.division_new>0].sort_values('net_time')
        q10= np.quantile(subset.diff_time, .1)
        subset = subset[subset.diff_time<q10]
        
        color = ['blue','orange','green','red','purple',\
                 'pink','gray', 'olive','cyan','lightseagreen','violet',\
                'darkred', 'cornflowerblue','navy','darkcyan']         
        color = color[:len(set(subset.state))]
        
        agg_stacked =\
            subset.groupby(['division_new','state'])['diff_time'].count().unstack().fillna(0)
        
        fig, ax = plt.subplots(figsize=(15,20), dpi=80)
        bottom = np.zeros(len(agg_stacked))
        
        for i, col in enumerate(agg_stacked.columns):
            ax.bar(agg_stacked.index, agg_stacked[col],bottom=bottom,label=col,\
                  width=.9, color=color[i]) 
            bottom+=np.array(agg_stacked[col])
        
        avg_totals = agg_stacked.sum(axis=1)
        y_offset=1
        
        for i, total in enumerate(avg_totals):
            ax.text(avg_totals.index[i], total+y_offset, round(total),\
                    ha='center', weight='bold', size=17)
        
        y_offset=-15
        for i, bar in enumerate(ax.patches):
            if bar.get_height()>0:
                if bar.get_height()<50:
                # Putting the text in the middle of each bar
                    ax.text(\
                        bar.get_x()+ bar.get_width(),\
                        bar.get_height() + bar.get_y() -.75,\
                        round(bar.get_height()),
                        ha='right', color='w', weight='bold', size=14)
                else:
                    ax.text(\
                        bar.get_x() +bar.get_width()/2,\
                        bar.get_height() + bar.get_y() -5,\
                        round(bar.get_height()),
                        ha='right', color='w', weight='bold', size=14)
            
        plt.grid()
        ax.set_ylim([0,70])
        ax.set_title('Counts of Participants by State (Top 10%)', size=20)
        ax.set_xlabel('Divisions by Age' , size=20)
        ax.set_ylabel('Difference in Time (`gun_time` - `net_time`) in Seconds (s)', size=20)
        ax.legend(prop={'size':17})

        # Side by Side Box Plot comparison of gender race results by division
        figure(figsize=(10,10), dpi=80)
        subset.head()
        ax=sns.boxplot(x='division_new', y='net_time', hue='gender',data=subset)
        plt.grid()
        plt.title("`net_time` by Division by Gender",size=17)
        plt.xlabel("Participants/Racers Division", size=15)
        plt.ylabel("`net_time` in Seconds (s)", size=15)
        plt.legend(prop={'size':17})
        plt.show()

