# IMPORT
import pandas as pd
import os
import datetime
import numpy as np

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 

pd.options.mode.chained_assignment = None

from functions.admin import * 
from functions.permits import *

# INTITIAL FUNCTIONS
index = 'Application #'
index_col = index
sum_columns = [
    'Total Days', 'Total Business Days', 'City Business Days', 'Rounds', 'Days till Routing'
]
date_columns = [
    'Days per Round', 
    'Submittal Date', 
    'Final Date'
]
keep_fields = [index] + sum_columns + date_columns

board_tasks = [
                    'Planning Commission', 'Design Review Commission',
                    'City Council', 'Zoning Administrator'
                ]

phases = {
    'Review':{
        'start':'Intake Review',
        'round start':'Resubmittal',
        'round end':'Consolidated Comments',
        'end':'Staff Analysis',
        'tasks':[
            'Intake Review',
            'Engineering Review', 'Planning Review',
            'CEQA Review', 'Other Review', 'Traffic Engineering Review',
            'Building Review', 'Arborist Review', 'Fire Review', 
            'Economic Development Review', 'Police Review', 
            'Resubmittal', 'Staff Analysis', 'Consolidated Comments'
        ]
    },
    'Decision':{
        'start':'Staff Level Decision',
        'round start':'Resubmittal',
        'end':'',
        'round end':'',
        'tasks':[
            'Staff Level Decision', 'Zoning Administrator', 
            'Planning Commission', 'Design Review Commission',
            'City Council', 'Close Out'
        ]
    }
}

task2rank ={
    # REVIEW
    'Intake Review': 11,
    'Resubmittal': 12,

    'Planning Review': 21,
    'Building Review': 22,
    'Engineering Review': 23,
    'Traffic Engineering Review': 25,
    'Fire Review': 26,
    'Economic Development Review': 31,
    'Police Review': 32,
    'Arborist Review': 33,
    'CEQA Review': 34,
    'Other Review': 35,

    'Consolidated Comments': 41,
    'Staff Analysis': 42,
    
    # DECISION
    'Staff Level Decision': 51,
    'Zoning Administrator': 61,
    'Planning Commission': 62,
    'Design Review Commission': 63,
    'City Council': 64,
    'Close Out': 71
 }


phase_nickname = {
    'R':'Review',
    'D':'Decision'
}

remove_tasks = [
    'Arborist Review',
    'CEQA Review'
]


sorts = [
    'Date Status', 
    'Task Rank',
    'Updated Date', 
    'Date Assigned'
    ]


def tasks(phase, item='tasks', phases=phases, phase_nickname=phase_nickname):
    if phase not in phases.keys():
        phase = phase_nickname[phase.upper()]
    return phases[phase][item]

phase2tasks = {v:tasks(v) for k,v in phase_nickname.items()}
p2t = phase2tasks
task2phase = {}
for t,ts in phase2tasks.items():
    for tks in ts:
        if tks != '':
            task2phase[tks] = t
t2p = task2phase

task2startend = {}
for action in ['end', 'round end', 'round start', 'start']:
    for k,v in phase_nickname.items():
        task = tasks(v, action)
        if task != '':
            task2startend[task] = action
t2se = task2startend

# 2. Preparing Admin & Count Dataframes
### 2.1 Core Format Functions


def sum_days(strt, nd, business_days=True, holidays=get_holidays()):
    if business_days == True:
        days = [d for d in pd.bdate_range(start=strt, end=nd).to_list() if d not in holidays]
    if business_days == False:
        days = pd.date_range(start=strt, end=nd).to_list()
    return days

def make_dates(permit_df):
    # turns date columns into datetime format
    date_cols = [tt for tt in list(permit_df) if ('Date ' in tt)or(' Date' in tt)]
    permit_df[date_cols] = permit_df[date_cols].apply(pd.to_datetime)
    return permit_df

#### rank tasks to better sort
def add_task2rank(df, tr_dict = task2rank, new_col = 'Task Rank', old_col = 'Task'):
    df[new_col] = df[old_col].map(tr_dict)
    return df

def prepare_times(raw_permit_df, extra_list=[]):
    #'APN', 'DESCRIPTION', 'Address'
    n=0
    ## check if Accela added a title/description in the first row
    raw_permit_df = raw_permit_df.copy()
    if 'P_ID' not in list(raw_permit_df):
        raw_permit_df = pd.read_csv(file_path, skiprows=0)
    
    raw_permit_df.rename(columns={'P_ID':'Application #'}, inplace=True)
    
    # check date columns for tracking date
    #if 'Date Status' not in date_cols:
    #    raw_permit_df['Date Status'] = raw_permit_df['Updated Date']
    
    # The main column of tracking
    raw_permit_df['Task Status'] = raw_permit_df['Task'] + ' - ' + raw_permit_df['Status']
    
    raw_admin_df = raw_permit_df.copy()
    
    raw_fields = ['Application #', 'Permit Type', 'Record Status']    
    
    raw_fields.extend([
        c for c in extra_list 
            if c in list(raw_admin_df) or 
            c.title() in list(raw_admin_df) or 
            c.upper() in list(raw_admin_df)
    ])

    #raw_admin_df = raw_admin_df[raw_fields].drop_duplicates().set_index('Application #')
    raw_admin_df = raw_admin_df \
        [[c for c in raw_admin_df 
            if 'Date' not in c or 
            c not in [
                'Task Status', 'Task', 'Status', 'E_ID'
            ]
            ]] \
        .drop_duplicates(subset=['Application #', 'Permit Type', 'Record Status']) \
        .set_index('Application #')
    
    raw_admin_df['Notes'] = \
        raw_admin_df['Notes'] \
            .fillna('') \
            .replace('nan', '')

    # sort tasks by rank
    raw_permit_df = add_task2rank(raw_permit_df.copy())
    # sort by when the statuses were updated
    raw_permit_df.sort_values(by=['Application #']+sorts, inplace=True)

    # Fields of Focus
    raw_permit_df = raw_permit_df[[
        'Application #', 'Task Status', 'Date Assigned', 'Date Status', 'Updated Date']]
    
    # only selects if the record is in our time period and in our focus Task Statuses
    #raw_permit_df_endfinder = raw_permit_df[raw_permit_df['Task Status'].isin(stat_d.keys())].copy()
    #ends_after_Q = set(raw_permit_df_endfinder.loc[(raw_permit_df_endfinder['Updated Date'].dt.to_period('Q')>less), 'Application #'].values)
    #raw_permit_df = raw_permit_df[raw_permit_df['Application #'].isin(ends_after_Q)]
    

    raw_permit_df = raw_permit_df.rename(columns={'Application #':'#'})
    raw_permit_df = raw_permit_df.set_index(['#', 'Task Status'])
    
 
    old_idx = raw_permit_df.index.names
    raw_permit_df = raw_permit_df \
        .reset_index() \
        .dropna(axis=0, subset=['#','Task Status']) \
        .set_index(old_idx)
    
    # For Planning Applications, allow received date
    if 'Status - Received' in raw_permit_df.index.get_level_values(1):
        
        raw_permit_df.reset_index(inplace=True)
        focus = set(raw_permit_df[raw_permit_df['Task Status'].isin(['Status - Assigned', 'Status - Received'])].index)
        #IDs = {i:[] for i in focus}
        #[IDs[i].append(ts) for i in focus]
        #IDs = [k for k,v in IDs.items() if len(v) > 1]
        
        #for x in focus:
        #    raw_permit_df.loc[x, 'Date Assigned'] = raw_permit_df.loc[x, 'Date Status']
        #    raw_permit_df.loc[x, 'Date Status'] = raw_permit_df.loc[x, 'Updated Date']

        raw_permit_df.set_index(['#', 'Task Status'], inplace=True)
        
    return raw_permit_df, raw_admin_df

### 2.1.1 First Clean

def first_clean(permit_df):

    permit_df = (
            permit_df
                .drop_duplicates(
                    subset=['P_ID', 'Permit Type', 'Task', 'Status', 'Date Status'], 
                    keep='first'
                )
                )
    return permit_df


### 2.2 Accessory Prep Functions
#### 2.2.1 Entitlements

# FIX MASTER SIGN PERIT
def fix_msp(row):
    ptype = row['Permit Type']
    pdesc = row['Description']

    msps = ['msp', 'msps', 'sign', 'signs']

    for dword in pdesc.split(' '): 
        dword = dword.lower()
        if dword in msps:
            return 'Design Review Sign'
    return pdesc

# FIX ENTITLEMENTS
type_fix = {
        'ZCL':['ZCL', 'Zoning Compliance'],
        'Design Review Sign':['MSP', 'SIGN', 'SIGNS', 'Sign', 'Signs', 'msp', 'MSPA', 'rebranding'],
        'PRE APP':['PRE APP', 'Pre-App', 'PRE-APP'],
        'Reasonable Accomodation':['Reasonable Accomodation'],
        'SUP':['Special Use Permit'],
        'LFDCH':['LFDCH']
    }
def fix_ents(
        test_df,
        type_fix=type_fix
        ):
    
    col_search = ['Description', 'Notes']
    type_col = 'Permit Type'
    other_values = 'Other'
    replace_col = 'Permit Type'

    test_df[[type_col]+col_search] = test_df[[type_col]+col_search].astype(str).fillna('')

    for tkey, tvalues in type_fix.items():
        for tvalue in tvalues:
            for focus_col in col_search:
                temp_filt = (test_df[type_col].str.contains(other_values))&(test_df[focus_col].str.contains(tvalue))
                if len(test_df[temp_filt]) > 0:
                    test_df.loc[temp_filt, replace_col] = tkey
    return test_df

ent_cols = ['Entitlements', 'Num of Entitlements']
nicknames = {
        'Design Review':'DR',
        'Use Permit':'UP',
        'Administrative':'Admin',
        'Residential':"Resi",
        'Commercial':'Comm'
    }
def group_entitlments(
        raw_permit_dataframe, 
        ent_cols = ent_cols,
        nicknames = nicknames):
    

    def clean_ent(ents):
        clean_ents = list(set(ents))
        return clean_ents
    def ent_agg(ents):
        lents = []
        
        for item in clean_ent(ents):
            item = str(item)
            for nick in nicknames.keys():
                if nick in item:
                    item = item.replace(nick, nicknames[nick])
            lents.append(item)
        lents.sort()

        sents = ', '.join([str(i) for i in lents])
        if len(lents) == 1:
            sents = sents.strip('*')
        return sents
    def ent_count(ents):
        return len(clean_ent(ents))

    group_permits = raw_permit_dataframe.copy()
    group_permits.loc[group_permits['Primary']=='Yes', 'Permit Type'] = '*' + group_permits['Permit Type'].astype(str)

    group_permits = group_permits.groupby('P_ID').agg({
        'Permit Type':[ent_agg, ent_count]
    })

    group_permits.columns = group_permits.columns.droplevel()
    group_permits.columns = ent_cols

    return raw_permit_dataframe.join(group_permits, on='P_ID')

#### 2.2.2 Decision & Intake

hearing_tasks = ['City Council', 'Planning Commission', 'Design Review Commission', 'Zoning Administrator']
def get_decisions(raw_permit_df):
    staff_tasks = ['Staff Level Decision']

    raw_permit_df['Staff Decision'] = 'False'
    staff_filt = raw_permit_df['Task'].isin(staff_tasks)
    raw_permit_df.loc[staff_filt, 'Staff Decision'] = 'True'
    staff_ids = raw_permit_df.loc[staff_filt, 'P_ID'].unique()
    raw_permit_df.loc[raw_permit_df['P_ID'].isin(staff_ids), 'Staff Decision'] = 'True'

    raw_permit_df['Board Decision'] = 'False'
    board_filt = raw_permit_df['Task'].isin(hearing_tasks)
    raw_permit_df.loc[board_filt, 'Board Decision'] = 'True'
    board_ids = raw_permit_df.loc[board_filt, 'P_ID'].unique()
    raw_permit_df.loc[raw_permit_df['P_ID'].isin(board_ids), 'Board Decision'] = 'True'

    raw_permit_df.loc[raw_permit_df['Board Decision']=='True', 'Staff Decision'] = 'False'

    return raw_permit_df

def get_planner(permit_df, planner_cols = ['Planner F', 'Planner L']):
    def combine_str(x):
        l = [i for i in list(x) if i!='']
        d = ', '.join(list(dict.fromkeys(l)))
        return d

    def group_permit_df_by_planner_count(permit_df, planner_cols = ['Planner F', 'Planner L']):
        grp = permit_df.groupby(['P_ID', planner_cols]) \
            .agg({'Primary':len}) \
            .rename(columns = {'Primary':'Row Count'}) \
            .reset_index() \
            .sort_values('Row Count', ascending=False)
        grp['Planner'] = grp['Planner'].fillna('')
        grp = grp.groupby('P_ID') \
            .agg({'Planner':combine_str})
        return grp['Planner'].to_dict()

    test_dict = group_permit_df_by_planner_count(permit_df, planner_cols = planner_cols)

    permit_df['Planner'] = permit_df['P_ID'].map(test_dict)

    permit_df['Planner'].value_counts()

    return permit_df


def add_list_to_dict(old_dict, new_list, fill=False):
    new_dict = old_dict.copy()
    for i in new_list:
        if i not in new_dict.keys():
            if fill == False:
                new_dict[i] = i
            else:
                new_dict[i] = fill
    return new_dict

def get_planner(permit_df, planner_col = 'Planner', first_suffix ='F', second_suffix = 'L'):
    p_first_col = planner_col + ' ' + first_suffix
    p_second_col = planner_col + ' ' + second_suffix
    def combine_str(x):
        l = [i for i in list(x) if i!='']
        d = ', '.join(list(dict.fromkeys(l)))
        return d

    def group_permit_df_by_planner_count(permit_df, planner_cols = [p_first_col, p_second_col]):
        grp = permit_df.groupby(['P_ID', planner_cols]) \
            .agg({'Primary':len}) \
            .rename(columns = {'Primary':'Row Count'}) \
            .reset_index() \
            .sort_values('Row Count', ascending=False)
        grp[planner_col] = grp[planner_col].fillna('')
        grp = grp.groupby('P_ID') \
            .agg({planner_col:combine_str})
        return grp[planner_col].to_dict()

    permit_df[planner_col] = ''
    plan_rev_filt = permit_df['Task']=='Planning Review'
    permit_df.loc[plan_rev_filt, planner_col] = permit_df.loc[plan_rev_filt, p_first_col] + ' ' + permit_df.loc[plan_rev_filt, p_second_col]

    def combine_names(x):
        return ', '.join(set(x)).title().strip()

    planner_group = (
        permit_df
            .loc[plan_rev_filt]
            .groupby('P_ID')
            .agg({planner_col:lambda x: combine_names(x)})
            [planner_col]
            .to_dict()
    )
    
    planner_group = add_list_to_dict(
        planner_group,
        permit_df['P_ID'].unique(),
        fill=''
    )
    
    permit_df[planner_col] = permit_df['P_ID'].map(planner_group)
    
    df_cols = [
        planner_col if col == p_first_col 
        else col
        for col in permit_df.columns.tolist()
        if col not in [planner_col, p_second_col]
        ]

    return permit_df[df_cols]


#### 2.2.3 Board Info & Days

def get_board_info(raw_permit_df):
    hearing_dict = {
        'Planning Commission' : 'PC',
        'Design Review Commission':'DRC',
        'City Council':'CC',
        'Zoning Administrator':'ZA'
    }

    def hearing_list(hearings):
        return [hearing_dict[h] for h in hearings]

    def count_hearings(hearings):
        return len(hearing_list(hearings))

    def string_list(list):
        return ', '.join(list)

    def hearing_string(hearings):
        return string_list(hearing_list(hearings))

    board_df =    raw_permit_df[raw_permit_df['Task'].isin(hearing_tasks)] \
            .drop_duplicates(subset=['P_ID', 'Task']) \
            [['P_ID', 'Task']] \
            .groupby('P_ID') \
            .agg({
                'Task':[hearing_string, count_hearings]
            }) \
            .droplevel(0, axis=1) \
            .rename(columns = {'hearing_string':'Boards','count_hearings':'Board Count'})

    #board_df
    raw_permit_df['Boards']=''
    raw_permit_df['Board Count']=0
    l_dict = board_df['Boards'].to_dict()
    c_dict = board_df['Board Count'].to_dict()

    board_filt = raw_permit_df['P_ID'].isin(c_dict.keys())
    raw_permit_df.loc[board_filt, 'Boards'] = raw_permit_df.loc[board_filt, 'P_ID'].map(l_dict)
    raw_permit_df.loc[board_filt, 'Board Count'] = raw_permit_df.loc[board_filt, 'P_ID'].map(c_dict)

    return raw_permit_df


#### 2.2.4 Address

# make a functional called clean_add_col that takes a column called "Address" and only keep the text before "WALCR"
def clean_add_col(add, splits = ['WALCR', 'WALNUT CREEK']):
    
    splits = list(splits)

    for split in splits:
        if split in add:
            add = add.split(split)[0]
    
    return add.strip().strip(',').strip().upper()

def clean_addresses(permit_df, add_col = 'Address', splits = ['WALCR', 'WALNUT CREEK'], remove_2nd_add = True):

    if add_col in list(permit_df):
        permit_df[add_col] = permit_df[add_col].apply(clean_add_col, splits = splits)

    if remove_2nd_add == True:
        permit_df[add_col] = permit_df[add_col].apply(clean_add_col, splits = [','])

    return permit_df

def add_phase_and_action(df):
    df['Phase'] = df['Task'].map(task2phase)
    df['Action'] = df['Task'].map(task2startend).fillna('')
    return df

def get_task_ids(pdf, id_col='P_ID'):
    pidf = pdf \
        .sort_values(sorts) \
        .reset_index(drop=True) \
        .copy()

    pidf['index'] = pidf.index
    pidf['T_ID'] = pidf \
        .groupby(id_col)['index'] \
        .transform(lambda x: (pd.factorize(x)[0]+1)) \
        .astype(str).str.zfill(3)
    pidf['T_ID'] = pidf[id_col] \
        .replace('Y','') + '-' + pidf['T_ID']
    return pidf

def get_permit_dictionary(prep_permit_df):
    pdict = prep_permit_df \
        .groupby(['P_ID','Phase']) \
        ['Record Type'].count() \
        .unstack() \
        .to_dict(orient='index') #.reset_index() \    #.set_index('P_ID') \
    pdict = {p:{k:{'counts':{'all':ph}} for k,ph in d.items()} for p,d in pdict.items()}
    return pdict

def fix_back2back_end_actions(pdf, pids, focus_cols = [
    'P_ID',
    'T_ID',
    #'Updated Date',
    'Date Assigned',
    'Date Status',
    'Task',
    'Status',
    #'Task Rank',
    'Phase',
    'Action'
]):

    pdf = pdf.copy()

    remove_phase_tids = []
    fix_no_end = []

    # find round ends touching ends
    def find_roundend_end_actions(fdf):
        fdf = fdf[fdf['Action']!=''].copy()
        fdf['Action Below'] = fdf['Action'].shift(-1)
        ree_filt = (fdf['Action']=='round end')&(fdf['Action Below']=='end')
        return fdf[ree_filt]['T_ID'].to_list()

    withdrawn_filt = (pdf['Task']=='Resubmittal')&(pdf['Status']=='Withdrawn')
    pdf.loc[withdrawn_filt, 'Action'] = 'end'

    for focus_id in pids:

        id_filt = (pdf['P_ID']==focus_id)
        phase_filt = (pdf['Phase']=='Review')
        task_filt = (~pdf['Task'].isin(remove_tasks))
        ts_filt = (~((pdf['Task']=='Design Review Commission')&(pdf['Status']=='Continued')))
        sa_filt = (~((pdf['Task']=='Staff Analysis')&(pdf['Status']=='Deemed Complete')))

        frow = pdf[id_filt&phase_filt&task_filt&ts_filt][focus_cols].copy()
        
        if 'Staff Analysis' not in frow['Task'].unique():
            cc_fix_filt = (frow['Task']=='Consolidated Comments')&(frow['Status']=='Deemed Complete')
            fix_no_end.extend(frow.loc[cc_fix_filt, 'T_ID'].to_list())

        remove_phase_tids.extend(find_roundend_end_actions(frow))

    pdf.loc[pdf['T_ID'].isin(remove_phase_tids), 'Action'] = ''
    pdf.loc[pdf['T_ID'].isin(fix_no_end), 'Action'] = 'end'

    return pdf
## Round Functions

def find_round(search_df, id_col = 'T_ID', action_col = 'Action'):
    # finds rounds then exports as list
    start_id = None
    end_id = None
    round_num = 0
    round_list = []

    pid = list(search_df[id_col].unique())[0].rsplit('-', 1)[0]

    # python replace list item (round_list) with another based on
    # 

    for index,row in search_df.iterrows():
        
        if (row['Phase'] == 'Review')and(pid not in bad):
            if 'start' in row[action_col]:
                start_id = row[id_col]
                end_id = None
                round_num += 1
                round_list.append(round_num)
                

            elif 'end' in row[action_col]:
                start_id = None 
                end_id = row[id_col]
                round_list.append(round_num)
                

            else:
                if start_id is not None:
                    round_list.append(round_num)
                else:
                    round_list.append(0)
                

        else:
            round_list.append(0)

        #print(index, start_id, end_id, round_num, round_list, row['Phase'], row['Action'])    

    return pd.Series(round_list, name='Round')

def find_round_dict(group_df, id_col = 'T_ID', action_col = 'Action'):
    # Goes through an id then finds the rounds    
    # exports as dict    
    start_id = None
    end_id = None
    round_num = 0
    round_dict = {}

    pid = group_df[id_col].unique()[0].rsplit('-', 1)[0]

    for _, row in group_df.iterrows():
        if (row['Phase'] == 'Review') and (pid not in bad):
            if 'start' in row[action_col]:
                start_id = row[id_col]
                end_id = None
                round_num += 1
                round_dict[row[id_col]] = round_num
            elif 'end' in row[action_col]:
                start_id = None 
                end_id = row[id_col]
                round_dict[row[id_col]] = round_num
            else:
                if start_id is not None:
                    round_dict[row[id_col]] = round_num
                else:
                    round_dict[row[id_col]] = 0
        else:
            round_dict[row[id_col]] = 0

    return round_dict

def get_rounds(
        prep_permit_df, 
        main_id='P_ID', sub_id='T_ID',
        round_col = 'Round'
        ):
    # runs find rounds function over task df
    # creates large dictionary
    # maps dictionary over IDs
    round_dict = {}
    for _, group in prep_permit_df.groupby(main_id):
        round_dict.update(find_round_dict(group))
        
    # Now you can map the dict to create a new column
    prep_permit_df[round_col] = prep_permit_df[sub_id].map(round_dict)

    return prep_permit_df

# 3. Aggregate Functions

## Main Functions (un used)?
def find_between_start_end(search_df, id_col = 'T_ID', action_col = 'Action'):
    # I have a initial dataframe (called "seach_df") with two columns,  
    # #"T_ID" that is a unique ID for each row and 
    # "Action" that has three potential values: "start", "end", and None. 
    # I want to have a list of each "T_ID" that is in a row between rows that have 
    # a "start" and "end" in the "Action" column. 
    # I want the lists of "T_ID" values to go into a new dataframe with each row being 
    # each list with two new columns showing the "T_ID" value of the "start" and 
    # one for the "end"

    #in_between = []
    start_id = None
    end_id = None
    round_num = 0
    round_list = []

    for _,row in search_df.iterrows():

        if 'start' in row[action_col]:
            start_id = row[id_col]
            end_id = None
            round_list = []
            round_num += 1
            continue

        elif 'end' in row[action_col]:
            end_id = row[id_col]
            round_list.append(row[id_col])
            yield start_id, end_id, round_list, round_num

            round_list = []
            start_id = None 
            continue

        else:
            if start_id is not None:
                round_list.append(row[id_col])
            continue


## get review round's dates & days df
def get_review_round_dates(prep_permit_df):
    rev_round_dates = \
        prep_permit_df \
            .loc[prep_permit_df['Round']>0] \
            .dropna(subset=['Date Status']) \
            .groupby(['P_ID','Round']) \
            .agg({'Date Status':[min, max]})
    rev_round_dates.columns = rev_round_dates.columns.droplevel(0)
    rev_round_dates.columns = ['Start Date', 'Stop Date']

    rev_round_dates['Dates'] = rev_round_dates.apply(lambda x: sum_days(x['Start Date'], x['Stop Date'], business_days=False), axis=1)
    rev_round_dates['Biz Dates'] = rev_round_dates.apply(lambda x: sum_days(x['Start Date'], x['Stop Date'], business_days=True), axis=1)
    rev_round_dates['Days'] = rev_round_dates['Dates'].apply(len)
    rev_round_dates['Biz Days'] = rev_round_dates['Biz Dates'].apply(len)

    return rev_round_dates

## get total review dates
def get_review_dates(
        group, id_col = 'P_ID', 
        start_col = 'Start Date', end_col = 'Stop Date'
        ):
    start_date = group[start_col].min()
    end_date = group[end_col].max()
    return sum_days
## combine sub-lists into main-list
def merge_lists(series):
    return list(set([i for sublist in series for i in sublist]))

## get review's total, customer, and staff dates
def get_review_total_dates(rev_round_dates):
    rdate_aggs = {
        'Start Date': 'min', 'Stop Date': 'max', 
        'Dates':merge_lists, 'Biz Dates':merge_lists,
        'Round':'count'}
    rev_rename = {
        'Round':'Rounds', 
        'Dates':'Review Staff Dates', 
        'Biz Dates':'Review Staff Biz Dates'
        }
    rev_tot_dates = rev_round_dates \
        .reset_index() \
        .loc[rev_round_dates.reset_index()['Round'] > 0] \
        .groupby('P_ID') \
        .agg(rdate_aggs) \
        .rename(columns=rev_rename)
    rev_tot_dates['Review Dates'] = \
        rev_tot_dates \
            .apply(lambda row: sum_days(
                row['Start Date'], row['Stop Date'], business_days=False
            ), axis=1)
    rev_tot_dates['Review Biz Dates'] = \
        rev_tot_dates.apply(lambda row: sum_days(row['Start Date'], row['Stop Date']), axis=1)

    for date_col in ['Dates', 'Biz Dates']:
        rev_tot_dates[f'Customer {date_col}'] = \
            rev_tot_dates.apply(lambda row: list(set(row[f'Review {date_col}']) ^ set(row[f'Review Staff {date_col}'])), axis=1)

    return rev_tot_dates

## get Decision dates
def get_decision_dates(prep_permit_df):

    prep_permit_df['Max Review Date'] = \
        prep_permit_df \
            .loc[prep_permit_df['Phase']=='Review'] \
            .groupby('P_ID')['Date Status'] \
            .transform('max')
    prep_permit_df['Max Review Date'] \
        .fillna(pd.Timestamp('1900-01-01'), inplace=True)

    phase_filt = (prep_permit_df['Phase']=='Decision')
    max_filt = (prep_permit_df['Date Status'] >= prep_permit_df['Max Review Date'])

    dec_df = \
        prep_permit_df \
            .loc[phase_filt&max_filt] \
            .groupby('P_ID') \
            .agg({'Date Status': ['min', 'max']}) \
            .rename(columns={'min':'Start Date', 'max':'Stop Date'}) \
            .droplevel(0, axis=1)
    
    dec_df['Decision Dates'] = \
        dec_df \
            .apply(lambda row: sum_days(
                row['Start Date'], row['Stop Date'], 
                business_days=False), axis=1)
    
    dec_df['Decision Biz Dates'] = \
        dec_df \
            .apply(lambda row: sum_days(
                row['Start Date'], row['Stop Date'], 
                business_days=True), axis=1)
    
    dec_rename = {
        'Start Date':'Decision Start Date',
        'Stop Date':'Decision Stop Date'
    }
    dec_df \
        .rename(dec_rename, axis=1, inplace=True)
    
    return dec_df


# get decisions

def get_decision_type(prep_permit_df):
    prep_permit_df['Task Status'] = prep_permit_df['Task'] + ' - ' + prep_permit_df['Status']

    def check_action(group, check_col, change_col, check_list, new_val='Yes'):
        if group[check_col].isin(check_list).any():
            group[change_col] = new_val
        else:
            group[change_col] = group[change_col]
        return group

    def check_action_filt(group, check_col, change_col, check_list, new_val='Yes',
                        filt_date_col='Date Status', filt_col='Phase', filt_val='Decision', old_val = None):
        # creating a filtered df so that it only checks the group that is the most recent
        # example: I only want those in the decision phase (yes_filt) that are more recent than the staff phase (~yes_filt)
        yes_filt = group[filt_col] == filt_val
        round_filt = group['Round'] > 0
        max_no_date = group.loc[~yes_filt&round_filt, filt_date_col].max()
        filt_group = group[yes_filt&(group[filt_date_col] >= max_no_date)].copy()

        if filt_group[check_col].isin(check_list).any():
            group[change_col] = new_val
        else:
            if old_val is not None:
                group[change_col] = old_val
            else:
                group[change_col] = group[change_col]
        return group

    def check_decision_filt(group, check_col, change_col, check_list, new_val='Yes',
                        filt_date_col= 'Date Status', filt_col='Phase', filt_val='Decision', old_val=None):
        return check_action_filt(group, check_col, change_col, check_list, new_val,
                        filt_date_col, filt_col, filt_val=filt_val, old_val=old_val)

    def Is_Decision_Task(group, change_col, new_val, old_val):
        return check_decision_filt(
            group, check_col='Task', 
            change_col=change_col, check_list=public_dec_t, new_val=new_val, old_val=old_val)

    def Is_Staff_TaskStatus(group, change_col, new_val):
        return check_decision_filt(
            group, check_col='Task Status', 
            change_col=change_col, check_list=staff_dec_ts, new_val=new_val)
    
    def In_Public_TaskStatus(group, change_col, new_val):
        return check_decision_filt(
            group, check_col='Task Status', 
            change_col=change_col, check_list=public_dec_ts, new_val=new_val)

    prep_permit_df['Staff Decision'] = 'No'
    prep_permit_df['Public Decision'] = 'No'

    prep_permit_df.loc[prep_permit_df['Permit Type'].isin(staff_decision_permit_types), 'Staff Decision'] = 'Yes'

    change_order = [('Staff Decision', 'No', 'Yes'), ('Public Decision', 'Yes', 'No')]
    for (change_col, new_val, old_val) in change_order:
        prep_permit_df = prep_permit_df \
            .groupby('P_ID') \
            .apply(Is_Decision_Task, 
                change_col, new_val, old_val)

    change_order = [('Staff Decision', 'Yes'), ('Public Decision', 'No')]
    for (change_col, new_val) in change_order:
        prep_permit_df = prep_permit_df \
            .groupby('P_ID') \
            .apply(Is_Staff_TaskStatus, change_col, new_val)
        
    change_order = [('Staff Decision', 'No'), ('Public Decision', 'Yes')]
    for (change_col, new_val) in change_order:
        prep_permit_df = prep_permit_df \
            .groupby('P_ID') \
            .apply(In_Public_TaskStatus, change_col, new_val)

    prep_permit_df.loc[prep_permit_df['Permit Type'].isin(pub_hearing_permit_types), 'Public Decision'] = 'Yes'

    prep_permit_df.drop(columns=['Task Status'], inplace=True)

    return prep_permit_df

#clean_addresses(raw_permit_df)['Address'].drop_duplicates().sort_values()

# 3. Main Day & Round Processing
#### 3.1 Calculate Days & Rounds Functions

def check_days(srt, ned, a_list, t_list):
    s = [ss for ss in sum_days(srt, ned, business_days=True) if ss not in t_list]
    t_list.update(s)
    a_list.append(len(s))
    return a_list, t_list

def get_rec_list(rec, formatted_permits, stat_d = PLAN_30):
    # Sort by the date & minute they were updated in Accela
    ##temp = formatted_permits.xs(rec).sort_values(by=sorts).copy()
    temp = formatted_permits.xs(rec).copy()
    
    # then drop it
    temp.drop('Updated Date', inplace=True, axis=1)
    
    #create 
    temp.index = temp.index.map(stat_d)
    temp.reset_index(inplace=True)
    temp.reset_index(inplace=True)
    temp.rename(columns={'index':index}, inplace=True)
    
    check_fields = ['Task Status', index, 'Date Assigned', 'Date Status']
    ts_df = temp[check_fields].copy()
    ts = temp.loc[~temp['Task Status'].isna()][check_fields].values
    if len(ts) == 0:
        ts = temp.loc[temp[index]==0][check_fields].values     
    return ts_df, ts

def round_check(test_df, test, dept='Planning'):
    if len(np.where(test[:,0] == 'End')[0]) > 1:
        while len(np.where(test[:,0] == 'End')[0]) > 1:
            false_end = np.where(test[:,0] == 'End')[0][-1]
            test = np.delete(test, false_end, 0)
    elif len(np.where(test[:,0] == 'End')[0]) == 0:
        test[-1,0] = 'End'

    if (len(test) == 1) and (len(test) != len(test_df)): 
        if test[0,1] > 0:
            start_date = pd.to_datetime(test_df.loc[test_df[index] == 0, 'Date Assigned'].values[0])
            route_date = pd.to_datetime(test_df.loc[test_df[index] == 0, 'Date Status'].values[0])
            test = np.vstack([np.array(['Start', 0, start_date, route_date]), test])
            n = True
    else:
        n = False
    round_check = {
        'Start':1,
        'Round_Start':1,
        'Round_End':0,
        'End':0
    }
    
    end_gap = False
    if (dept != 'Building'):
        if(test[-2,0] == 'Round_End')&(test[-1,0] == 'End'):
            end_gap = True
    
    check = [round_check[l] for l in list(test[:,0])]

    f = 0
    for e,c in enumerate(check):
        if e == 0:
            if (c == 0)&(len(check)>1):
                test = np.insert(test, e + f, np.array(('temp_start', 
                                                            test[e+f,1], 
                                                            test[e+f,2], 
                                                            test[e+f,2])), 
                                     0)  
                f += 1
            pass
        else:
            if c != cc:
                pass
            elif c == cc:
                if c == 0:
                    test = np.insert(test, e + f, np.array(('temp_start', 
                                                            test[e+f,1], 
                                                            test[e+f,2], 
                                                            test[e+f,2])), 
                                     0)  
                    f += 1
                elif (c == 1)and(n==False):
                    #try:
                    pindex = test[e+f,1]
                    #except:
                    #    print(false_index)
                    if (pindex > 2)&(pindex-2 != test[e+f-1,1]): 
                        false_index = pindex-2
                    else:
                        false_index = pindex-1
                    try:
                        false_date = pd.to_datetime(test_df.loc[test_df[index] == false_index, 'Date Status'].values[0])
                    except:
                        print(index, test)
                        print(test_df.loc[test_df[index] == index, 'Date Status'])
                        break
                    test = np.insert(test, e + f, np.array(('temp_end', 
                                                                false_index, 
                                                                false_date, 
                                                                false_date)), 
                                         0)
                    #except Exception as exc:
                    #    print(false_index, index, e)
                    #    logger.error(str(exc))

                    f += 1
        ee = e
        cc = c
    if (ee > 0) & (cc == 1):
        pindex = test[-1,1]
        false_index = pindex
        mx = test_df[index].max()-1
        if mx == pindex:
            mx = test_df[index].max()
        false_date = pd.to_datetime(test_df.loc[test_df[index] == mx, 'Date Status'].values[0])
        test = np.insert(test, len(test), np.array(('temp_end', 
                                                false_index, 
                                                false_date, 
                                                false_date)), 
                         0)  
    return test[:,2:], end_gap

def find_round_length(array, startindex):

    if len(array) > 1:
        endindex = startindex + 1
    else:
        endindex = startindex

    startdate = array[startindex]
    enddate = array[endindex]
    round_length = len(sum_days(startdate, enddate, business_days=True))
    return round_length

def describe_rounds(rounds_arr, double_end_problem=False):
    SUBMITTAL_DATE = min(rounds_arr[0])#rounds_arr[0][0]
    ROUTING_DATE = rounds_arr[0][1]
    if (pd.isna(SUBMITTAL_DATE))or(pd.isnull(SUBMITTAL_DATE)):
        SUBMITTAL_DATE = ROUTING_DATE
        print('first blank')
    if (SUBMITTAL_DATE > ROUTING_DATE):
        SUBMITTAL_DATE = ROUTING_DATE 
    if (pd.isna(SUBMITTAL_DATE))or(pd.isnull(SUBMITTAL_DATE)):
        SUBMITTAL_DATE = rounds_arr[1][1]
        print('second blank')
    END_DATE = max(rounds_arr[-1]) #rounds_arr[-1][1]
        
    
    DAYS_TILL_ROUTING = len(sum_days(SUBMITTAL_DATE, ROUTING_DATE, business_days=True))-1
    
    TOTAL_BIZ_DAYS = len(sum_days(SUBMITTAL_DATE, END_DATE, business_days=True))
    TOTAL_DAYS = len(sum_days(SUBMITTAL_DATE, END_DATE, business_days=False))

    updates = rounds_arr[:,1]

    if ((len(updates) % 2) != 0)and(len(updates)>1):
        print('not even', len(updates))
        
    DAYS_PER_ROUND = [find_round_length(updates, i) for i in range(0,len(updates), 2)]

    if double_end_problem == True:
        del DAYS_PER_ROUND[-1]
    
    ROUNDS_COUNT = len(DAYS_PER_ROUND)

    PROCESSING_DAYS = sum(DAYS_PER_ROUND)
    
    FINAL_DAYS_PER_ROUND = ', '.join([str(d) for d in DAYS_PER_ROUND])

    PROCESSED_LIST = [TOTAL_DAYS, TOTAL_BIZ_DAYS, PROCESSING_DAYS, ROUNDS_COUNT, DAYS_TILL_ROUTING, FINAL_DAYS_PER_ROUND, SUBMITTAL_DATE, END_DATE]

    return PROCESSED_LIST, rounds_arr[:,1]

#### 3.2 Run Calculations


def get_counts(formatted_permits, stat_d = PLAN_30):
    temprecords = np.array(keep_fields).reshape(1,len(keep_fields))
    
    error_ids = []
    
    amts = set(formatted_permits.index.get_level_values(0))
    rounds_dict = {}
    for record_id in amts:
        try:
            ts_df, ts = get_rec_list(record_id, formatted_permits, stat_d=stat_d)
            temprounds, end_check = round_check(ts_df, ts)

            double_end_problem = False

            described_list, rounds = describe_rounds(temprounds, double_end_problem=double_end_problem)

            new_row = [record_id] + described_list
            temprecords = np.vstack([temprecords, new_row])
            rounds_dict[record_id] = rounds
        except IndexError:
            error_ids.append(record_id)
            continue
    records_df = pd.DataFrame(columns=temprecords[0,1:], data=temprecords[1:,1:], index=temprecords[1:,0])
    records_df.index.name = 'Application #'
    records_df[sum_columns] = records_df[sum_columns].astype(int)
    
    print('these ids had an index error', error_ids)
    
    return records_df, rounds_dict

# 4. Attach & Export

## 4.1 Board Counts

def get_comm_days(
        p_permit_df,
        final_day_dict,
        board_tasks = [
                    'Planning Commission', 'Design Review Commission',
                    'City Council', 'Zoning Administrator'
                ],
        start_date_col = 'Date Status',
        p_id_col = 'P_ID'
        ):
    
    board_ids = p_permit_df.loc[p_permit_df['Task'].isin(board_tasks), p_id_col].unique()
    temp_bids = p_permit_df.loc[(p_permit_df['Status']=='Set for Hearing')&(p_permit_df['Task']=='Staff Analysis'), p_id_col].unique()
    board_ids = list(set(list(temp_bids)+list(board_ids)))
    board_ids = [l for l in board_ids if l in final_day_dict.keys()]

    p_permit_df = make_dates(p_permit_df)

    def get_start_date(pid):
        tdf = p_permit_df.loc[(p_permit_df[p_id_col]==pid)]
        hearing_name = 'hearing'
        if 'Staff Analysis' in tdf['Task'].values:
            sat = tdf.loc[tdf['Task']=='Staff Analysis', 'Status'].values[0]
            if hearing_name in sat.lower(): 
                return tdf.loc[tdf['Task']=='Staff Analysis', start_date_col].max()
        
        return p_permit_df.loc[(p_permit_df[p_id_col]==pid)&(p_permit_df['Task'].isin(board_tasks)), start_date_col].min()

    def get_end_date(pid):
        return p_permit_df \
            .loc[(p_permit_df[p_id_col]==pid)&(p_permit_df['Task'].isin(board_tasks)), start_date_col].max()

    board_d = {pid:{
        'id':pid,
        'start_date':get_start_date(pid),
        'end_date':get_end_date(pid),
        'final_date':final_day_dict[pid]
    } for pid in board_ids}

    same_end = [pid for pid in board_ids if board_d[pid]['start_date']==board_d[pid]['end_date']]

    print(board_d)

    for pid, pdict in board_d.items():
        sdate = pdict['start_date']
        edate = pdict['end_date']
        fdate = pdict['final_date']
        if edate > fdate:
            edate = fdate

        tot_days = len(sum_days(sdate, edate, business_days=False))
        biz_days = len(sum_days(sdate, edate, business_days=True))

        board_d[pid].update({
            'total_days':tot_days,
            'business_days':biz_days
        })
        
    p_permit_df['Board Total Days'] = 0
    p_permit_df['Board Business Days'] = 0
    p_permit_df['Board Start Date'] = np.NaN
    p_permit_df['Board End Date'] = np.NaN

    board_filt = p_permit_df[p_id_col].isin(board_ids)

    p_permit_df.loc[board_filt, 'Board Decision'] = 'True'
    p_permit_df.loc[board_filt, 'Board Total Days'] = p_permit_df.loc[board_filt, p_id_col].map(lambda x: board_d[x]['total_days'])
    p_permit_df.loc[board_filt, 'Board Business Days'] = p_permit_df.loc[board_filt, p_id_col].map(lambda x: board_d[x]['business_days'])
    
    p_permit_df.loc[board_filt, 'Board Start Date'] = p_permit_df.loc[board_filt, p_id_col].map(lambda x: board_d[x]['start_date'])
    p_permit_df.loc[board_filt, 'Board End Date'] = p_permit_df.loc[board_filt, p_id_col].map(lambda x: board_d[x]['end_date'])
    
    p_permit_df[['Board Start Date','Board End Date']] = p_permit_df[['Board Start Date','Board End Date']].astype('datetime64[ns]')
    
    return p_permit_df



## NEW

def preview_row(ind, per_df, check_df=pd.DataFrame(), pids = []):
    if type(ind) == str:
        focus_id = ind
    else:
        focus_id = pids[ind]
    focus_df = per_df[per_df['P_ID']==focus_id].copy()

    date_cols = [c for c in list(focus_df) if 'date' in c.lower()]
    
    focus_df[date_cols] = focus_df[date_cols] \
        .apply(lambda d: pd.to_datetime(d, format = "%m/%d/%Y %I:%M:%S %p"))
    focus_df.sort_values(by=sorts, inplace=True)

    #focus_df[date_cols] = focus_df[date_cols].apply(lambda x: x.str.split(' ', n=1).str[0])
    #focus_df[date_cols] = focus_df[date_cols].apply(lambda x: x.strftime('%m/%d/%Y'))
    
    focus_df['T_ID'] = focus_df['T_ID'].str.split('-').str[-1]

    focus_row = focus_df[['P_ID', 'Entitlements']].values[0]
    print(*focus_row, sep=' | ')

    if len(check_df) > 0:
        check_df.reset_index(inplace=True)
        fr = check_df[check_df['Application #']==focus_id] \
            [['Total Days', 'Rounds', 'Days per Round']].values[0]
        
        print("Total {} | Rounds {} | Days per Round {}".format(fr[0], fr[1], fr[2]))

    return focus_df \
        [[
            'Task', #'Task Rank', 
            'Status', 'T_ID', 'Date Assigned', 'Date Status', 'Phase', 'Action'
        ]] \
        .set_index(['Phase', 'Task', 'Status'])
