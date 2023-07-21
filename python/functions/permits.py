# IMPORT
import pandas as pd
import os
import datetime
import numpy as np

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 

pd.options.mode.chained_assignment = None


# ADMIN FUNCTIONS

def get_holidays(holiday_path=r'data\raw_data\City_Holidays.xlsx'):
    return set([
    pd.Timestamp(h) 
    for h in pd.read_excel(holiday_path)['date'].unique()
    ])
holidays = get_holidays()

# PLANNER FUNCTIONS

planner_initials = {
    'ABC': 'Alan Carreon',
    'AS': 'Ana Spinola',
    'AMS': 'Andrew Smith',
    'CG': 'Chip Griffin',
    'EB': 'Ethan T Bindernagel',
    'GDK': 'Greg Kapovich',
    'GV': 'Gerardo Victoria',
    'HEC': 'Haley E Croffoot',
    'HH': 'Haley Hubbard',
    'JG': 'Jessica J Gonzalez',
    'JCav': 'Jeanine Cavalli',
    'KN': 'Ken Nodder',
    'OA': 'Ozzy Arce',
    'SKG': 'Simar Gill',
    'SP': 'Sukhamrit S Purewal',
    'TC': 'Trishia Caguiat'
}
planner_names = {v:k for k,v in planner_initials.items()}
planner_names['Haley Hubbard'] = 'HEC'
planner_initials['HH'] = 'Haley E Croffoot'


# ENTITLEMENTS

#Complete_Closed = ['Approved - Closed', 'Closed', 'Not Approved - Closed']
#Planner_Closed = ['Close Out', 'Approved', 'Not Approved']
Ent = {
 'Zoning Amendment': 'Amend',
 'Use Permit Minor': 'MUP',
 'Design Review': 'DR',
 'Design Review Oversized Home': 'DR',
 'Design Review Other': 'DR',
 'Design Review Commercial': 'DR',
 'General Plan Amendment': 'Amend',
 'Design Review Antenna': 'DR',
 'Design Review Residential': 'DR',
 'Use Permit Conditional': 'CUP',
 'Use Permit Administrative': 'AUP',
 'ZCL':'ZCL',
 'Variance': 'Vari',
 'Rezoning': 'ReZone',
 'Tree Dripline Encroachment': 'Tree',
 'Tree Removal Permit': 'Tree',
 '': 'Other',
 'Other': 'Other',
 'Tentative Map Major Subdivision': 'Maj Sub',
 'Drip Line Encroachment': 'Other',
 'Tentative Map Minor Subdivision': 'Min Sub',
 'Tentative Map Condo Conversion': 'Conv',
 'Hillside Performance Standards': 'Other'}
Ent_names = {v:k for k,v in Ent.items()}
Ent_names['Other'] = 'Other'


# PERMIT TYPES

BUILD = {
    'Application Submittal - Route':'Start',
    'Consolidated Comments - With Customer for Response':'Round_End',
    'Resubmittal or Revision - Route':'Round_Start',
    #'Building Review - Notes':'Pause',
    'Ready to Issue - Conditionally Approved':'End',
    'Ready to Issue - Issued':'End',
    'Ready to Issue - Approved':'End'
}

SDP = {
    'Status - Received':'Start',
    'Application Submittal - Route':'Start',
    'Consolidated Comments - With Customer for Response':'Round_End',
    'Consolidated Comments - Resubmittal':'Round_Start',
    #'Consolidated Comments - Ready to Issue': 'Round_Start',
    #'Status - Approved':'End'
    'Ready to Issue - Issue':'End'
    #'Application Submittal - Ready to Issue':'End'
}

PLAN_30 = {
    'Status - Received':'Start',
    'Intake Review - Application Accepted':'Start',
    'Consolidated Comments - Deemed Incomplete':'Round_End',
    'Resubmittal - Route for Review':'Round_Start',
    'Consolidated Comments - Deemed Complete':'End',
    'Staff Analysis - Set for Hearing':'End',
    'Staff Analysis - Staff Level Decision':'End'
}

# decision types
path = r'data\clean_data\entitlement_info.xlsx'
# open xlsx file as df and make first row the column names
ent_df = pd.read_excel(path, header=1) \
    [['Permit Type', 'Entitlement?', 'Public Hearing?']]

pub_hearing_permit_types = \
    ent_df[ent_df['Public Hearing?']=='YES']['Permit Type'].unique()

staff_decision_permit_types = \
    ent_df[ent_df['Public Hearing?']=='NO']['Permit Type'].unique()

# other decision indicators
staff_dec_ts = [
    'Staff Level Decision - Approved'
]
public_dec_ts = [
    'Staff Level Decision - Appealed'
    ]

public_dec_t = [
    'Design Review Commission', 'Planning Commission', 
    'Zoning Administrator', 'City Council'
]
board_dec_t = [
    t for t in public_dec_t if t != 'Zoning Administrator'
]


# BAD PERMITS?
bad = [
    'Y18-031',  # resub after PC decision
    'Y18-036',  # inconsistent starts
    'Y18-072',
    'Y18-013',  # CEQA caused odd resubmittal
    'Y18-039',  # CEQA problem
    'Y18-058',  # resub after PC decision
    'Y18-066',  # missing resubmittal
    'Y18-088',  # missing resubmittal
    'Y19-010',  # resub after PC decision
    'Y19-020',  # appealed then resub
    'Y19-041',  # weird ordering of resubmittal
    'Y19-057',  # resub after staff & PC decision
    'Y19-110',  # resub after PC decision
    'Y19-122',  # resub after PC decision
    'Y19-123',  # resub after PC decision
    'Y19-140',   # withdrawn
    'Y21-011',  # missing resubmittal,  
    'Y21-053',   # weird multi deemed incomplete with DRC in the front
    'Y21-065',  # missing resubmittal
    'Y21-099',  # missing resubmittal
    'Y21-111',
    'Y21-114',  # missing resubmittal
    'Y21-033',  # missing resubmittal
    ''
]
sorta_bad = [
    'Y19-123',
    'Y20-052',
    'Y21-011',   # round end, review, round end
    'Y20-040',   # design review continued?
    'Y19-122',  # review after comm
    
    ''
]