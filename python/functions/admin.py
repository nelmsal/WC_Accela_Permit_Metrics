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