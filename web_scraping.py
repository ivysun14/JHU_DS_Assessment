import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from bs4 import BeautifulSoup

def parse_disaster_type(value):
    '''
    Parse and process disaster type into a unanimous format.
    Input can either by a string representing the id of an h3 tag or 
    the disaster type already labeled in the column 'Type'
    '''
    
    if pd.isna(value):
        return np.nan
    
    value = value.lower()

    # if '_' in type name, replace with ' '
    if '_' in value:
        value = re.sub(r'_', ' ', value)

    # give a unanimous name to types related to landslide/landslip
    if 'avalanche' in value or 'landslip' in value:
        value = 'landslide'

    # give a unanimous name to types related to cyclone/tropical cyclone
    if 'cyclone' in value or 'tropical cyclone' in value:
        value = 'tropical cyclone'
    
    # remove tailing 's' from all types for consistency
    if value[-1] == 's':
        value = value[:-1]
    
    return value


def parse_disaster_type_from_event(value):
    '''
    Parse and process disaster type from an input string describing the event
    '''

    if pd.isna(value):
        return np.nan
    
    # parse event by empty space
    parts = value.split()
    value = parts[-1]
    value = parse_disaster_type(value)
    
    return value


def parse_year(value):
    '''
    Parse and process year of disasters from the 'year' column.
    The input can be either integer or string, and only string values
    will went through the processing. Integer values will be returned
    as they were.

    For disasters with a duration such as famines or epidemics, the end
    year was taken.
    '''

    if pd.isna(value):
        return np.nan
    
    # only process years represented in a string
    if not isinstance(value, int):
        # for disasters with a duration, take the end year
        if '–' in value:
            parts = re.split('–|-', value)
            value = parts[-1]
            # one disease outbreak up to 2024
            if value == 'present':
                value = 2024
            else:
                # remove annotations
                value = re.sub(r'\[.*?\]', '', value)
                value = re.sub(r'\s*\(.*?\)', '', value)
        else:
            # for disasters with specific date, leave only year
            value = value.split()[-1]
    
    return value


def parse_death_toll(value):
    '''
    Parse and process death tolls from the 'death toll' column.
    The input can be either integer or string, and only string values
    will went through the processing. Integer values will be returned
    as they were.

    For death tolls with a range, the midpoint was taken. For death tolls with
    an upper bound, the bound was taken. For those with both the bound was first taken
    then the midpoint was calculated. Any annotations on death toll were not included.
    '''
    
    if pd.isna(value):
        return np.nan
    
    # only process death tolls represented in a string
    if not isinstance(value, int):

        million = False
        if 'million' in value:
            million = True
            value = re.sub(r' million', '', value)
        
        # 1). remove annotations ([-] etc.), commas in numbers, ~
        value = re.sub(r'\[.*?\]', '', value)
        value = re.sub(r'\s*\(.*?\)', '', value)
        value = re.sub(r',|~', '', value)
        # standardize hyphens
        value = value.replace('−', '-')

        # 2). take the bound when '+' is given (e.g. 20000+)
        if value.endswith('+'):
            value = value[:-1]
    
        # 3). take midpoint if a range is given (e.g. 6000–12000)
        if '–' in value or '-' in value:
            parts = re.split('–|-', value)
            low = int(parts[0])
            high = int(parts[1])
            value = (low + high) // 2
            if million:
                value = value * (10**6)
                million = False

        # 4). add million to death toll if needed
        if million:
            value = float(value) * (10**6)

    return int(value)

## ===================================================

# setting
pd.set_option("display.max_rows", 500)

# fetch all tables on the page
url = 'https://en.wikipedia.org/wiki/List_of_natural_disasters_by_death_toll'
resp = requests.get(url)
soup = BeautifulSoup(resp.content, 'lxml')
tables = soup.find_all('table', {'class': 'wikitable'}) # 16 tables total

# 1. read in tables as pandas df, assign a 'Type' column if not present, and process disaster types
tables_by_id = {}
for table in tables:
    tag = table.find_previous('h3')
    # all tables except the first one comes after an h3 tag
    if tag:
        title = tag['id']
        tables_by_id[title] = pd.read_html(str(table))[0]
        # add disaster type
        if 'Type' not in tables_by_id[title].columns:
            tables_by_id[title]['Type'] = title
        tables_by_id[title]['Type'] = tables_by_id[title]['Type'].apply(parse_disaster_type)
    else:
        # name the first table as "top_ten"
        tables_by_id['top_ten'] = pd.read_html(str(table))[0]
        # add disaster type from event
        tables_by_id['top_ten']['Type'] = tables_by_id['top_ten']['Event'].apply(parse_disaster_type_from_event)

# 2. align column names for dataframes, leave only selected columns for further processing
for key, df in tables_by_id.items():
    df.columns = df.columns.str.lower()
    if 'year' in df.columns and 'date' in df.columns:
        df.drop('date', axis=1, inplace=True)
    df.rename(
        columns={col: 'death toll' for col in df.columns if 'death toll' in col},
        inplace=True
    )
    df.rename(
        columns={col: 'year' for col in df.columns if 'date' in col},
        inplace=True
    )
    tables_by_id[key] = df[['year', 'death toll', 'event', 'type']]

# 3. merge into one df (258 x 4)
df = pd.concat(tables_by_id.values(), ignore_index=True)

# 4. process years, filter for 20th and 21st century disasters (198 x 4)
df['year'] = df['year'].apply(parse_year)
df = df[df["year"] != 'BC']
df['year'] = df['year'].astype(int)
df = df[(df['year'] >= 1900) & (df['year'] <= 2025)]
df = df.sort_values(by="year").reset_index(drop=True)

# 5. process death tolls
df['death toll'] = df['death toll'].apply(parse_death_toll)

# 6. filter redundant disasters (177 x 4)
df['event'] = df['event'].str.lower()
df['type'] = df['type'].str.lower()
df = df.drop_duplicates(subset=['year', 'event', 'type'], keep='first')
df = df.reset_index(drop=True)

# plot death toll vs. year colored by disaster types
plt.figure(figsize=(15, 8))
sns.set_style("whitegrid")
sns.scatterplot(
    data=df,
    x='year',
    y='death toll',
    hue='type',
    palette='tab20',
    s=80,
    edgecolor='black'
)
plt.title("Natural Disaster Death Tolls by Year", fontsize=16)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Estimated Death Toll", fontsize=12)
plt.legend(title='Disaster Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('Q1.png')

#max_row = df.loc[df['death toll'].idxmax()]
#print(max_row)