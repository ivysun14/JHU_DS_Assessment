import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from bs4 import BeautifulSoup

# setting
pd.set_option("display.max_rows", 500)

# fetch the page
url = 'https://en.wikipedia.org/wiki/List_of_natural_disasters_by_death_toll'
resp = requests.get(url)
soup = BeautifulSoup(resp.content, 'lxml')

tables_by_century = {}

# search through <h3> tags to find 20th and 21st century all cause disasters tables
h3_tags = soup.find_all('h3')
for h3 in h3_tags:
     # grab id attribute
    h3_id = h3['id'].lower()
    if h3_id == '20th_century' or h3_id == '21st_century':
        # look for the next <table> that follows this heading
        table = h3.find_next('table', {'class': 'wikitable'})
        century = '20th' if h3_id == '20th_century' else '21st'
        tables_by_century[century] = pd.read_html(str(table))[0]

# correct one mismatch in column names ('Death tolls' and 'Death toll')
tables_by_century['21st'].columns = tables_by_century['20th'].columns

# merge dataframes (127 x 6)
df = pd.concat([tables_by_century['20th'], tables_by_century['21st']], ignore_index=True)

#print(df.dtypes)
def parse_death_toll(value):
    
    # case where there is an NA entry
    if pd.isna(value):
        return np.nan

    # 1). remove annotations ([~] etc.)
    value = re.sub(r'\[.*?\]', '', value)
    
    # 2). remove commas in numbers
    value = re.sub(r',', '', value)
    
    # 3). take midpoint if a range is given (e.g. 6000–12000)
    if '–' in value or '-' in value:
        parts = re.split('–|-', value)
        try:
            low = int(parts[0])
            high = int(parts[1])
            return (low + high) // 2
        except:
            return np.nan # there is one edge case here with both range and "+" (row #22)
    
    # 4). take the bound when '+' is given (e.g. 20000+)
    if value.endswith('+'):
        try:
            return int(value[:-1])
        except:
            return np.nan

    # Plain integer value
    try:
        return int(value)
    except:
        return np.nan

# parse death toll values
df['Death tolls'] = df['Death tolls'].apply(parse_death_toll)

print(df)

# plot death toll vs. year colored by disaster type
plt.figure(figsize=(15, 8))
sns.set_style("whitegrid")
sns.scatterplot(
    data=df,
    x='Year',
    y='Death tolls',
    hue='Type',
    palette='tab10',
    s=90,
    edgecolor='black'
)
plt.title("Natural Disaster Death Tolls by Year", fontsize=16)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Estimated Death Toll", fontsize=12)
plt.legend(title='Disaster Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('Q1.png')