import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import requests
from bs4 import BeautifulSoup


def clean_columns_1(df):
    
    '''
    This function cleans the column titles by replacing whitespaces with underscores and converting everything into lowercase.
    '''
    
    df.columns = df.columns.str.lower()

    df.rename(columns={'dr_no': 'file_number', "premis": "premise"}, inplace=True)


def clean_columns_2(df):
    
    '''
    This function cleans the column titles by replacing whitespaces with underscores and converting everything into lowercase.
    It also drops extra columns that we don't need.
    It renames column titles to match the ones in the first dataset.
    '''
    
    #We clean the column titles so they are standardized.
    df.columns = df.columns.str.replace(" ", "_").str.lower()
    df.drop(["area_", "rpt_dist_no", 
             "part_1-2", "crm_cd", 
             "mocodes", "premis_cd", 
             "weapon_used_cd", 
             "status", 
             "crm_cd_1", 
             "crm_cd_2", 
             "crm_cd_3", 
             "crm_cd_4", 
             "cross_street"], 
            axis="columns", 
            inplace=True)
    #We rename two of the columns appropriately.
    df.rename(columns={'dr_no': 'file_number', 
                       "date_rptd": "date_reported", 
                       "date_occ": "date_occured", 
                       "time_occ": "time_occured", 
                       "area_name": "area", 
                       "crm_cd_desc": "crime_code", 
                       "vict_age": "victim_age", 
                       "vict_sex": "victim_sex", 
                       "vict_descent": "victim_descent", 
                       "premis_desc": "premise", 
                       "weapon_desc": "weapon", 
                       "status_desc": "status"}, 
              inplace=True)
    

def descent_cleaning(df):
    
    '''
    This function renames the descent so that the names are easier to understand.
    It also replaces the null values with "not_specified".
    '''
    
    descent_mapping = {
        'W': 'white', 
        'B': 'black', 
        'H': 'hispanic', 
        'A': 'asian', 
        'O': 'other', 
        'X': 'not_specified', 
        'I': 'american indian', 
        'P': 'pacific islander',
        "C": "chinese",
        "D": "cambodian",
        "F": "filipino",
        "G": "guamanian",
        "J": "japanese",
        "K": "korean",
        "L": "laotian",
        "S": "samoan",
        "U": "hawaiian",
        "V": "vietnamese",
        "Z": "asian indian",
        "-": "not_specified"

    }
        
    df["victim_descent"] = df["victim_descent"].map(descent_mapping)

    df["victim_descent"].fillna("not_specified", inplace=True)
  
    
def sex_cleaning(df):
    
    '''
    This function replaces the null values for "not_specified".
    It also replaces wrongly inout values for "not_specified".
    '''
    
    df["victim_sex"].fillna('not_specified', inplace=True)
    df["victim_sex"].replace({"H": "not_specified", "-": "not_specified", "X": "not_specified", "N": "not_specified"}, inplace=True)

    
def age_cleaning(df):
    '''
    This function replaces the wrongly input data with a 0, which in this case represents a null (not specified).
    '''
    df["victim_age"].replace([-1, -2, -3, -4, 120], 0, inplace=True)

    
def age_cleaning_2(df):
    
    '''
    This function replaces the wrongly input data with a 0, which in this case represents a null (not specified).
    '''
    
    df["victim_age"].replace([-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, 120, 114, 118], 0, inplace=True)
    
    
def premise_cleaning(df):
    
    '''
    This function drops the null values in column premise because they are insignificant compared to the size of the dataset.
    '''
    
    df.dropna(subset="premise", inplace=True)

    
def weapon_cleaning(df):
    
    '''
    This function replaces the null values for "not_specified". They represent a significant part of the dataset so we cant drop them.
    '''
    
    df["weapon"].fillna("not_specified", inplace=True)
    

def date_cleaning(df):
    
    '''
    This function cleans the date and type columns.
    It converts the date columns to datetime type.
    It also creates aditional columns by splitting the date column so we have the day, month and year each in a seperate column.
    '''
    
    #we use zfill() to add a 0 in the incorrect inputs, example 815 => 0815
    df.loc[:, 'time_occured'] = df['time_occured'].astype(str).str.zfill(4)
    # Insert a colon between the hour and minute parts
    df.loc[:, 'time_occured'] = df['time_occured'].str[:2] + ':' + df['time_occured'].str[2:]
    #we convert the column to a date time
    # Specify the format for the date conversion
    df.loc[:, 'date_reported'] = pd.to_datetime(df['date_reported'], errors = "coerce").dt.date 
    df['date_reported'] = pd.to_datetime(df['date_reported'])
    #We splited the columns date_repoted into 3 columns year / month / day
    df['dr_year'] = df['date_reported'].dt.year
    df['dr_month'] = df['date_reported'].dt.month
    df['dr_day'] = df['date_reported'].dt.day
    #we convert the column to a date time
    # Specify the format for the date conversion
    df.loc[:, 'date_occured'] = pd.to_datetime(df['date_occured'], errors = "coerce").dt.date 
    df['date_occured'] = pd.to_datetime(df['date_occured'])
    #We splited the columns date_occured into 3 columns year / month / day
    df['do_year'] = df['date_occured'].dt.year
    df['do_month'] = df['date_occured'].dt.month
    df['do_day'] = df['date_occured'].dt.day
    
     
def weapon_class(weapon):
    
    '''
    This function uses regex to group the weapons used in crimes into broad categories so it will be easier to visualize.
    '''
    
    weapon_lower = weapon.lower()
    if re.search(r'handgun|revolver', weapon_lower):
        return 'handgun'
    elif re.search(r'shotgun', weapon_lower):
        return 'shotgun'
    elif re.search(r'rifle|semiautomatic|assault|automatic', weapon_lower):
        return 'rifle'
    elif re.search(r'knife|blade|razor|scissors|dagger|cutting', weapon_lower):
        return 'blade'
    elif re.search(r'cleaver|sword|machete|axe', weapon_lower):
        return 'long_blade'
    elif re.search(r'hammer|brass|board|blackjack|pipe|tire|club|bat|stick|blunt', weapon_lower):
        return 'blunt_weapon'
    elif re.search(r'hand|fist|presence', weapon_lower):
        return 'bare_hands'
    elif re.search(r'not_specified|unknown', weapon_lower):
        return 'not_specified'
    else:
        return "other"

    
def crime_categories(crime_code):
    
    '''
    This function uses regex to group the types of crimes commited into broad categories so it will be easier to visualize.
    '''
    
    crime_lower = crime_code.lower()
    
    if re.search(r'theft|burglary|shoplifting|vehicle|identity|pickpocketing|purse|stolen|bunco|embezzlement|card|innkeeper|coin|forgery|till|robbery|pickpocket|DWOC|computer|extortion', crime_lower):
        return 'theft_related_offense' 
    elif re.search(r'assault|battery|animals|rape|human|homicide|manslaughter|threats|weapon|abandonment|stealing|neglect|reckless|driving|kidnapping|abuse|pornography|stalking|partner', crime_lower): 
        return 'violent_crime'
    elif re.search(r'sodomy|crm|oral|pimping|peeping|sex|CRM|pandering|lewd|sexual|incest|beastiality|exposure|annoying|penetration', crime_lower): 
        return 'sex_crime'
    elif re.search(r'vandalism|scare|court|drunk|bombing|weapons|firearms|arson|trespassing|dumping|shots', crime_lower): 
        return 'property_crime'
    elif re.search(r'bribery|false|conspiracy|counterfeit|false', crime_lower): 
        return 'fraud'
    elif re.search(r'peace|prowler|wrecking|riot|disperse|order|arrest|yield|door|phone', crime_lower): 
        return 'public_order_offense'
    elif re.search(r'trafficking|abortion|miscellaneous|bigamy|contributing|drug|worthless|school|lynching', crime_lower): 
        return 'miscellaneous_crime'            
    else:
        return (crime_code)    
    

def premise_class(premise):
    
    '''
    This function uses regex to group the premises where crimes were commited into broad categories so it will be easier to visualize.
    '''
    
    premise_lower = premise.lower()
    if re.search(r'tram|aircraft|mta|train|metro|taxi|bus|airport|truck|delivery|vehicle|station', premise_lower):
        return 'public_transport'    
    elif re.search(r'condominium|hotel|shed|apartment|balcony|residential|dwelling|building|residence|motel|home|housing|house', premise_lower):
        return 'residential_area'    
    elif re.search(r'store|laundromat|rental|goods|dealership|connection|thru|sales|appliance|mortuary|market|mart|wash|bar|restaurant|shop|supply', premise_lower):
        return 'commercial_space'    
    elif re.search(r'school|college|care', premise_lower):
        return 'educational_facilities'    
    elif re.search(r'hospital|health|parlor|salon|hospice|clinic|medical', premise_lower):
        return 'healthcare_facilities'    
    elif re.search(r'park|club|museum|coliseum|commercial|tow|center|movie|theater|arcade|stadium|rink|music|cultural|entertainment|monument|sports|bowling|golf|pool|beach', premise_lower):
        return 'leisure_area'    
    elif re.search(r'library|jail|defense|public|police|fire|government|post', premise_lower):
        return 'government_facilities' 
    elif re.search(r'bank|savings|financial|finance|check|atm|union', premise_lower):
        return 'financial_institutions'     
    elif re.search(r'church|worship|mosque|temple', premise_lower):
        return 'religious_facilities'       
    elif re.search(r'plant|factory|refinery|facility|manufacturer|telecommunication|manufacturing', premise_lower):
        return 'industrial_facilities'     
    elif re.search(r'website|cyberspace', premise_lower):
        return 'cyber_space'   
    elif re.search(r'street|sidewalk|garage|freeway|encampment|valet|elevator|stairwell|patio|vacant|driveway|phone|pedestrian|mail|river|alley|shelter|escalator|cemetary|bridge|dam|tunnel|mass|trash|dock|construction|court', premise_lower):
        return 'public_place'       
    elif re.search(r'not_specified|unknown', premise_lower):
        return 'not_specified'
    else:
        return "other"

    
def categorize_cleaned_time(time):
    
    """
    This function categorizes times into time slots for better visualization
    """
    
    for item in range(len(time)):
        if pd.isnull(time):
            return 'Unknown'
        elif time >= "05:00" and time < "08:00": 
            return '05:00-08:00'
        elif time >= "08:00" and time < "12:00":
            return "08:00-12:00"
        elif time >= "12:00" and time < "15:00": 
            return '12:00-15:00'
        elif time >= "15:00" and time < "17:00":
            return "15:00-17:00"
        elif time >= "17:00" and time < "20:00":
            return '17:00-20:00'
        else:
            return '20:00-05:00'
        
        
def chief_info():
    
    '''
    This function webscrapes wikipedia for basic info about the LAPD police chief.
    '''
    
    chief_url= "https://en.wikipedia.org/wiki/Dominic_Choi"

    chief_response = requests.get(chief_url)
    chief_response
    
    chief_soup = BeautifulSoup(chief_response.content)
    
    chief_dict = {}
    
    name = chief_soup.find("div", class_ = "fn").get_text()
    
    chief_dict["Name"] = name
    
    for item in range(16):
        try:
            info_key = chief_soup.find_all("th", class_ = "infobox-label")[item].get_text().replace("\xa0", "")
            info_value = chief_soup.find_all("td", class_ = "infobox-data")[item].get_text()
            if info_key != "Mayor":
                chief_dict[info_key] = info_value
        except:
            pass
    
    for key, values in chief_dict.items():
        print(f"{key}: {values}")
        

def mayor_info():
        
    '''
    This function webscrapes wikipedia for basic info about the mayor of LA.
    '''
    
    mayor_url = "https://en.wikipedia.org/wiki/Mayor_of_Los_Angeles"

    mayor_response = requests.get(mayor_url)
    
    mayor_soup = BeautifulSoup(mayor_response.content)
    
    mayor_dict = {}
    
    name = mayor_soup.find("a", title = "Karen Bass").get_text()
    
    mayor_dict["Name"] = name
    
    for item in range(16):
        try:
            info_key = mayor_soup.find_all("th", class_ = "infobox-label")[item].get_text().replace("\xa0", "")
            info_value = mayor_soup.find_all("td", class_ = "infobox-data")[item].get_text()
        
            mayor_dict[info_key] = info_value
        except:
            pass
        
    for key, values in mayor_dict.items():
        print(f"{key}: {values}")
        

def president_info():
        
    '''
    This function webscrapes wikipedia for basic info about the President of the United States.
    '''
    
    president_url = "https://en.wikipedia.org/wiki/President_of_the_United_States"

    president_response = requests.get(president_url)
    
    president_soup = BeautifulSoup(president_response.content)
    
    president_dict = {}
    
    name = president_soup.find("a", title = "Joe Biden").get_text()
    
    president_dict["Name"] = name
    
    for item in range(16):
        try:
            info_key = president_soup.find_all("th", class_ = "infobox-label")[item].get_text().replace("\xa0", "")
            info_value = president_soup.find_all("td", class_ = "infobox-data")[item].get_text()
        
            president_dict[info_key] = info_value
        except:
            pass
        
    for key, values in president_dict.items():
        print(f"{key}: {values}")