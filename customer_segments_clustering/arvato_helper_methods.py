import numpy as np
import pandas as pd

#perform all pre-processing and cleaning steps in one function call
def clean_data(df, missing_values):
    """
    Perform feature trimming, re-encoding, and engineering for demographics
    data
    
    INPUT: Demographics DataFrame
    OUTPUT: Trimmed and cleaned demographics DataFrame
    """
    
    # Put in code here to execute all main cleaning steps:
    # convert missing value codes into NaNs, ...
    columns = df.columns

    #iterate through missing values list
    for i, item in enumerate(missing_values):
        for element in item:
            #then there are no missing values
            if (element == None):
                continue
            
            #else there are missing values and we need to replace them
            else:
                #replace custom encoded values for each column in df
                df[columns[i]].replace(element, np.nan, inplace=True)
    
    
    #drop all unnecessary columns up front
    to_drop = ['AGER_TYP', 'GEBURTSJAHR', 'TITEL_KZ', 'ALTER_HH', 'KK_KUNDENTYP',
               'KBA05_BAUMAX', 'LP_STATUS_FEIN', 'CAMEO_DEU_2015', 'CAMEO_DEUG_2015', 
               'CJT_GESAMTTYP', 'FINANZTYP', 'GFK_URLAUBERTYP', 'ZABEOTYP', 'GEBAEUDETYP',
              'LP_FAMILIE_GROB', 'LP_STATUS_GROB', 'SHOPPER_TYP', 'LP_FAMILIE_FEIN']

    
    for col in to_drop:
        df.drop(col, axis=1, inplace=True)

    #calculate row and columns
    rows, cols = df.shape
    missing_per_row = df.isnull().sum(axis=1) / cols * 100.0
    
    #only keep rows with less missing data data
    sub_idx = np.where(missing_per_row == 0)[0].tolist()
    df_sub = df.copy().iloc[sub_idx, :]
    
    
    #process OST_WEST_KZ column, then drop the categorical column
    df_sub = pd.concat([df_sub, pd.get_dummies(df_sub['OST_WEST_KZ'], prefix = 'OST_WEST_KZ')], axis=1)
    df_sub.drop('OST_WEST_KZ', axis=1, inplace=True)
    
    #get dummy variables for the rest of the categorical columns, leaving alone binary categorical columns
    cat_dummies = ['NATIONALITAET_KZ']
    
    for col in cat_dummies:
        df_sub = pd.concat([df_sub, pd.get_dummies(df_sub[col], prefix=col)], axis=1)
        df_sub.drop(col, axis=1, inplace=True)
    
    
    #now process mixed value columns
    mix_cols = ['LP_LEBENSPHASE_FEIN', 'LP_LEBENSPHASE_GROB', 'PRAEGENDE_JUGENDJAHRE', 
                'WOHNLAGE', 'CAMEO_INTL_2015', 'PLZ8_BAUMAX']
    
    
    decade_copy = df_sub['PRAEGENDE_JUGENDJAHRE'].copy()

    #specify conditions
    conditional_40 = decade_copy <= 2
    conditional_50 = np.logical_and(decade_copy > 2, decade_copy <= 4)
    conditional_60 = np.logical_and(decade_copy > 4, decade_copy <= 7)
    conditional_70 = np.logical_and(decade_copy > 7, decade_copy <= 9)
    conditional_80 = np.logical_and(decade_copy > 9, decade_copy <= 13)
    conditional_90 = decade_copy > 13

    decade_copy[conditional_40] = 40
    decade_copy[conditional_50] = 50
    decade_copy[conditional_60] = 60
    decade_copy[conditional_70] = 70
    decade_copy[conditional_80] = 80
    decade_copy[conditional_90] = 90

    #create new column here for decade
    df_sub['PRAEGENDE_JUGENDJAHRE_decade'] = decade_copy

    #mainstream and avantgarde values, from Data_Dictionary.md
    main_idx = [1, 3, 5, 8, 10, 12, 14]
    avant_idx = [2, 4, 6, 7, 9, 11, 13, 15]

    movement = df_sub['PRAEGENDE_JUGENDJAHRE'].copy()

    #let's encode the movement to be 1 if mainstream, 0 if avantgarde
    move = []

    for value in movement:
        if value in main_idx:
            move.append(1)
        else:
            move.append(0)

    #create new column for movement
    df_sub['PRAEGENDE_JUGENDJAHRE_move'] = move
    
        
    #convert cameo_intl_2015 column into two columns
    #one for the ten's place, the other for the one's place
    df_sub['CAMEO_INTL_2015_tens'] = df_sub['CAMEO_INTL_2015'].copy().astype(int) // 10
    df_sub['CAMEO_INTL_2015_ones'] = df_sub['CAMEO_INTL_2015'].copy().astype(int) % 10

    
    #get dummy variables for interesting mixed columns
    mix_cols_keep = ['LP_LEBENSPHASE_GROB', 'WOHNLAGE']
    
    for col in mix_cols_keep:
        df_sub = pd.concat([df_sub, pd.get_dummies(df_sub[col], prefix=col)], axis=1)

    for col in mix_cols:
        df_sub.drop(col, axis=1, inplace=True)
    
    #return the cleaned dataframe.
    return df_sub
