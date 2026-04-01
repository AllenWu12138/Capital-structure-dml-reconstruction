import pandas as pd
import numpy as np
from pathlib import Path

# PATH
BASE_DIR = Path('/Users/wulingzhou/VS_code_for_Python/capital-structure-dml-replication')
RAW_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
PROCESSED_DIR.mkdir(parents=True,exist_ok=True)

# clean CPI
def clean_cpi(raw_dir,processed_dir):
    cpi = pd.read_csv(raw_dir / 'CPI.csv')
    print(cpi.shape)
    print(cpi.columns)
    print(cpi.head())
    required_cols = ['TIME_PERIOD','REF_AREA','OBS_VALUE','Transformation']
    assert all(col in cpi.columns for col in required_cols),'missing required columns'
    cpi = cpi[cpi['Transformation']!='Not applicable'].copy()
    cpi = cpi.rename(columns={
        'OBS_VALUE':'CPI',
        'REF_AREA':'fic',
        'TIME_PERIOD':'fyear'
        })
    cpi = cpi[['CPI','fic','fyear']].copy()
    print(cpi.head())
    cpi.info()
    assert cpi['fic'].notna().all(),'cpi fic contains missing values'
    assert cpi['fyear'].notna().all(), 'cpi fyear contains missing values'
    assert cpi['CPI'].notna().all(),'cpi column contains missing values'
    assert cpi.duplicated(subset=['fic','fyear']).sum() == 0, 'cpi has duplicated fic-fyear keys'
    cpi.to_csv(processed_dir / 'CPI.csv',index=False)
    return cpi

# clean cab_pct_gdp
def clean_cab(raw_dir,processed_dir):
    cab = pd.read_csv(raw_dir / 'current_account_balance_pct_gdp.csv.csv')
    print(cab.shape)
    print(cab.columns)
    print(cab.head())
    required_cols = ['OBS_VALUE','REF_AREA','TIME_PERIOD']
    assert all(col in cab.columns for col in required_cols), 'cab contains missing columns'
    cab = cab.rename(columns={
        'OBS_VALUE':'cab',
        'REF_AREA':'fic',
        'TIME_PERIOD':'fyear'
        })
    cab = cab[['cab','fic','fyear']].copy()
    print(cab.head())
    cab.info()
    assert cab['fic'].notna().all(),'cab fic contains missing values'
    assert cab['fyear'].notna().all(), 'cab fyear contains missing values'
    assert cab['cab'].notna().all(),'cab column contains missing values'
    assert cab.duplicated(subset = ['fic','fyear']).sum() == 0, 'cab has duplicated fic-fyear keys'
    cab.to_csv(processed_dir / 'cab_pct_gdp.csv',index=False)
    return cab

# clean corruption
def clean_corruption(processed_dir):
    corr_raw = pd.read_excel(processed_dir / 'Corruption.xlsx')
    print(corr_raw.shape)
    print(corr_raw.columns)
    print(corr_raw.head())
    corr_raw = corr_raw.rename(columns={'Unnamed: 0':'fic'})
    corr_raw = corr_raw.set_index('fic')
    corr = corr_raw.T
    corr.index = corr.index.astype(str).str.replace('v','',regex=False).astype(int)
    print(corr.head())
    corr = corr.reset_index()
    corr = corr.rename(columns={'index':'fyear'})
    corr = corr.melt(id_vars='fyear',var_name='fic',value_name='corr')
    print(corr.head())
    assert corr['fic'].notna().all(),'corr fic contains missing values'
    assert corr['fyear'].notna().all(),'corr fyear contains missing values'
    assert corr.duplicated(subset=['fic','fyear']).sum() == 0, 'corr has duplicated fic-fyear keys'
    return corr

# build country panel
def build_country_panel(cpi,corr,cab,processed_dir):
    country = pd.merge(corr,cpi,on=['fic','fyear'],how='left')
    country = pd.merge(country,cab,on=['fic','fyear'],how='left')
    print(country.head())
    print(country.shape)
    print(country.columns.tolist())
    print(country.dtypes)
    print(country.isna().sum())

    #--TO DO--
    # v1: Keep the 4 missing values in the raw data.
    # v2: Replace incorrect variables, impute missing data, and revert assert to == 0.
    assert country.isna().sum().sum() == 4 , 'missing values still exist in the dataframe'
    assert country.duplicated(subset=['fic','fyear']).sum() == 0, 'country has duplicated fic-fyear keys'
    print(country.groupby('fic')['fyear'].nunique().sort_values())
    print(country[['corr', 'CPI', 'cab']].describe())
    country.to_csv(processed_dir / 'country.csv',index=False)
    return country

# build firm panel
def winsorize(s,lower_q=0.05,upper_q=0.95):
    lower = s.quantile(lower_q)
    upper = s.quantile(upper_q)
    return s.clip(lower = lower, upper = upper )

def build_firm_panel(raw_dir,processed_dir):
    USA = pd.read_stata(raw_dir / 'America.dta')
    global_df = pd.read_stata(raw_dir / 'G.dta')
    print(USA.head())
    print(global_df.head())
    print('USA duplicates:', USA.duplicated(subset=['gvkey', 'fyear']).sum())
    print('Global duplicates:', global_df.duplicated(subset=['gvkey','fyear']).sum())
    print("Global exact duplicates:", global_df.duplicated().sum())
    print("USA exact duplicates:", USA.duplicated().sum())
    dup_global = global_df[global_df.duplicated(subset=['gvkey', 'fyear'], keep=False)].sort_values(['gvkey', 'fyear'])
    dup_usa = USA[USA.duplicated(subset=['gvkey','fyear'],keep=False)].sort_values(['gvkey','fyear'])
    print(dup_global.head(10))
    print(dup_usa.head(10))
    USA['n_missing'] = USA.isna().sum(axis=1)
    global_df['n_missing'] = global_df.isna().sum(axis=1)
    USA = USA.sort_values(['gvkey','fyear','n_missing'])
    global_df = global_df.sort_values(['gvkey','fyear','n_missing'])
    USA = USA.drop_duplicates(subset=['gvkey','fyear'],keep='first').copy()
    global_df = global_df.drop_duplicates(subset=['gvkey','fyear'],keep='first').copy()
    assert USA.duplicated(subset=['gvkey', 'fyear']).sum() == 0 , 'USA still has duplicated gvkey-fyear key'
    assert global_df.duplicated(subset=['gvkey','fyear']).sum() == 0 , 'Global still has duplicated gvkey-fyear key'
    FIRM = pd.concat([USA,global_df],ignore_index=True)
    print(FIRM.head())
    assert FIRM.duplicated(subset=['gvkey','fyear']).sum() == 0, 'FIRM has duplicated gvkey-fyear key'
    FIRM['num_year'] = FIRM.groupby('gvkey')['fyear'].transform('nunique')
    FIRM = FIRM[FIRM['num_year']==15].copy()
    FIRM['ipoyear'] = FIRM['ipodate'].dt.year
    print((FIRM['at']<=0).sum())
    print((FIRM['lct']<=0).sum())
    print(((1+FIRM['fyear']-FIRM['ipoyear'])<=0).sum())
    FIRM = FIRM.assign(
        Age = lambda df: np.where(((1+df['fyear']-df['ipoyear'])>0),np.log(1+df['fyear']-df['ipoyear']),np.nan),
        LEV = lambda df: np.where(df['at']>0,(df['dlc']+df['dltt'])/df['at'],np.nan),
        SIZE = lambda df: np.where(df['at']>0 ,np.log(1+df['at']),np.nan),
        PROF = lambda df: np.where(df['at']>0,df['ni']/df['at'],np.nan),
        NDTS = lambda df: np.where(df['at']>0,df['dp']/df['at'],np.nan),
        LIQ = lambda df: np.where(df['lct']>0,df['act']/df['lct'],np.nan),
        TANG = lambda df: np.where(df['at']>0,df['ppent']/df['at'],np.nan)
    )
    print(FIRM[['Age', 'LEV', 'SIZE', 'PROF', 'NDTS', 'LIQ', 'TANG']].head())
    print(FIRM[['Age', 'LEV', 'SIZE', 'PROF', 'NDTS', 'LIQ', 'TANG']].isna().sum())

    FIRM = FIRM.assign(
        Missing = lambda df: df[['Age', 'LEV', 'SIZE', 'PROF', 'NDTS', 'LIQ', 'TANG']].isna().any(axis=1),
        company_has_missing = lambda df: df.groupby('gvkey')['Missing'].transform('max')
    )
    print(FIRM['company_has_missing'])
    FIRM = FIRM[FIRM['company_has_missing']==False].copy()
    winsor_vars = ['Age', 'LEV', 'SIZE', 'PROF', 'NDTS', 'LIQ', 'TANG']
    for col in winsor_vars:
        FIRM.loc[:,col] = FIRM.groupby('fic')[col].transform(
        lambda s: winsorize(s,0.05,0.95)
    )
    print(FIRM.columns)
    FIRM = FIRM[['Age', 'LEV', 'SIZE', 'PROF', 'NDTS', 'LIQ', 'TANG','fyear','fic','gvkey']].copy()
    print(FIRM)
    FIRM.to_csv(processed_dir / 'FIRM.csv',index=False)

def main():
    cpi = clean_cpi(RAW_DIR,PROCESSED_DIR)
    cab = clean_cab(RAW_DIR,PROCESSED_DIR)
    corruption = clean_corruption(PROCESSED_DIR)
    country = build_country_panel(cpi,corruption,cab,PROCESSED_DIR)
    firm = build_firm_panel(RAW_DIR,PROCESSED_DIR)

if __name__ == '__main__':
    main()
