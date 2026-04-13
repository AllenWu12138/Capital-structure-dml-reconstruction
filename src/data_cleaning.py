import pandas as pd
import numpy as np
from pathlib import Path

# PATH
BASE_DIR = Path('/Users/wulingzhou/VS_code_for_Python/capital-structure-dml-replication')
RAW_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
PROCESSED_DIR.mkdir(parents=True,exist_ok=True)

# clean EIR
def clean_EIR(raw_dir,processed_dir):
    EIR = pd.read_csv(raw_dir/ 'EIR.csv')
    print(EIR.head(7))
    EIR['COUNTRY'] = EIR['COUNTRY'].replace({
        'United States':'USA',
        'United Kingdom':'GBR',
        'France':'FRA',
        'Japan':'JPN',
        'South Africa':'ZAF',
        'Brazil':'BRA',
        'Korea, Republic of':'KOR'})
    year = {str(y):str(y-1) for y in range(2010,2026)}
    EIR = EIR.rename(columns=year)
    EIR = EIR.drop(columns='2009')
    print(EIR.columns)
    cols = ['COUNTRY','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021','2022','2023','2024']
    EIR = EIR[cols].copy()
    EIR = EIR.rename(columns={'COUNTRY':'fic'}).copy()
    EIR = EIR.set_index('fic')
    EIR = EIR.T
    EIR.index = EIR.index.astype(int)
    print(EIR.head())
    EIR=EIR.reset_index()
    EIR = EIR.rename(columns={'index':'fyear'})
    EIR = EIR.melt(id_vars='fyear',var_name='fic',value_name='EIR')
    EIR['fyear']=pd.to_numeric(EIR['fyear'],errors='coerce').astype(int)
    EIR['EIR']=pd.to_numeric(EIR['EIR'],errors='coerce')
    assert EIR['EIR'].notna().all(),'EIR contains missing values'
    assert EIR['fic'].notna().all(),'EIR fic contains missing values'
    assert EIR['fyear'].notna().all(),'EIR fyear contains missing values'
    assert EIR.duplicated(subset=['fic','fyear']).sum() == 0, 'EIR has duplicated fic-fyear keys'
    EIR.to_csv(processed_dir / 'EIR.csv',index=False)
    return EIR

# clean GDP_growth_rate
def clean_GDP(raw_dir,processed_dir):
    GDP = pd.read_excel(raw_dir / 'GDP_growth_rate.xlsx')
    print(GDP.head())
    print(GDP.columns)
    GDP = GDP.rename(columns={' ':'fic'}).copy()
    GDP['fic'] = GDP['fic'].replace({
        'France':'FRA',
        'United States':'USA',
        'United Kingdom':'GBR',
        'South Africa':'ZAF',
        'Japan':'JPN',
        'Korea, Rep.':'KOR',
        'Brazil':'BRA'
    })
    GDP = GDP.drop(columns='Unnamed: 11',index=7)
    GDP = GDP.melt(id_vars='fic',var_name='fyear',value_name='GDP')
    GDP['fyear']=pd.to_numeric(GDP['fyear'],errors='coerce').astype(int)
    GDP['GDP']=pd.to_numeric(GDP['GDP'],errors='coerce')
    assert GDP['GDP'].notna().all(),'GDP contains missing values'
    assert GDP['fic'].isna().sum()==0,'GDP fic contians missing value'
    assert GDP['fyear'].isna().sum()==0,'GDP fyear contains missing value'
    assert GDP['GDP'].notna().all(), 'GDP contains missing values'
    assert GDP.duplicated(subset=['fic','fyear']).sum() == 0, 'GDP has duplicated fic-fyear keys'
    GDP.to_csv(processed_dir / 'GDP.csv',index=False)
    return GDP

# clean corruption
def clean_corruption(processed_dir):
    Corruption_raw = pd.read_excel(processed_dir / 'Corruption.xlsx')
    print(Corruption_raw.shape)
    print(Corruption_raw.columns)
    print(Corruption_raw.head())
    Corruption_raw = Corruption_raw.rename(columns={'Unnamed: 0':'fic'})
    Corruption_raw = Corruption_raw.set_index('fic')
    Corruption = Corruption_raw.T
    Corruption.index = Corruption.index.astype(str).str.replace('v','',regex=False).astype(int)
    print(Corruption.head())
    Corruption = Corruption.reset_index()
    Corruption = Corruption.rename(columns={'index':'fyear'})
    Corruption = Corruption.melt(id_vars='fyear',var_name='fic',value_name='Corruption')
    print(Corruption.head())
    assert Corruption['fic'].notna().all(),'Corruption fic contains missing values'
    assert Corruption['fyear'].notna().all(),'Corruption fyear contains missing values'
    assert Corruption.duplicated(subset=['fic','fyear']).sum() == 0, 'Corruption has duplicated fic-fyear keys'
    return Corruption

# build country panel
def build_country_panel(EIR,Corruption,GDP,processed_dir):
    country = pd.merge(Corruption,EIR,on=['fic','fyear'],how='left')
    country = pd.merge(country,GDP,on=['fic','fyear'],how='left')
    print(country.head())
    print(country.shape)
    print(country.columns.tolist())
    print(country.dtypes)
    print(country.isna().sum())
    assert country.isna().sum().sum() == 0 , 'missing values still exist in the dataframe'
    assert country.duplicated(subset=['fic','fyear']).sum() == 0, 'country has duplicated fic-fyear keys'
    print(country.groupby('fic')['fyear'].nunique().sort_values())
    print(country[['Corruption', 'EIR', 'GDP']].describe())
    country.to_csv(processed_dir / 'country.csv',index=False)
    return country

# build firm panel
def winsorize(s,lower_q=0.05,upper_q=0.95):
    lower = s.quantile(lower_q)
    upper = s.quantile(upper_q)
    return s.clip(lower = lower, upper = upper )

def build_firm_panel(raw_dir, processed_dir):
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
    FIRM = FIRM.drop_duplicates(subset=['gvkey','fyear'],keep='first').copy()
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
    FIRM = FIRM[['Age', 'LEV', 'SIZE', 'PROF', 'NDTS', 'LIQ', 'TANG','fyear','fic','gvkey','gsector']].copy()
    print(FIRM)
    FIRM.to_csv(processed_dir / 'FIRM.csv',index=False)
    return FIRM

def final_data(firm_df,country_df,processed_dir):
    baseline_data = pd.merge(
        firm_df,
        country_df,
        on=['fic','fyear'],
        how='left'
    )
    baseline_data.loc[baseline_data['gsector']==15,'IND2']=1
    baseline_data.loc[baseline_data['gsector']==20,'IND3']=1
    baseline_data.loc[baseline_data['gsector']==25,'IND4']=1
    baseline_data.loc[baseline_data['gsector']==30,'IND5']=1
    baseline_data.loc[baseline_data['gsector']==35,'IND6']=1
    baseline_data.loc[baseline_data['gsector']==45,'IND7']=1
    baseline_data.loc[baseline_data['gsector']==50,'IND8']=1
    baseline_data[['IND2','IND3','IND4','IND5','IND6','IND7','IND8']]= baseline_data[['IND2','IND3','IND4','IND5','IND6','IND7','IND8']].fillna(0)
    assert baseline_data['EIR'].isna().sum() == 0, 'Unexpected NaNs found in EIR data'
    assert baseline_data['Corruption'].isna().sum() == 0 , 'Unexpected NaNs found in corruption data'
    assert baseline_data['GDP'].isna().sum() == 0 , 'Unexpected NaNs found in GDP data'
    baseline_data.to_csv(processed_dir / 'baseline_data.csv',index=False)
    return baseline_data

def main():
    EIR = clean_EIR(RAW_DIR,PROCESSED_DIR)
    GDP = clean_GDP(RAW_DIR,PROCESSED_DIR)
    corruption = clean_corruption(PROCESSED_DIR)
    country = build_country_panel(EIR,corruption,GDP,PROCESSED_DIR)
    firm = build_firm_panel(RAW_DIR,PROCESSED_DIR)
    baseline_data = final_data(firm,country,PROCESSED_DIR)

if __name__ == '__main__':
    main()
