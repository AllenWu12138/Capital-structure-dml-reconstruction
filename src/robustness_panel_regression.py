import pandas as pd
import numpy as np
from pathlib import Path
from linearmodels import PanelOLS, PooledOLS, RandomEffects
import statsmodels.api as sm
from linearmodels.panel import compare
from scipy import stats
import numpy.linalg as la

BASE_DIR = Path('/Users/wulingzhou/VS_code_for_Python/capital-structure-dml-replication')
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
OUTPUT_DIR = BASE_DIR / 'output'
OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

data=pd.read_csv(PROCESSED_DIR/'baseline_data.csv')
data = data[data['fyear']>2019].copy()
data =data.set_index(['gvkey','fyear'])

RE_Countries = ['FRA','BRA','ZAF']
Countries = ['All','GBR','USA','JPN','KOR','FRA','BRA','ZAF']
ind_dummies = ['IND_15','IND_20','IND_25','IND_30','IND_35','IND_45','IND_50']
exog = ['Age', 'SIZE', 'PROF', 'NDTS', 'LIQ', 'TANG','EIR','GDP','Corruption']

def robustness_panel_regression(data,RE_Countries,Countries,exog,ind_dummies):
    all_models = {}
    for i in Countries:
        if i =='All':
            sub_data = data.copy()
            X=sm.add_constant(sub_data[exog])
            fe_model = PanelOLS(sub_data['LEV'],X,entity_effects=True)
            fe_res = fe_model.fit(cov_type='clustered', cluster_entity = True, cluster_time = True)
            all_models['Full_samples']=fe_res
        elif i in RE_Countries:
            sub_data = data[data['fic']==i].copy()
            X=sm.add_constant(sub_data[exog])
            if ind_dummies:
                valid_ind_dummies=[]
                for col in ind_dummies:
                    if sub_data[col].nunique()>1:
                        valid_ind_dummies.append(col)
                if valid_ind_dummies and (sub_data[valid_ind_dummies].sum(axis=1)==1).all():
                    valid_ind_dummies.pop(0)
                X_with_ind = pd.concat([X,sub_data[valid_ind_dummies]],axis=1)
            else:
                X_with_ind = X
            re_model = RandomEffects(sub_data['LEV'],X_with_ind)
            re_res = re_model.fit(cov_type='clustered', cluster_entity = True, cluster_time = True)
            all_models[i]=re_res
        else:
            sub_data = data[data['fic']==i].copy()
            X=sm.add_constant(sub_data[exog])
            fe_model = PanelOLS(sub_data['LEV'],X,entity_effects=True)
            fe_res = fe_model.fit(cov_type='clustered', cluster_entity = True, cluster_time = True)
            all_models[i]=fe_res
    robustness_panel_regression = compare(all_models,stars=True)
    print(robustness_panel_regression)
    output_csv = OUTPUT_DIR / 'regression_robust.csv'
    with open(output_csv, 'w') as f:
        f.write(str(robustness_panel_regression.summary.as_csv()))
    return robustness_panel_regression

if __name__ == '__main__':
    results_df = robustness_panel_regression(data,RE_Countries,Countries,exog,ind_dummies)
