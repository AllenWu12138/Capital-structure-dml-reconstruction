import pandas as pd
import numpy as np
from pathlib import Path
from linearmodels import PanelOLS, PooledOLS, RandomEffects
import statsmodels.api as sm
from linearmodels.panel import compare
from scipy import stats
import numpy.linalg as la

def auto_panel_regression(data,y,exog,ind_dummies):
    y = data[y]
    exog = sm.add_constant(data[exog])
    exog_with_ind = pd.concat([exog,data[ind_dummies]],axis=1)

    if ind_dummies:
        valid_ind_dummies = []
        for col in ind_dummies:
            if data[col].nunique() > 1:
                valid_ind_dummies.append(col)
        if valid_ind_dummies and (data[valid_ind_dummies].sum(axis=1) == 1).all():
            valid_ind_dummies.pop(0) 
        exog_with_ind = pd.concat([exog, data[valid_ind_dummies]], axis=1)
    else:
        exog_with_ind = exog

    # fit baseline models (unadjusted stndard errors)

    fe_model = PanelOLS(y,exog,entity_effects=True)
    fe_res = fe_model.fit(cov_type='unadjusted')
    re_model = RandomEffects(y,exog_with_ind)
    re_res = re_model.fit()
    pooled_model = PooledOLS(y,exog_with_ind)
    pooled_res = pooled_model.fit()

    # extract f-test p-value

    f_pvalue =fe_res.f_pooled.pval

    # construct lm-test

    resids = pooled_res.resids
    T = 10
    N = len(resids.index.levels[0])
    sum_seq_res = (resids**2).sum()
    sum_entity_res = (resids.groupby(level=0).sum()**2).sum()
    LM_stat = ((N*T)/(2*(T-1)))*(((sum_entity_res/sum_seq_res)-1)**2)
    lm_pvalue = stats.chi2.sf(LM_stat,1)

    # construct hausman-test

    b_fe = fe_res.params
    b_re = re_res.params
    v_fe = fe_res.cov
    v_re = re_res.cov
    common_vars = set(b_fe.index).intersection(set(b_re.index))
    if 'const' in common_vars:
        common_vars.remove('const')
    common_vars = list(common_vars)
    b_fe_sub = b_fe[common_vars]
    b_re_sub = b_re[common_vars]
    v_fe_sub = v_fe.loc[common_vars,common_vars]
    v_re_sub = v_re.loc[common_vars,common_vars]
    diff = b_fe_sub - b_re_sub
    cov_diff = v_fe_sub - v_re_sub
    inv_cov_diff = la.inv(cov_diff)
    hausman_stat = diff.dot(inv_cov_diff).dot(diff)
    df_hausman = len(common_vars)
    hausman_pvalue = stats.chi2.sf(hausman_stat,df_hausman)

    # model selection

    if f_pvalue >= 0.05 and lm_pvalue >= 0.05:
        print('Individual effects are not significant. Pooled OLS is recommended')
        final_model_name = 'Pooled'
    else:
        if hausman_pvalue < 0.05:
            print('reject H0. Fixed effect model is recommended')
            final_model_name = 'FE'
        else:
            print('fial to reject H0. Random effect model is recommended')
            final_model_name = 'RE'
    if final_model_name == 'FE':
        final_res = fe_model.fit(cov_type='clustered', cluster_entity = True, cluster_time = True)
    elif final_model_name == 'RE':
        final_res = re_model.fit(cov_type='clustered', cluster_entity = True, cluster_time = True)
    else:
        final_res = pooled_model.fit(cov_type='clustered', cluster_entity = True, cluster_time = True)
    return final_res, final_model_name

BASE_DIR = Path('/Users/wulingzhou/VS_code_for_Python/capital-structure-dml-replication')
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
OUTPUT_DIR = BASE_DIR / 'output'
OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

data = pd.read_csv('/Users/wulingzhou/VS_code_for_Python/capital-structure-dml-replication/data/processed/baseline_data.csv')
data = data[data['fyear']<=2019].copy()
ind_dummies = ['IND_15','IND_20','IND_25','IND_30','IND_35','IND_45','IND_50']
data =data.set_index(['gvkey','fyear'])
country_order = ['GBR','USA','FRA','JPN','KOR','BRA','ZAF']
y = 'LEV'
exog = ['Age', 'SIZE', 'PROF', 'NDTS', 'LIQ', 'TANG','EIR','GDP','Corruption']
all_models = {}
full_res,full_model =auto_panel_regression(
    data=data,
    y = y,
    exog= exog,
    ind_dummies= ind_dummies
)
print(full_res)
all_models['Full_Sample'] = full_res
for country in country_order:
    data_c = data[data['fic']== country]
    country_res, country_model =auto_panel_regression(
        data=data_c,
        y = y,
        exog= exog,
        ind_dummies= ind_dummies
    )  
    print(country_res)
    all_models[country] = country_res
regression_table = compare(all_models,stars=True)
print(regression_table)
output_csv = OUTPUT_DIR / 'regression_table.csv'
with open(output_csv, 'w') as f:
    f.write(str(regression_table.summary.as_csv()))
