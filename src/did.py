import pandas as pd
from pathlib import Path
from linearmodels import PanelOLS
import statsmodels.api as sm

BASE_DIR = Path('/Users/wulingzhou/VS_code_for_Python/capital-structure-dml-replication')
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
OUTPUT_DIR = BASE_DIR / 'output'
OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

data = pd.read_csv(PROCESSED_DIR/'baseline_data.csv')
treatment_countries = (data.loc[(data['fyear']==2020)&(data['GDP']<-3),'fic'].unique())
data['treatment'] = data['fic'].isin(treatment_countries).astype(int)
data['post'] = (data['fyear']>=2020).astype(int)
data['did'] = data['treatment']*data['post']
data = data.set_index(['gvkey','fyear'])
print(treatment_countries)
print(data.reset_index().groupby("fic")["treatment"].first())
print(pd.crosstab(data.reset_index()["treatment"],data.reset_index()["post"]))

y = data['LEV']
X = sm.add_constant(data[['did', 'SIZE', 'PROF', 'NDTS', 'LIQ', 'TANG','Age', 'EIR', 'Corruption']])
model = PanelOLS(y,X,entity_effects=True,time_effects=True)
model_res = model.fit(cov_type='clustered',cluster_entity=True)
print(model_res)

result_table = pd.DataFrame({
    'coef':model_res.params,
    'std_err': model_res.std_errors,
    't_stat':model_res.tstats,
    'p_value':model_res.pvalues
})
result_table.to_csv(OUTPUT_DIR/'did_results.csv')
