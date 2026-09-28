import pandas as pd
import numpy as np
from pathlib import Path
import doubleml as dml
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from doubleml import DoubleMLData, DoubleMLPLR

BASE_DIR = Path('/Users/wulingzhou/VS_code_for_Python/capital-structure-dml-replication')
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
OUTPUT_DIR = BASE_DIR / 'output'
OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

def run_dml(processed_dir, output_dir):
    data = pd.read_csv(processed_dir/'baseline_data.csv')
    data = data[data['fyear']<=2019].copy()
    year_dummies = pd.get_dummies(data=data['fyear'],prefix='year',drop_first=True,dtype=float)
    data = pd.concat([data,year_dummies],axis=1)
    year_cols = year_dummies.columns.tolist()
    countries = ['All','GBR','USA','JPN','KOR','FRA','BRA','ZAF']
    treatments = ['SIZE','PROF','NDTS','LIQ','TANG']
    ind_dummies = ['IND_15','IND_20','IND_25','IND_30','IND_35','IND_45','IND_50']
    base_controls = ['Age','EIR','GDP','Corruption']
    results = []
    for i in countries:
        if i == 'All':
            sub_data = data.copy()
        else:
            sub_data = data[data['fic'] == i].copy()
        for var in treatments:
            other_controls = [v for v in treatments if v != var]
            controls = (base_controls + other_controls + ind_dummies + year_cols)
            data_dml_base = dml.DoubleMLData(sub_data,y_col='LEV',d_cols=var,x_cols=controls)
            lasso = make_pipeline(StandardScaler(),LassoCV(cv=5,max_iter=10000))
            np.random.seed(123)
            dml_plr_lasso = dml.DoubleMLPLR(data_dml_base, ml_l=lasso,ml_m=lasso,n_folds=5)
            dml_plr_lasso.fit(store_models = True)
            summary = dml_plr_lasso.summary.loc[var]
            results.append({
                'period': '2010-2019',
                'country': i,
                'treatment': var,
                'coef': summary['coef'],
                'std_err': summary['std err'],
                't': summary['t'],
                'p_value': summary['P>|t|'],
                'ci_lower': summary['2.5 %'],
                'ci_upper': summary['97.5 %']
            })
    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv(output_dir/'DML_results.csv',index=False)
    return results_df

if __name__ == '__main__':
    results_df = run_dml(
        PROCESSED_DIR,
        OUTPUT_DIR
    )
