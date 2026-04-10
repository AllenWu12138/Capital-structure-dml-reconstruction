import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# PATH 
BASE_DIR = Path('/Users/wulingzhou/VS_code_for_Python/capital-structure-dml-replication')
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
OUTPUT_DIR = BASE_DIR / 'output'
OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

# filter data

baseline_data = pd.read_csv(PROCESSED_DIR / 'baseline_data.csv')
baseline_data = baseline_data[baseline_data['fyear']<=2019].copy()

# Figure1-3 

i = 1
for col in ['Corruption','EIR','GDP']:
    plt.figure()
    sns.lineplot(x='fyear',y=col,data=baseline_data,hue='fic',marker='o',linestyle = '-')
    plt.title(f'{col} across countries')
    plt.legend(fontsize=10,loc='upper center',bbox_to_anchor=(0.5, -0.18),ncol=4)
    plt.xlabel('Year')
    plt.ylabel(str(col))
    plt.subplots_adjust(bottom=0.25)
    # plt.savefig(OUTPUT_DIR / f'Figure{i}-{col}-across-countries.png',dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    i += 1

# Table1: descriptive statistics of firm-level variables

vars_for_table = ['LEV','SIZE','PROF','NDTS','LIQ','TANG','Age']
country_order = ['GBR','USA','FRA','JPN','KOR','BRA','ZAF']
rows = []
for var in vars_for_table:
    for i,country in enumerate(country_order):
        a = baseline_data.loc[baseline_data['fic'] == country,var]
        row = {
            'Variables' : var if i == 0 else '',
            'Countries' : country,
            'Mean' : round(a.mean(),5),
            'Median' : round(a.median(),5),
            'Max' : round(a.max(),5),
            'Min' : round(a.min(),5),
            'ST.DEV': round(a.std(),5),
            'N': a.count()
        }
        rows.append(row)
statistics = pd.DataFrame(rows)
statistics.to_excel(OUTPUT_DIR / 'descriptive_statistics_of_firm-level_variables.xlsx',index=False)

# correlation matrix for studied countries

correlatin_matrix = baseline_data[vars_for_table].corr()
plt.figure()
sns.heatmap(data=correlatin_matrix,annot=True,fmt='.2f')
plt.title('Correlation Matrix for Studied Countries')
plt.savefig(OUTPUT_DIR / 'Correlation_matrix_for_studied_countries.png',dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# correlation matrix by country

for country in country_order:
    country_data = baseline_data.loc[baseline_data['fic']==country,vars_for_table]
    correlation = country_data.corr()
    plt.figure()
    sns.heatmap(data=correlation,annot=True,fmt='.2f')
    plt.title(f'Correlation matrix for {country}')
    plt.savefig(OUTPUT_DIR / f'Correlation_Matrix_for_{country}.png',dpi=300,bbox_inches='tight')
    plt.show()
    plt.close()
