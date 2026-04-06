import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# PATH 

BASE_DIR = Path('/Users/wulingzhou/VS_code_for_Python/capital-structure-dml-replication')
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
OUTPUT_DIR = BASE_DIR / 'output'
OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

# Figure 1-3: 

i = 1
for col in ['corr','cab','CPI']:
    sns.lineplot(x='fyear',y=col,data=baseline_data,hue='fic',marker='o',linestyle = '-')
    plt.title(f'{col} across countries')
    plt.legend(fontsize=10,loc='upper center',bbox_to_anchor=(0.5, -0.18),ncol=4)
    plt.xlabel('Year')
    plt.ylabel(str(col))
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(OUTPUT_DIR / f'Figure{i}-{col}-across-countries',dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    i += 1
