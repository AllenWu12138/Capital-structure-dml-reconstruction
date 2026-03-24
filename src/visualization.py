import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# Figure 1: Corruption across countries

data = pd.read_excel('data/processed/Corruption.xlsx')
data = data.rename(columns={'Unnamed: 0':'Country'})
data.set_index('Country',inplace=True)
corruption = data.T
corruption.index = corruption.index.str.replace('v','').astype(int)
sns.set_theme(style='white')
for column in corruption.columns:
    sns.lineplot(
        x=corruption.index,
        y=corruption[column],
        data=corruption,
        marker='o',
        linestyle='-',
        label=column
        )
plt.title('Corruption across countries')
plt.legend(fontsize=10,loc='upper center',bbox_to_anchor=(0.5, -0.18),ncol=4)
plt.xlabel('Year')
plt.ylabel('Corruption')
plt.subplots_adjust(bottom=0.25)
plt.savefig('output/Figure1-Corruption-across-countries.png',dpi=300, bbox_inches='tight')
plt.show()
