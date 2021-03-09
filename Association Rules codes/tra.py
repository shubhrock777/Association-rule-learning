import pandas as pd

tran= pd.read_csv("D:/BLR10AM/Assi/09.Association rules/Datasets_Association Rules/transactions_retail1.csv")


#######feature of the dataset to create a data dictionary

data_details =pd.DataFrame({"column name":tran.columns,
                "data types ":tran.dtypes})

# creating new df which has only tran_1s name columns

tran_1 = tran.iloc[:, 5:15]
tran_1.info()



###########Data Pre-processing 

#unique value for each columns 
col_uni =tran_1.nunique()
col_uni


#details of dataframe
tran_1.describe()

#checking for null or na vales 
tran_1.isna().sum()
tran_1.isnull().sum()


########etran_1ploratory data analysis

EDA = {"columns_name ":tran_1.columns,
                  "mode":tran_1.mode(),
                  "standard_deviation":tran_1.std(),
                  "variance":tran_1.var(),
                  "skewness":tran_1.skew(),
                  "kurtosis":tran_1.kurt()}

EDA


# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(tran_1.iloc[:, :])


#tran_1plot for every column
import matplotlib.pyplot as plt

for column in tran_1:
    plt.figure()
    tran_1.tran_1plot([column])


from mlxtend.frequent_patterns import apriori, association_rules

####Model Building
#Application of Apriori Algorithm.
        


frequent_itemsets = apriori(tran_1, min_support = 0.02, max_len = 3, use_colnames = True)

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True)


# graph for Most Frequent item sets based on support
plt.bar(x = list(range(0,5 )), height = frequent_itemsets.support[0:5], color ='rgmyk')
plt.xticks(list(range(0, 5)), frequent_itemsets.itemsets[0:5], rotation=30)
plt.xlabel('tran_1')
plt.ylabel('support')
plt.show()

#finding the rules 
rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head()

# shorting rules by lift ratio 
rules.sort_values('lift', ascending = False).head(10)

#custom define function
def to_list(i):
    return (sorted(list(i)))

ma_tran_1 = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

ma_tran_1 = ma_tran_1.apply(sorted)

rules_sets = list(ma_tran_1)

#unique rules only 
unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]



indetran_1_rules = []
for i in unique_rules_sets:
    indetran_1_rules.append(rules_sets.index(i))

# getting rules without any redudancy 
rules_no_redudancy = rules.iloc[indetran_1_rules, :]

# Sorting them with respect to list and getting top 5 rules 
# Build most frequent item sets and plot the rules
#top  5 rules for bussines 
top_5_rules =rules_no_redudancy.sort_values('lift', ascending = False).head()
top_5_rules
