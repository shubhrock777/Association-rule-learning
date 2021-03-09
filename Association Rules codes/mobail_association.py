import pandas as pd

mobail= pd.read_csv("D:/BLR10AM/Assi/09.Association rules/Datasets_Association Rules/myphonedata.csv")


#######feature of the dataset to create a data dictionary

data_details =pd.DataFrame({"column name":mobail.columns,
                "data types ":mobail.dtypes})

# creating new df which has only mobail_1s name columns

mobail_1 = mobail.iloc[:, 3:10]
mobail_1.info()



###########Data Pre-processing 

#unique value for each columns 
col_uni =mobail_1.nunique()
col_uni


#details of dataframe
mobail_1.describe()

#checking for null or na vales 
mobail_1.isna().sum()
mobail_1.isnull().sum()


########emobail_1ploratory data analysis

EDA = {"columns_name ":mobail_1.columns,
                  "mode":mobail_1.mode(),
                  "standard_deviation":mobail_1.std(),
                  "variance":mobail_1.var(),
                  "skewness":mobail_1.skew(),
                  "kurtosis":mobail_1.kurt()}

EDA


# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(mobail_1.iloc[:, :])


#mobail_1plot for every column
import matplotlib.pyplot as plt

for column in mobail_1:
    plt.figure()
    mobail_1.mobail_1plot([column])


from mlxtend.frequent_patterns import apriori, association_rules

####Model Building
#Application of Apriori Algorithm.
        


frequent_itemsets = apriori(mobail_1, min_support = 0.009, max_len = 2, use_colnames = True)

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True)


# graph for Most Frequent item sets based on support
plt.bar(x = list(range(0,3 )), height = frequent_itemsets.support[0:3], color ='rgmyk')
plt.xticks(list(range(0, 3)), frequent_itemsets.itemsets[0:3], rotation=30)
plt.xlabel('mobail_1')
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

ma_mobail_1 = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

ma_mobail_1 = ma_mobail_1.apply(sorted)

rules_sets = list(ma_mobail_1)

#unique rules only 
unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]



indemobail_1_rules = []
for i in unique_rules_sets:
    indemobail_1_rules.append(rules_sets.index(i))

# getting rules without any redudancy 
rules_no_redudancy = rules.iloc[indemobail_1_rules, :]

# Sorting them with respect to list and getting top 3 rules 
# Build most frequent item sets and plot the rules
#top  3 rules for bussines 
top_3_rules =rules_no_redudancy.sort_values('lift', ascending = False).head(3)
top_3_rules
