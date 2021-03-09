import pandas as pd

book= pd.read_csv("D:/BLR10AM/Assi/09.Association rules/Datasets_Association Rules/book.csv")


#######feature of the dataset to create a data dictionary

data_details =pd.DataFrame({"column name":book.columns,
                "data types ":book.dtypes})


###########Data Pre-processing 

#unique value for each columns 
col_uni = book.nunique()
col_uni


#details of dataframe
book.describe()
book.info()

#checking for null or na vales 
book.isna().sum()
book.isnull().sum()


########ebookploratory data analysis

EDA = {"columns_name ":book.columns,
                  "mode":book.mode(),
                  "standard_deviation":book.std(),
                  "variance":book.var(),
                  "skewness":book.skew(),
                  "kurtosis":book.kurt()}

EDA


# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(book.iloc[:, :])


#bookplot for every columns
book.nunique()

#bookplot for every column
import matplotlib.pyplot as plt

for column in book:
    plt.figure()
    book.bookplot([column])


from mlxtend.frequent_patterns import apriori, association_rules

####Model Building
#Application of Apriori Algorithm.
        


frequent_itemsets = apriori(book, min_support = 0.01, max_len = 4, use_colnames = True)

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True)


# graph for Most Frequent item sets based on support
plt.bar(x = list(range(0, 10)), height = frequent_itemsets.support[0:10], color ='rgmyk')
plt.xticks(list(range(0, 10)), frequent_itemsets.itemsets[0:10], rotation=90)
plt.xlabel('book')
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

ma_book = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

ma_book = ma_book.apply(sorted)

rules_sets = list(ma_book)

#unique rules only 
unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]



indebook_rules = []
for i in unique_rules_sets:
    indebook_rules.append(rules_sets.index(i))

# getting rules without any redudancy 
rules_no_redudancy = rules.iloc[indebook_rules, :]

# Sorting them with respect to list and getting top 10 rules 
# Build most frequent item sets and plot the rules
#top  10 rules for bussines 
top_10_rules =rules_no_redudancy.sort_values('lift', ascending = False).head(10)
top_10_rules
