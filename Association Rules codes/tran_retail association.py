
import pandas as pd

tran= pd.read_csv("D:/BLR10AM/Assi/09.Association rules/Datasets_Association Rules/transactions_retail1.csv",header = None)

tran.nunique()


#Getting the list of transactions from the dataset 
transactions = []
for i in range(0, 557041):
    transactions.append([str(tran.values[i,j]) for j in range(0, 6)])


#for each product

all_transaction_list = [i for item in transactions for i in item]



from collections import Counter # ,OrderedDict

#
item_frequencies = Counter(all_transaction_list)



# after sorting
item_frequencies = sorted(item_frequencies.items(), key = lambda x:x[1])

# Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))

# barplot of top 10 
import matplotlib.pyplot as plt

plt.bar(height = frequencies[1:10], x = list(range(1, 10)), color = 'rgbkymc')
plt.xticks(list(range(1, 10), ), items[1:10])
plt.xlabel("items")
plt.ylabel("Count")
plt.show()


# Creating Data Frame for the transactions data
tran_series = pd.DataFrame(pd.Series(all_transaction_list))
tran_series.tail()


tran_series.columns = ["transactions"]

# creating a dummy columns for the each item in each transactions ... Using column names as item name
X = tran_series['transactions'].str.join(sep = '*').str.get_dummies(sep = '*')


from mlxtend.frequent_patterns import apriori, association_rules
frequent_itemsets = apriori(X, min_support = 0.0075, max_len = 4, use_colnames = True)

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True)

plt.bar(x = list(range(0, 11)), height = frequent_itemsets.support[0:11], color ='rgmyk')
plt.xticks(list(range(0, 11)), frequent_itemsets.itemsets[0:11], rotation=20)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head(20)
rules.sort_values('lift', ascending = False).head(10)


def to_list(i):
    return (sorted(list(i)))

ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]

index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

# getting rules without any redudancy 
rules_no_redudancy = rules.iloc[index_rules, :]

# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift', ascending = False).head(10)
