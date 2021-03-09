# Load the dataset
library(readr)
book <- read_csv(file.choose())

library("arules") # Used for building association rules i.e. apriori algorithm

 # Groceries is in transactions format

summary(book)

# making rules using apriori algorithm 
# Keep changing support and confidence values to obtain different rules

# Building rules using apriori algorithm
arules <- apriori(as.matrix(book), parameter = list(support = 0.01, confidence = 0.90, maxlen = 4))
arules

# Viewing rules based on lift value
inspect(head(sort(arules, by = "lift"))) # to view we use inspect 

# Overal quality 
head(quality(arules))

# install.packages("arueslViz")
library("arulesViz") # for visualizing rules

# Different Ways of Visualizing Rules
plot(arules,method = "scatterplot")

windows()
plot(arules, method = "grouped")
plot(arules[1:10], method = "graph") # for good visualization try plotting only few rules

#saving the rules 
write(arules, file = "a_rules.csv", sep = ",")
