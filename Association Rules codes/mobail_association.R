# Load the dataset
library(readr)
mobail <- read_csv(file.choose())

mobail_1 <- mobail[ , c(4:9)]


library("arules") # Used for building association rules i.e. apriori algorithm

# Groceries is in transactions format

summary(mobail_1)

# making rules using apriori algorithm 
# Keep changing support and confidence values to obtain different rules

# Building rules using apriori algorithm
arules <- apriori(as.matrix(mobail_1), parameter = list(support = 0.007, confidence = 0.60, minlen = 2))
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
plot(arules[1:4], method = "graph") # for good visualization try plotting only few rules

#saving the rules 
write(arules, file = "a_rules.csv", sep = ",")
