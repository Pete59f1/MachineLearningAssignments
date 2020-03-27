import pandas as pd
from apyori import apriori

# Looking at the current dataset
dataset = pd.read_csv("heart.csv", header=None)
# print(dataset)


# Placing our data into an array
# 0 to 304 is the number of rows our dataset have
# 0 to 14 the number of columns our dataset have
records = []
for i in range(0, 303):
    records.append([str(dataset.values[i, j]) for j in range(0, 13)])

# Creating our association rules and making a list
associationRules = apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)
associationList = list(associationRules)

# Now we can print our rules, support, confidence and lift
for item in associationList:
    pair = item[0]
    items = [x for x in pair]
    print("Rule: " + items[0] + "-> " + items[1])
    print("Support: " + str(item[1]))
    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("===================================")

# https://stackabuse.com/association-rule-mining-via-apriori-algorithm-in-python/
