import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
#0 is yes for brick 1 is no
# 0 is north 1 is east 2 is south 3 is west
neighborhood_mapping = {
    'North': 0,
    'East': 1,
    'South': 2,
    'West': 3
}
df = pd.read_csv('house-prices.csv')
df['Neighborhood_numeric'] = df['Neighborhood'].map(neighborhood_mapping)
df.to_csv('house-prices.csv', index=False)
newfile = open('house-prices.csv', 'r+')
lines = newfile.readlines()
list2 = []
for x in lines:
    if x == 0:
        continue
    list2.append([item.replace('"', '') for item in x.split(',')])
factorslist = []
priceslist = []
idlist = []
#0 is ID 1 is price 2 is sqft 3 is bedrooms 4 is bathrooms 6 is brick 7 is neighborhood
#make factorslist like =[[1, 1, 1,], [1, 3, 2]]
#convert everything to int, including brick and neighborhood
for x in range(len(list2)):
    if x == 0:
        continue
    if list2[x][6] == 'Yes':
        factorslist.append([int(list2[x][2]), int(list2[x][3]), int(list2[x][4]), 0, int(list2[x][8])])
    else:
        factorslist.append([int(list2[x][2]), int(list2[x][3]), int(list2[x][4]), 1, int(list2[x][8])])
    priceslist.append(int(list2[x][1]))
    idlist.append(int(list2[x][0]))
priceslist = np.array(priceslist)
factorslist = np.array(factorslist)
model = LinearRegression()
model.fit(factorslist, priceslist)
newfile.close()
predictedprice = []
pricedifference = []
for factors, price in zip(factorslist, priceslist):
    predictedprice.append(int(model.predict([factors])[0]))
for x in range(len(priceslist)):
    pricedifference.append(abs(predictedprice[x] - priceslist[x]))
updatedhouseprice = 'updatedhouseprices.csv'
data = {'House ID': idlist,
        'Original Price ($)': priceslist,
        'Predicted Price ($)': predictedprice,
        'Price Difference ($)': pricedifference
}
df2 = pd.DataFrame(data)
df2.to_csv(updatedhouseprice, index=False)
customquestions = [
    ("What is the Square footage: ", int),
    ("How many Bedrooms: ", int),
    ("How many Bathrooms: ", int),
    ("Is the House made of Brick (Yes or No): ", ["yes", "no"]),
    ("What Neighborhood (North, East, South, or West): ", ['north', 'east', 'south', 'west'])
]
answers = []
print('The predicted prices along with other data has been written to updatedhouseprices.csv')
print('If you would like the predicted price with custom factors, answer the following questions')
for prompt, expected in customquestions:
    while True:
        answer = input(prompt).strip().lower()
        if expected == int:
            try:
                answer = int(answer)
                answers.append(answer)
                break
            except:
                print('Please enter a Integer: ')
        else:
            if answer in expected:
                answers.append(answer)
                break
            else:
                print(f"Incorrect input, please type one of the following: {', '.join(expected)}")      
answers[4] = neighborhood_mapping[(answers[4].capitalize())]
if answers[3] == 'yes':
    answers[3] = 0
else:
    answers[3] = 1
answers = np.array(answers)
answers = answers.reshape(1, -1)
customprice = model.predict(answers)[0]
print(f"The predicted price for your house is ${int(customprice)}")