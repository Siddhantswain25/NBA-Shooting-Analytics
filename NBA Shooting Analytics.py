from sklearn import tree, neighbors
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import time
start_time = time.time()

teams = ['ATL','BOS','BRO','CHA','CHI','CLE','DAL','DEN','DET','GSW','HOU','IND','LAC','LAL','MEM','MIA','MIL','MIN',
         'NOP','NYK','OKL','ORL','PHI','PHX','POR','SAC','SAS','TOR','UTA','WAS']

found = False

while not found:
    
    print ("Player Name ")
    currentPlayer = input().title()

    x = [] # location x
    y = [] # location y
    player = []
    outcome = [] # shot outcome (0 or 1)
    change = []
 
    for m in teams:
        df = pd.read_csv('datasets/shot log ' + m + '.csv', parse_dates = True)
        

        for i in df[['location x']]:
            for j in df[i]:
                if math.isnan(j):
                    x_temp = 200
                else:
                    x_temp = j

                if x_temp < 470:
                    x.append(-1*x_temp + 470)
                    change.append(0)
                else:
                    x.append(x_temp - 470)
                    change.append(1)


        for i in df[['location y']]:
            count = 0
            for j in df[i]:
                if math.isnan(j):
                    y_temp = 250
                else:
                    y_temp = j

                if change[count] == 0:
                    y.append(500 - y_temp)
                else:
                    y.append(y_temp)
                    
                count += 1
                

        for i in df[['shoot player']]:
            for j in df[i]:
                if j.title() == currentPlayer:
                    found = True
                player.append(j)

        for i in df[['current shot outcome']]:
            for j in df[i]:
                outcome.append(j)

print("\nPlayer Shot Analytics for season 2016-2017")
print("********************************************")
print("1 - Raw Data")
print("2 - Summary")
print("3 - Shooting Hotspots")
mode = 0
while not (mode > 0) and (mode < 4):
    print("Select the type of analysis: ")
    mode = int(input())
acc = 20
total_spots = 25

features = []
labels = []

for i in range(len(player)):
    if player[i].title() == currentPlayer:
        features.append([x[i],y[i]])
        if outcome[i] == "SCORED":
            labels.append(1)  # 1 = right court 
        else:
            labels.append(0)  # 0 = left court

features_standard = []
labels_standard = []

for i in range(len(player)):  
    features_standard.append([x[i],y[i]])
    if outcome[i] == "SCORED":
        labels_standard.append(1)
    else:
        labels_standard.append(0)
            
clf = neighbors.KNeighborsClassifier()
clf_standard = neighbors.KNeighborsClassifier()
clf.fit(features, labels)
clf_standard.fit(features_standard, labels_standard)


test = []
predictions = []
predictions_standard = []


for i in range(0,951,acc):
    for j in range(0,501,acc):
        test.append([i,j])
        predictions.append(clf.predict([[i,j]]))
        predictions_standard.append(clf_standard.predict([[i,j]]))

xs_shotsMade =[]
ys_shotsMade = []
xs_shotsMissed =[]
ys_shotsMissed = []

for i in range(len(features)):
    if labels[i] == 1:
        xs_shotsMade.append(features[i][0])
        ys_shotsMade.append(features[i][1])
    else:
        xs_shotsMissed.append(features[i][0])
        ys_shotsMissed.append(features[i][1])


if mode == 3:
    xs_shotsMade =[]
    ys_shotsMade = []
    xs_shotsMissed = []
    ys_shotsMissed = []
    xs = []
    ys = []

    for i in range(len(test)):
        if predictions[i] > 0:
            xs_shotsMade.append(test[i][0])
            ys_shotsMade.append(test[i][1])
        else:
            xs_shotsMissed.append(test[i][0])
            ys_shotsMissed.append(test[i][1])


#split court into grid of squares of 2 feet side length
spots = [] # spot coordinates with number of shots in each spot
spot_shots = []
summ_spots = []
shotAccuracy = acc

for i in range(0,940,shotAccuracy):
    for j in range(0,500,shotAccuracy):
        num_shots = 0
        for k in range(0,len(features)):
            if features[k][0] >= i - shotAccuracy and features[k][0] < i + 2*shotAccuracy and features[k][1] >= j - shotAccuracy and features[k][1] < j + 2*shotAccuracy:
                num_shots += 1
                
        spots.append([i,j])
        spot_shots.append(num_shots)
                
for i in range(total_spots):
    curr = max(spot_shots)
    index = 0
    for j in range(0,len(spot_shots)):
        if spot_shots[j] == curr:
            index = j
            break
    another_temp = spots[index]    
    summ_spots.append(another_temp)
    summ_spots[-1].append(curr)
    
    #removes adjacent squares
    to_remove = []
    for j in range(-1*shotAccuracy,2*shotAccuracy,shotAccuracy):
        if [spots[index][0] + j,spots[index][1] - shotAccuracy] in spots:
            to_remove.append(spots.index([spots[index][0] + j,spots[index][1] - shotAccuracy]))
            
        if [spots[index][0] + j,spots[index][1]] in spots:
            to_remove.append(spots.index([spots[index][0] + j,spots[index][1]]))
            
        if [spots[index][0] + j,spots[index][1] + shotAccuracy] in spots:
            to_remove.append(spots.index([spots[index][0] + j,spots[index][1] + shotAccuracy]))

    spots.remove(spots[index])
    spot_shots.remove(spot_shots[index])
    for j in reversed(to_remove):    
        spot_shots.remove(spot_shots[j])
        spots.remove(spots[j])
    
summary_xs = []
summary_ys = []
summaryShotPercentage = []
summaryShotAccuracy = []
for i in summ_spots:
    
    num_shots_made = 0
    num_shots_missed = 0

    s_x = 0
    s_y = 0
    
    for k in range(0,len(features)):
        if features[k][0] >= i[0] - shotAccuracy and features[k][0] < i[0] + 2*shotAccuracy and features[k][1] >= i[1] - shotAccuracy and features[k][1] < i[1] + 2*shotAccuracy:
            if labels[k] == 1:
                num_shots_made += 1
            else:
                num_shots_missed += 1
            s_x += features[k][0]
            s_y += features[k][1]

    if num_shots_made + num_shots_missed == 0 or round((num_shots_made + num_shots_missed)/len(features)*100,2) < 0.5:
         summary_xs.append(s_x)
         summary_ys.append(s_y)
         summaryShotPercentage.append(0)      
         summaryShotAccuracy.append(0)
            
    else:
        summary_xs.append(int(s_x/(num_shots_made + num_shots_missed)))
        summary_ys.append(int(s_y/(num_shots_made + num_shots_missed)))

        summaryShotPercentage.append(round((num_shots_made + num_shots_missed)/len(features)*100,2))      
        summaryShotAccuracy.append(round(num_shots_made/(num_shots_made + num_shots_missed),2))

img = plt.imread('court.png')
fig = plt.figure(num = currentPlayer)
plt.title(currentPlayer +"'s" + " Shot Analysis for \n 2016-2017 Season")
plt.xlabel('')
plt.ylabel('')
if mode != 2:
    plt.scatter(xs_shotsMissed, ys_shotsMissed, color='#FF3333', alpha = 0.5,zorder=1,s=40)
    plt.scatter(xs_shotsMade, ys_shotsMade, color='#00994C', alpha = 0.5,zorder=3,s=40)

if mode < 3:
    for i in range(len(summary_xs)):
        plt.scatter(summary_xs[i], summary_ys[i], color='#99FF33', alpha = summaryShotAccuracy[i],zorder=4,s=((summaryShotPercentage[i]**0.5)*12)**2)

plt.imshow(img,zorder=0)
plt.axis('off')

def onclick(event):

    xbasket = 423
    ybasket = 248

    dist_basket = round((((event.xdata - xbasket)**2 + (event.ydata - ybasket)**2)**0.5)/10,2)
    print("\nDistance from the basket: " + str(dist_basket) + " feet")
        
    # Information of the player in different positions of the court 
    if mode < 3:
        
        for i in range(len(summary_xs)):
            
            if ((event.xdata - summary_xs[i])**2 + (event.ydata - summary_ys[i])**2)**0.5 < 10:
                print("Shot Accuracy: " + str(round(summaryShotAccuracy[i]*100,2)) +
                      "% \nShot percentage taken at this position: "
                      + str(round(summaryShotPercentage[i],2)) + "%")
                break
                    
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
