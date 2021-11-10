import requests
from bs4 import BeautifulSoup

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris




headers = {'User-Agent':
           'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}

page = "https://www.transfermarkt.co.uk/transfers/transferrekorde/statistik?saison_id=alle&land_id=0&ausrichtung=&spielerposition_id=&altersklasse=&leihe=&w_s=&plus=1"
pageTree = requests.get(page, headers=headers)
pageSoup = BeautifulSoup(pageTree.content, 'html.parser')

Playersdata = pageSoup.find_all("td", {"class": "hauptlink"})
Values = pageSoup.find_all("td", {"class": "rechts hauptlink"})
Age = pageSoup.find_all("td", {"class": "zentriert"})



PlayersList = []
ValuesList = []
rank = []
age = []
season = []
team = []
missing = []

for i in range(0, 25):
    PlayersList.append(Playersdata[i*4].text)
    ValuesList.append(Values[i].text)
    rank.append(Age[(5 * i)].text)
    age.append(Age[1 + (5 * i)].text)
    season.append(Age[2 + (5 * i)].text)
   # missing.append(Age[3 + (5 * i)].text)
    team.append(Age[4 + (5 * i)].text)

df = pd.DataFrame({"Players": PlayersList,
                   "Value_post": ValuesList,
                   "rank": rank,
                   "age": age,
                  "season": season,
                   #"missing": missing,
                   "team": team})

df.head()

#print(Players[1].text)
print(df.to_string())