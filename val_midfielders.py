from numpy import float64
import requests
from bs4 import BeautifulSoup as bs
import pandas as pd

headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}

data_page = "https://www.transfermarkt.com/spieler-statistik/wertvollstespieler/marktwertetop?land_id=0&ausrichtung=Mittelfeld&spielerposition_id=alle&altersklasse=alle&jahrgang=0&kontinent_id=0&plus=1&page=20"
# Most valuable midfielders - page 1

response = requests.get(data_page, headers=headers)

bs_response = bs(response.content,'html.parser')    # html of webpage

#print(bs_response)

player_names=[]

name_cells = bs_response.find_all("img", {"class":"bilderrahmen-fixed"})
#print(name_cells)

for name_cell in name_cells:
    value = list(name_cell.attrs.values())
    player_names.append(value[2])

#print(player_names)
#------------------------------------------------------------------------------------------
country_names = []

cells_no_class = bs_response.find_all("td",{"class":"zentriert"})

for cell_no_class in cells_no_class:
    country_name = cell_no_class.find("img",{"class":"flaggenrahmen"})
    if(country_name != None):
        value = list(country_name.attrs.values())
        country_names.append(value[1])
        
#print(country_names)
#------------------------------------------------------------------------------------------
player_values = []

value_cells = bs_response.find_all("td",{"class":"rechts hauptlink"})
#print(value_cells)

for value_cell in value_cells:
    #print(value_cell)
    value = value_cell.contents[0].next
    value = value.replace("m","")
    value = value.replace("€","")
    value = float(value)
    value = value * 1000000
    player_values.append(value)

#print(player_values)
#-------------------------------------------------------------------------------------------

# <td class="zentriert">25</td>
age_cells = bs_response.find_all("td",{"class":"zentriert"})
#print(age_cells)
i = 0
# row number
# age
player_ages = []

# nationality
# club
# matches played
matches_played=[]
# goals
goals = []
# own goals
own_goals = []
# assists
assists = []
# yellow cards
yellow_cards = []
# second yellow cards
second_yellows = []
# red cards
red_cards = []
# subbed on
subbed_on = []
# subbed off
subbed_off = []

for age_cell in age_cells:
    if(i==0):
        #print("Row number: ")
        #print(age_cell.next)
        i=i+1
        # do x
    elif(i==1):
        #print("Age: ")
        #print(age_cell.next)
        player_ages.append(age_cell.next)
        i=i+1
    elif(i==2):
        #print("Nationality: ")
        #print(age_cell.next)
        i=i+1
        # do x
    elif(i==3):
        #print("Club: ")
        #print(age_cell.next)
        i=i+1
    elif(i==4):
        #print("Matches: ")
        #print(age_cell.next)
        matches_played.append(age_cell.next)
        i=i+1
        # do x
    elif(i==5):
        #print("Goals: ")
        #print(age_cell.next)
        goals.append(age_cell.next)
        i=i+1
    elif(i==6):
        #print("Own Goals: ")
        #print(age_cell.next)
        own_goals.append(age_cell.next)
        i=i+1
        # do x
    elif(i==7):
        #print("Assists: ")
        #print(age_cell.next)
        assists.append(age_cell.next)
        i=i+1
    elif(i==8):
        #print("Yellow Cards: ")
        #print(age_cell.next)
        yellow_cards.append(age_cell.next)
        i=i+1
        # do x
    elif(i==9):
        #print("2nd Yellow: ")
        #print(age_cell.next)
        second_yellows.append(age_cell.next)
        i=i+1
    elif(i==10):
        #print("Red Cards: ")
        #print(age_cell.next)
        red_cards.append(age_cell.next)
        i=i+1
        # do x
    elif(i==11):
        #print("Subbed on: ")
        #print(age_cell.next)
        subbed_on.append(age_cell.next)
        i=i+1
    elif(i==12):
        #print("Subbed off: ")
        #print(age_cell.next)
        subbed_off.append(age_cell.next)
        i=0

#-------------------------------------------------------------------------------------------

df = pd.DataFrame({"Name":player_names,"Nationality":country_names,"Age":player_ages, "Games Played":matches_played,"Goals":goals
                    ,"Own Goals":own_goals,"Assists":assists,"Yellow Cards":yellow_cards,"Red Cards":red_cards, "Second Yellow":second_yellows,
                    "Subbed On":subbed_on,"Subbed Off":subbed_off,"Value":player_values})

print(df)

#df.to_csv('Midfielders.csv',index=False,header=True)
df.to_csv('Midfielders.csv',mode='a',index=False,header=False)