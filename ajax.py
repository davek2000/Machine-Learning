# https://dev.to/lisandramelo/extracting-data-from-transfermarkt-an-introduction-to-webscraping-2i1c
import requests
from bs4 import BeautifulSoup as bs
import pandas as pd

headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.106 Safari/537.36'}

# endereco_da_pagina stands for the data page address
data_page = "https://www.transfermarkt.co.uk/ajax-amsterdam/transferrekorde/verein/610/saison_id//pos//detailpos/0/w_s//altersklasse//plus/1"

response = requests.get(data_page, headers=headers)

bs_response = bs(response.content,'html.parser')    # html of webpage

print(bs_response)

# info we need is in a table

# each row represents a player

# name represented by an anchor (<a>) with the class "spielprofil_tooltip"
# country of origin represented as flag image with class "flaggenrahmen" in the 7th column of each row
# cost represented by a table cell (<td>) with class "rechts hauptlink"
# age represented by a table cell (<td>) with class "zentriert" ??

# Name 
player_names = []

#print(bs_response)

#name_cells = bs_response.find_all("a", {"class": "spielprofil_tooltip"})
name_cells = bs_response.find_all("img", {"class": "bilderrahmen-fixed lazy lazy"})

#print(name_cells)

for name_cell in name_cells:
    #print(name_cell.attrs)
    value = list(name_cell.attrs.values())
    player_names.append(value[2])

#print(player_names)

# Country
country_names = []

cells_no_class = bs_response.find_all("td",{"class":"zentriert"})

for cell_no_class in cells_no_class:
    country_name = cell_no_class.find("img",{"class":"flaggenrahmen"})
    if(country_name != None):
        value = list(country_name.attrs.values())
        country_names.append(value[1])
        
#print(country_names)

# Value

player_values = []

value_cells = bs_response.find_all("td",{"class":"rechts hauptlink"})
#print(value_cells)

for value_cell in value_cells:
    #print(value_cell)
    player_values.append(value_cell.string)

print(player_values)

df = pd.DataFrame({"Name":player_names,"Nationality":country_names,"Value":player_values})

print(df)