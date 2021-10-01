import numpy as np
import pandas as pd
import folium
import openrouteservice as ors

# read data and shift indexes so they are the same as numbering in map
locations = pd.read_csv("WoolworthsLocations.csv")
locations.index = np.arange(1, len(locations) + 1)

# demand data from averages. Distribution Centre was given demand of arbitrary high number 1000
weekday_demands = pd.read_csv("Demands as CSV.csv")
line = pd.DataFrame({"Store": "Distribution Centre", "Demand": 1000}, index=[56])
weekday_demands = pd.concat([weekday_demands.iloc[:55], line, weekday_demands.iloc[55:]]).reset_index(drop=True)
weekday_demands.index = np.arange(1, len(weekday_demands) + 1)
weekend_demands = pd.read_csv("Weekend Demands as CSV.csv")
line = pd.DataFrame({"Store": "Distribution Centre", "Demand": 1000}, index=[56])
weekend_demands = pd.concat([weekend_demands.iloc[:55], line, weekend_demands.iloc[55:]]).reset_index(drop=True)
weekend_demands.index = np.arange(1, len(weekend_demands) + 1)

distances = pd.read_csv("WoolworthsDistances.csv")
distances.index = np.arange(1, len(distances) + 1)
durations = pd.read_csv("WoolworthsTravelDurations.csv")
durations.index = np.arange(1, len(durations) + 1)

"""
HOW TO INDEX DATA IN DF

locations: 
    locations has 5 index types being: Type, Location, Store, Lat and Long
    E.g: to get the name of store 1: "locations.Store[1]" which returns "Countdown Airport"
    Works similarly for the other index types
demands:
    need averages data Sebastian created into a csv
distances:
    works differently to locations data. easy to access this one using the ".loc" method
    E.g: "distances.loc[1]" grabs the entire column for Countdown Airport and all the rows of corresponding
         distances.
    Then "distances.loc[1][2]" gives us the distance from Countdown Auckland City (store 2) to Countdown Airport
    (store 1). Piazza post Kevin says that we should go rows to columns, so this means to get the distance from store 32
    to store 33 we use "distances.loc[33][32]" ie "distances.loc[destination][origin]"
durations:
    works identically to distances data.
    E.g: "durations.loc[2][1]" returns the time taken to travel from store 1 to store 2 (Countdown Airport to Countdown
    Auckland City)
demands:
    works like locations and has two different index types being: Store and Demand
    E.g: to get weekday demand for Countdown Airport: "weekday_demands.Demand[1]" returns 8
"""


def colour_map():
    """
    Function taken from Week 9 Pre-Lab just colours and plots all the data points in an intractable map. Then saves
    the map to root directory as an html file viewable in browsers.
    """
    # make coordinates into list of lists
    coords = locations[['Long', 'Lat']]
    coords = coords.to_numpy().tolist()

    m = folium.Map(location=list(reversed(coords[2])), zoom_start=10)
    folium.Marker(list(reversed(coords[0])), popup=locations.Store[0], icon=folium.Icon(color='black')).add_to(m)

    for i in range(0, len(coords)):
        if locations.Type[i] == "Countdown":
            iconCol = "green"
        elif locations.Type[i] == "FreshChoice":
            iconCol = "blue"
        elif locations.Type[i] == "SuperValue":
            iconCol = "red"
        elif locations.Type[i] == "Countdown Metro":
            iconCol = "orange"
        elif locations.Type[i] == "Distribution Centre":
            iconCol = "black"
        folium.Marker(list(reversed(coords[i])), popup=locations.Store[i], icon=folium.Icon(color=iconCol)).add_to(m)

    # save the map to an html file (viewable in browser)
    # m.save('map.html')
