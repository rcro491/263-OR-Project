import numpy as np
import pandas as pd
import folium
import openrouteservice as ors
import itertools

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

def weekday_routes():
    #Groups of stores split into 5 Auckland regions - South, East, Central, West, North
    south = [1, 22, 23, 24, 25, 26, 40, 41, 44, 49, 59, 60, 63, 65]
    east = [3, 6, 14, 16, 27, 28, 34, 39, 45, 48, 58]
    central = [2, 9, 10, 11, 29, 30, 32, 33, 35, 38, 42, 43, 46, 53, 54]
    west = [5, 13, 15, 17, 18, 19, 20, 37, 51, 52, 55, 57, 61, 62, 64, 66]
    north = [4, 7, 8, 12, 21, 31, 36, 47, 50]
    #Distribution centre
    dist_centre = 56
    #Create a list of regions
    regions = [south, east, central, west, north]
    #Route matrix for region 
    routes = np.zeros((0, 4))
    

    #Loop through the regions
    for region in regions:
        #For each node in the region
        for route in itertools.combinations(region, 3):
            i1, i2, i3 = route
            #Record demand for each of the nodes
            demand1 = weekday_demands.loc[i1]['Demand']
            demand2 = weekday_demands.loc[i2]['Demand']
            demand3 = weekday_demands.loc[i3]['Demand']
            #Calculate total demand
            total_demand = (demand1 + demand2 + demand3)
            #If total demand is more than 26 the route is not feasible
            if total_demand > 26:
                continue
            #Record the total time for each of the nodes - travel from distribution centre + unloading
            #Distribution centre to node1, node2, node3
            time0_1 = (durations.loc[i1][dist_centre])/60 + 7.5*demand1
            #Node1 to node2
            time1_2 = (durations.loc[i2][i1])/60 + 7.5*demand2
            #Node2 to node3
            time2_3 = (durations.loc[i3][i2])/60 + 7.5*demand3
            #Node3 back to distribution centre
            time3 = (durations.loc[dist_centre][i3])/60
            #Calculate the total time for route 1: distribution centre - node1 - node2 - node3 - distribution centre
            total_time = time0_1 + time1_2 + time2_3 + time3
            #If the total time is less than or equal to 4h calculate cost using standard truck pricing
            if total_time <= 240:
                cost = total_time*3.75
            #Otherwise calculate cost with overtime pricing
            else:
                overtime = total_time - 240
                cost = 240*3.75 + overtime*4.583
            #Add route1 and total cost to route matrix
            routes = np.append(routes, np.array([[i1, i2, i3, cost]]), axis = 0)
    #Return route matrices
    return routes

def weekend_routes():
    #Groups of stores split into 5 Auckland regions - South, East, Central, West, North
    south = [1, 22, 23, 24, 25, 26, 40, 41, 44, 49]
    east = [3, 6, 14, 16, 27, 28, 34, 39, 45, 48]
    central = [2, 9, 10, 11, 32, 33, 35, 38, 42, 43, 46, 53, 54, 54]
    west = [5, 13, 15, 17, 18, 19, 20, 37, 51, 52, 55]
    north = [4, 7, 8, 12, 21, 31, 36, 47, 50]
    #Distribution centre
    dist_centre = 56
    #Create a list of regions
    regions = [south, east, central, west, north]
    #Route matrix for region 
    routes = np.zeros((0, 5))
    

    #Loop through the regions
    for region in regions:
        #For each node in the region
        for route in itertools.combinations(region, 4):
            i1, i2, i3, i4 = route
            #Record demand for each of the nodes
            demand1 = weekend_demands.loc[i1]['Demand']
            demand2 = weekend_demands.loc[i2]['Demand']
            demand3 = weekend_demands.loc[i3]['Demand']
            demand4 = weekend_demands.loc[i4]['Demand']
            #Calculate total demand
            total_demand = (demand1 + demand2 + demand3 + demand4)
            #If total demand is more than 26 the route is not feasible
            if total_demand > 26:
                continue
            #Record the total time for each of the nodes - travel from distribution centre + unloading
            #Distribution centre to node1, node2, node3
            time0_1 = (durations.loc[i1][dist_centre])/60 + 7.5*demand1
            #Node1 to node2
            time1_2 = (durations.loc[i2][i1])/60 + 7.5*demand2
            #Node2 to node3
            time2_3 = (durations.loc[i3][i2])/60 + 7.5*demand3
            #Node 3 to node4
            time3_4 = (durations.loc[i4][i3])/60 + 7.5*demand4
            #Node4 back to distribution centre
            time4 = (durations.loc[dist_centre][i4])/60
            #Calculate the total time for route 1: distribution centre - node1 - node2 - node3 - distribution centre
            total_time = time0_1 + time1_2 + time2_3 + time3_4 + time4
            #If the total time is less than or equal to 4h calculate cost using standard truck pricing
            if total_time <= 240:
                cost = total_time*3.75
            #Otherwise calculate cost with overtime pricing
            else:
                overtime = total_time - 240
                cost = 240*3.75 + overtime*4.583
            #Add route1 and total cost to route matrix
            routes = np.append(routes, np.array([[i1, i2, i3, i4, cost]]), axis = 0)
    #Return route matrices
    return routes

def store_name(node):
    return weekday_demands.loc[node]['Store']


def main():
    weekday_costs = weekday_routes()
    weekdays = pd.DataFrame(data = weekday_costs, columns = ['Store 1', 'Store 2', 'Store 3', 'Cost'])
    weekdays["Store 1"] = weekdays["Store 1"].apply(store_name)
    weekdays["Store 2"] = weekdays["Store 2"].apply(store_name)
    weekdays["Store 3"] = weekdays["Store 3"].apply(store_name)
    print(weekdays)
    weekend_costs = weekend_routes()
    weekends = pd.DataFrame(data = weekend_costs, columns = ['Store 1', 'Store 2', 'Store 3', 'Store 4', 'Cost'])
    weekends["Store 1"] = weekends["Store 1"].apply(store_name)
    weekends["Store 2"] = weekends["Store 2"].apply(store_name)
    weekends["Store 3"] = weekends["Store 3"].apply(store_name)
    weekends["Store 4"] = weekends["Store 4"].apply(store_name)
    print(weekends)

if __name__ == "__main__":
    main()
