import numpy as np
import pandas as pd
import folium
import openrouteservice as ors
import itertools
from pulp import *

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

# print(weekend_demands.loc[50]['Store'])
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
    # Groups of stores split into 5 Auckland regions - South, East, Central, West, North
    south = [1, 22, 23, 24, 25, 26, 40, 41, 44, 49, 59, 60, 63, 65]
    east = [3, 6, 14, 16, 27, 28, 34, 39, 45, 48, 58]
    central = [2, 9, 10, 11, 29, 30, 32, 33, 35, 38, 42, 43, 46, 53, 54]
    west = [5, 13, 15, 17, 18, 19, 20, 37, 51, 52, 55, 57, 61, 62, 64, 66]
    north = [4, 7, 8, 12, 21, 31, 36, 47, 50]
    # Distribution centre
    dist_centre = 56
    # Create a list of regions
    regions = [south, east, central, west, north]
    # Route matrix for region
    routes = np.zeros((0, 3))
    costs = np.zeros(0)
    extra_truck_costs = np.zeros(0)

    # Loop through the regions
    for i in range(len(regions)):
        region = regions[i]
        two_nodes = False
        m = (0,0)
        p = (0,0)
        # For each node in the region
        for route in itertools.combinations(region, 3):
            i1, i2, i3 = route
            # Record demand for each of the nodes
            demand1 = weekday_demands.loc[i1]['Demand']
            demand2 = weekday_demands.loc[i2]['Demand']
            demand3 = weekday_demands.loc[i3]['Demand']
            # Calculate total demand
            total_demand = (demand1 + demand2 + demand3)
            # If total demand is more than 26 the route is not feasible
            if total_demand > 26:
                #i1 and i2
                time0_1 = (durations.loc[i1][dist_centre]) / 60 + 7.5 * demand1
                time1_2 = (durations.loc[i2][i1]) / 60 + 7.5 * demand2
                time2 = (durations.loc[dist_centre][i2]) / 60
                total_time = time0_1 + time1_2 + time2
                if total_time <= 240:
                    cost = total_time * 3.75
                    # pricing if extra trucks from daily freight are required - 2000 for standard 4h shift
                    extra_truck_cost = 2000
                # Otherwise calculate cost with overtime pricing
                else:
                    overtime = total_time - 240
                    cost = 240 * 3.75 + overtime * 4.583
                    # pricing if extra trucks from daily freight are required - 4000 for 8h shift
                    extra_truck_cost = 4000
                # Add route1 and total cost to route matrix
                routes = np.append(routes, np.array([[i1, i2, 0]]), axis=0)
                costs = np.append(costs, cost)
                #i2 and i3
                extra_truck_costs = np.append(extra_truck_costs, extra_truck_cost)
                time0_2 = (durations.loc[i2][dist_centre]) / 60 + 7.5 * demand1
                time2_3 = (durations.loc[i3][i2]) / 60 + 7.5 * demand3
                time3 = (durations.loc[dist_centre][i3]) / 60
                total_time = time0_2 + time2_3 + time3
                if total_time <= 240:
                    cost = total_time * 3.75
                    # pricing if extra trucks from daily freight are required - 2000 for standard 4h shift
                    extra_truck_cost = 2000
                # Otherwise calculate cost with overtime pricing
                else:
                    overtime = total_time - 240
                    cost = 240 * 3.75 + overtime * 4.583
                    # pricing if extra trucks from daily freight are required - 4000 for 8h shift
                    extra_truck_cost = 4000
                # Add route1 and total cost to route matrix
                routes = np.append(routes, np.array([[i2, i3, 0]]), axis=0)
                costs = np.append(costs, cost)
                extra_truck_costs = np.append(extra_truck_costs, extra_truck_cost)
                #i1 and i3
                time1_3 = (durations.loc[i3][i1])/60 + 7.5 * demand3
                total_time = time0_1 + time1_3 + time3
                if total_time <= 240:
                    cost = total_time * 3.75
                    # pricing if extra trucks from daily freight are required - 2000 for standard 4h shift
                    extra_truck_cost = 2000
                # Otherwise calculate cost with overtime pricing
                else:
                    overtime = total_time - 240
                    cost = 240 * 3.75 + overtime * 4.583
                    # pricing if extra trucks from daily freight are required - 4000 for 8h shift
                    extra_truck_cost = 4000
                # Add route1 and total cost to route matrix
                routes = np.append(routes, np.array([[i2, i3, 0]]), axis=0)
                costs = np.append(costs, cost)
                extra_truck_costs = np.append(extra_truck_costs, extra_truck_cost)
                continue
            # Record the total time for each of the nodes - travel from distribution centre + unloading
            # Distribution centre to node1, node2, node3
            time0_1 = (durations.loc[i1][dist_centre]) / 60 + 7.5 * demand1
            # Node1 to node2
            time1_2 = (durations.loc[i2][i1]) / 60 + 7.5 * demand2
            # Node2 to node3
            time2_3 = (durations.loc[i3][i2]) / 60 + 7.5 * demand3
            # Node3 back to distribution centre
            time3 = (durations.loc[dist_centre][i3]) / 60
            # Calculate the total time for route 1: distribution centre - node1 - node2 - node3 - distribution centre
            total_time = time0_1 + time1_2 + time2_3 + time3
            # If the total time is less than or equal to 4h calculate cost using standard truck pricing
            if total_time <= 240:
                cost = total_time * 3.75
                # pricing if extra trucks from daily freight are required - 2000 for standard 4h shift
                extra_truck_cost = 2000
            # Otherwise calculate cost with overtime pricing
            else:
                overtime = total_time - 240
                cost = 240 * 3.75 + overtime * 4.583
                # pricing if extra trucks from daily freight are required - 4000 for 8h shift
                extra_truck_cost = 4000
            # Add route1 and total cost to route matrix
            routes = np.append(routes, np.array([[i1, i2, i3]]), axis=0)
            costs = np.append(costs, cost)
            extra_truck_costs = np.append(extra_truck_costs, extra_truck_cost)
    # Return route matrices
    return routes, costs, extra_truck_costs


def weekend_routes():
    # Groups of stores split into 5 Auckland regions - South, East, Central, West, North
    south = [1, 22, 23, 24, 25, 26, 40, 41, 44, 49]
    east = [3, 6, 14, 16, 27, 28, 34, 39, 45, 48]
    central = [2, 9, 10, 11, 32, 33, 35, 38, 42, 43, 46, 53, 54, 54]
    west = [5, 13, 15, 17, 18, 19, 20, 37, 51, 52, 55]
    north = [4, 7, 8, 12, 21, 31, 36, 47, 50]
    # Distribution centre
    dist_centre = 56
    # Create a list of regions
    regions = [south, east, central, west, north]
    # Route matrix for region
    routes = np.zeros((0, 4))
    costs = np.zeros(0)
    extra_truck_costs = np.zeros(0)

    # Loop through the regions
    for i in range(len(regions)):
        region = regions[i]
        # For each node in the region
        for route in itertools.combinations(region, 4):
            i1, i2, i3, i4 = route
            # Record demand for each of the nodes
            demand1 = weekend_demands.loc[i1]['Demand']
            demand2 = weekend_demands.loc[i2]['Demand']
            demand3 = weekend_demands.loc[i3]['Demand']
            demand4 = weekend_demands.loc[i4]['Demand']
            # Calculate total demand
            total_demand = (demand1 + demand2 + demand3 + demand4)
            # If total demand is more than 26 the route is not feasible
            if total_demand > 26:
                continue
            # Record the total time for each of the nodes - travel from distribution centre + unloading
            # Distribution centre to node1, node2, node3
            time0_1 = (durations.loc[i1][dist_centre]) / 60 + 7.5 * demand1
            # Node1 to node2
            time1_2 = (durations.loc[i2][i1]) / 60 + 7.5 * demand2
            # Node2 to node3
            time2_3 = (durations.loc[i3][i2]) / 60 + 7.5 * demand3
            # Node 3 to node4
            time3_4 = (durations.loc[i4][i3]) / 60 + 7.5 * demand4
            # Node4 back to distribution centre
            time4 = (durations.loc[dist_centre][i4]) / 60
            # Calculate the total time for route 1: distribution centre - node1 - node2 - node3 - distribution centre
            total_time = time0_1 + time1_2 + time2_3 + time3_4 + time4
            # If the total time is less than or equal to 4h calculate cost using standard truck pricing
            if total_time <= 240:
                cost = total_time * 3.75
                extra_truck_cost = 2000
            # Otherwise calculate cost with overtime pricing
            else:
                overtime = total_time - 240
                cost = 240 * 3.75 + overtime * 4.583
                extra_truck_cost = 4000
            # Add route1 and total cost to route matrix
            routes = np.append(routes, np.array([[i1, i2, i3, i4]]), axis=0)
            costs = np.append(costs, cost)
            extra_truck_costs = np.append(extra_truck_costs, extra_truck_cost)
    # Return route matrices
    return routes, costs, extra_truck_costs



def LP_weekday():
    prob = LpProblem("Weekday Routes", LpMinimize)
    # include all stores (get rid of distribution centre)
    store_names = pd.concat([locations.Store[:55], locations.Store[56:]]).reset_index(drop=True)
    # Right hand sides are all equal to one (each node/store has one delivery)
    RHS = pd.Series([1] * 65, index=store_names)
    # put the routes into a series indexed by the route number
    weekday_routes1 = weekday_routes()  # so function only called once to save time
    routes = pd.Series(list(weekday_routes1[0]))
    # variables are the individual routes
    route_vars = list(range(0, 946))
    vars = LpVariable.dicts("Route", route_vars, 0, None, LpBinary)
    vars2 = LpVariable.dicts("Extra Route", route_vars, 0, None, LpBinary)
    # put costs into easy to access variable
    costs = pd.Series(list(weekday_routes1[1]))
    costs2 = pd.Series(list(weekday_routes1[2]))
    # objective function
    prob += lpSum([vars[i] * costs[i] + vars2[i] * costs2[i] for i in route_vars]), "Costs"

    # sort through all routes, so that each node has list of routes it is in
    f = []
    # create empty lists for each store and put it into series
    for it in range(65):
        f.append([])
    # initialise empty arrays for each store
    indexes = list(range(1, 56))  # first 55 stores
    indexes.extend(list(range(57, 67)))  # rest of stores
    stores = pd.Series(f, index=indexes)
    count = 0
    for j in routes:
        stores[round(j[0])].append(count)
        stores[round(j[1])].append(count)
        stores[round(j[2])].append(count)
        count += 1
    # truck availability constraint
    prob += lpSum([vars[i] for i in routes]) <= 60, "Trucks"
    # stores have one delivery constraint
    for i in store_names:
        prob += lpSum([vars[i]]) == 1


def main():
    weekday_feasible_routes, weekday_costs, weekday_extra_costs = weekday_routes()
    weekdays = pd.DataFrame(data=weekday_feasible_routes, columns=['Store 1', 'Store 2', 'Store 3'])
    print(weekdays)
    weekday_cost = pd.Series(weekday_costs)
    weekday_extra = pd.Series(weekday_extra_costs)
    weekend_feasible_routes, weekend_costs, weekend_extra_costs = weekend_routes()
    weekends = pd.DataFrame(data=weekend_feasible_routes, columns=['Store 1', 'Store 2', 'Store 3', 'Store 4'])
    weekend_cost = pd.Series(weekend_costs)
    weekend_extra = pd.Series(weekend_extra_costs)


if __name__ == "__main__":
    main()
