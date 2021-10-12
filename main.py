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
    east = [3, 6, 14, 16, 27, 28, 34, 39, 45, 48, 58, 59, 60]
    central = [2, 9, 10, 11, 29, 30, 32, 33, 35, 38, 42, 43, 46, 53, 54, 59, 60]
    west = [5, 13, 15, 17, 18, 19, 20, 37, 51, 52, 55, 57, 59, 61, 62, 64, 66]
    north = [4, 7, 8, 12, 21, 29, 30, 31, 36, 47, 50, 59, 60]
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
        m = (0, 0)
        p = (0, 0)
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
                # i1 and i2
                time0_1 = (durations.loc[i1][dist_centre]) / 60 + 7.5 * demand1
                time1_2 = (durations.loc[i2][i1]) / 60 + 7.5 * demand2
                time2 = (durations.loc[dist_centre][i2]) / 60
                # i2 and i3
                time0_2 = (durations.loc[i2][dist_centre]) / 60 + 7.5 * demand2
                time2_3 = (durations.loc[i3][i2]) / 60 + 7.5 * demand3
                time3 = (durations.loc[dist_centre][i3]) / 60
                # i1 and i3
                time1_3 = (durations.loc[i3][i1]) / 60 + 7.5 * demand3
                
                total_time = [0] * 3

                # i1 and i2
                total_time[0] = time0_1 + time1_2 + time2
                
                # i2 and i3
                total_time[1] = time0_2 + time2_3 + time3

                # i1 and i3
                total_time[2] = time0_1 + time1_3 + time3

                for i in range(len(route)):

                    if total_time[i] <= 240:
                        cost = total_time[i] * 3.75
                        # pricing if extra trucks from daily freight are required - 2000 for standard 4h shift
                        extra_truck_cost = 2000
                    # Otherwise calculate cost with overtime pricing
                    else:
                        overtime = total_time[i] - 240
                        cost = 240 * 3.75 + overtime * 4.583
                        # pricing if extra trucks from daily freight are required - 4000 for 8h shift
                        extra_truck_cost = 4000
                    # Add route1 and total cost to route matrix
                    routes = np.append(routes, np.array([[route[i], route[(i+1)%3], 0]]), axis=0)
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
    west = [5, 13, 15, 17, 18, 19, 20, 37, 51, 52, 55, 38, 53, 33]
    north = [4, 7, 8, 12, 21, 31, 36, 47, 50, 38, 53, 43, 35]
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
        # add distribution to one store routes
        for node in region:
            time_to = (durations.loc[node][dist_centre]) / 60 + 7.5 * weekend_demands.loc[node]['Demand']
            time_back = (durations.loc[dist_centre][node]) / 60
            total_time = time_to + time_back
            # If the total time is less than or equal to 4h calculate cost using standard truck pricing
            if total_time <= 240:
                cost = total_time * 3.75
                extra_truck_cost = 2000
            # Otherwise calculate cost with overtime pricing
            else:
                overtime = total_time - 240
                cost = 240 * 3.75 + overtime * 4.583
                extra_truck_cost = 4000
            routes = np.append(routes, np.array([[node, 0, 0, 0]]), axis=0)
            costs = np.append(costs, cost)
            extra_truck_costs = np.append(extra_truck_costs, extra_truck_cost)
    # Return route matrices
    return routes, costs, extra_truck_costs


def LP_weekday():
    # initialise LP problem
    prob = LpProblem("Weekday Routes", LpMinimize)

    weekday_routes1 = weekday_routes()  # so function only called once to save time

    routes = pd.Series(list(weekday_routes1[0]))

    # variables are the individual routes
    route_vars = list(range(0, 3672))
    vars = LpVariable.dicts("Route", route_vars, cat=const.LpBinary)  # normal routes
    vars2 = LpVariable.dicts("Extra Route", route_vars, cat=const.LpBinary)  # extra routes if needed (Daily Freight)

    costs = pd.Series(list(weekday_routes1[1]))  # costs of normal routes
    costs2 = pd.Series(list(weekday_routes1[2]))  # costs of extra routes

    # objective function
    prob += lpSum([vars[i] * costs[i] + vars2[i] * costs2[i] for i in route_vars]), "Costs"

    # sort through all routes, so that each node has list of routes it is in
    f = []  # create empty lists for each store and put it into series
    for it in range(65):
        f.append([])  # initialise empty arrays for each store
    # splice indexes so distribution centre is not included
    indexes = list(range(1, 56))  # first 55 stores
    indexes.extend(list(range(57, 67)))  # rest of stores
    stores = pd.Series(f, index=indexes)

    # place all route numbers into the node they carry
    count = 0
    for j in routes:
        # check if route only contains two stores
        if j[2] == 0:
            stores[round(j[0])].append(count)
            stores[round(j[1])].append(count)
            count += 1
        else:
            stores[round(j[0])].append(count)
            stores[round(j[1])].append(count)
            stores[round(j[2])].append(count)
            count += 1

    # truck availability constraint
    prob += lpSum([vars[i] for i in vars]) <= 60, "Trucks"

    # stores have one delivery constraint
    count = 1
    for k in indexes:
        prob += lpSum([vars[j] + vars2[j] for j in stores[k]]) == 1
        count += 1

    # solve LP
    prob.writeLP('Weekdays.lp')
    prob.solve()

    print_message = False
    if print_message is True:
        # The status of the solution is printed to the screen
        print("Status:", LpStatus[prob.status])

        # Each of the variables is printed with its resolved optimum value
        for v in prob.variables():
            if v.varValue != 0:
                print(v.name, "=", v.varValue)

        # The optimised objective function value of Ingredients pue is printed to the screen
        print("Total cost from Routes = ", value(prob.objective))


def LP_weekend():
    # nodes with deliveries for weekends
    south = [1, 22, 23, 24, 25, 26, 40, 41, 44, 49]
    east = [3, 6, 14, 16, 27, 28, 34, 39, 45, 48]
    central = [2, 9, 10, 11, 32, 33, 35, 38, 42, 43, 46, 53, 54]
    west = [5, 13, 15, 17, 18, 19, 20, 37, 51, 52, 55]
    north = [4, 7, 8, 12, 21, 31, 36, 47, 50]
    weekend_nodes = south + east + central + west + north

    # initialise problem
    prob = LpProblem("Weekend Routes", LpMinimize)

    weekend_routes1 = weekend_routes()  # so function only called once to save time

    routes = pd.Series(list(weekend_routes1[0]))

    # variables are the individual routes
    route_vars = list(range(0, 3198))
    vars = LpVariable.dicts("Route", route_vars, cat=const.LpBinary)  # normal routes
    vars2 = LpVariable.dicts("Extra Route", route_vars, cat=const.LpBinary)  # extra routes (Daily Freight)

    costs = pd.Series(list(weekend_routes1[1]))  # normal costs
    costs2 = pd.Series(list(weekend_routes1[2]))  # extra costs

    # objective function
    prob += lpSum([vars[i] * costs[i] + vars2[i] * costs2[i] for i in route_vars]), "Costs"

    # sort through all routes, so that each node has list of routes it is in
    f = []  # create empty lists for each store and put it into series
    for it in range(53):
        f.append([])    # initialise empty arrays for each store
    stores = pd.Series(f, index=weekend_nodes)

    # place all route numbers into the node they carry
    count = 0
    for j in routes:
        # check if route contains single store
        if j[1] == 0:
            stores[round(j[0])].append(count)
            count += 1
        else:
            stores[round(j[0])].append(count)
            stores[round(j[1])].append(count)
            stores[round(j[2])].append(count)
            stores[round(j[3])].append(count)
            count += 1

    # truck availability constraint
    prob += lpSum([vars[i] for i in vars]) <= 60, "Trucks"

    # stores have one delivery constraint
    count = 1
    for k in weekend_nodes:
        prob += lpSum([vars[j] + vars2[j] for j in stores[k]]) == 1
        count += 1

    # solve LP
    prob.writeLP('Weekends.lp')
    prob.solve()

    print_message = False
    if print_message is True:
        # The status of the solution is printed to the screen
        print("Status:", LpStatus[prob.status])

        # Each of the variables is printed with its resolved optimum value
        for v in prob.variables():
            if v.varValue != 0:
                print(v.name, "=", v.varValue)

        # The optimised objective function value of Ingredients pue is printed to the screen
        print("Total cost from Routes = ", value(prob.objective))

def traffic(duration):
    """
    Uncertainty in durations between nodes
    """
    # Values to edit 
    # add three minutes
    time = np.random.normal(1.2*duration, 0.5*duration) + 180
    return time

def simulate_weekdays(routes, n, df, a=3):
    """
    Inputs: Routes = array of routes that have been selected
            n = number of simulations to be done
            df = dataframe of generated routes to pull selected routes from
            a = standard deviation in store demands, calculated to be approximately 2.6
    Output: costs = array of costs of routing for each simulation, length n

    """
    # index for distribution centre is 56
    dist_centre = 56
    # initialise cost array 
    costs = [0]*n
    
    # run n simulations
    for j in range(n):
        # for each route in the selected routes
        for route in routes:
            # pull stores from each route
            nodes = df.iloc[route,]

            demand = [0]*len(nodes)
            total_demand = 0

            # generate variation in demand
            for i in range(len(nodes)):
                if nodes[i] != 0:
                    demand[i] = round(np.random.normal(weekday_demands.loc[nodes[i]]['Average'], a))
                    total_demand += demand[i]

            # Make into integers so can use for indexing
            n1 = int(nodes[0])
            n2 = int(nodes[1])
            n3 = int(nodes[2])
            
            # if total demand on a route > 26:
            # need to factor in cost of extra truck
            if total_demand > 26:
                # Start generating variation in durations
                time0_1 = (traffic(durations.loc[n1][dist_centre])) / 60
                time1_2 = (traffic(durations.loc[n2][n1])) / 60
                time2 = (traffic(durations.loc[dist_centre][n2])) / 60
                time0_2 = (traffic(durations.loc[n2][dist_centre])) / 60
                # If route contains three nodes
                if n3 != 0:
                    # store 6 values, three for normal routes, three for extra trucks
                    total_time = [[0, 0]]*3
                    # cost of each pair
                    c = [0] * 3

                    time2_3 = (traffic(durations.loc[n3][n2])) / 60
                    time3 = (traffic(durations.loc[dist_centre][n3])) / 60
                    time1_3 = (traffic(durations.loc[n3][n1])) / 60

                    total_time[0] = [time0_1+(demand[0]*7.5)+time1_2+(demand[1]*7.5)+time2, (time3*2)+(demand[2]*7.5)]
                    total_time[1] = [time0_2 + time2_3 + time3, (time0_1*2)+(demand[0]*7.5)]
                    total_time[2] = [time0_1+(demand[0]*7.5)+ time1_3+ (7.5 * demand[2]) + time3, (time0_2*2)+(demand[1]*7.5)]
                # if only two nodes are visited on route
                else:
                    total_time = [[0, 0]]*2
                    c = [0]*2

                    total_time[0] = [time0_1*2+7.5*demand[0], time0_2*2+7.5*demand[1]]
                    total_time[1] = [time0_2*2+7.5*demand[1], time0_1*2+7.5*demand[0]]
                
                # find cost of the routes with just two nodes by iteration
                # Choose lowest cost and add cost for extra to go to the third node
                for i in range(len(total_time)):
                    
                    # Calculate costs for route to visit only two nodes
                    if total_time[i][1] <= 240:
                        c[i] = total_time[i][1] * 3.75
                     # Otherwise calculate cost with overtime pricing
                    else:
                        overtime = total_time[i][1] - 240
                        c[i] = 240 * 3.75 + overtime * 4.583

                    # add extra costs for route to third node
                    if total_time[i][0] <=240:
                        c[i] += 2000
                    else:
                        c[i] += 4000
                    
                    # Choose the smallest cost to add
                    costs[j] += min(c)

            else: # demand does not exceed 26
                time0_1 = (traffic(durations.loc[n1][dist_centre])) / 60 + 7.5 * demand[0]
                
                # If route contains three nodes
                if n3 != 0 & n2 !=0:
                    time1_2 = (traffic(durations.loc[n2][n1])) / 60 + 7.5 * demand[1]
                    time2_3 = (traffic(durations.loc[n3][n2])) / 60 + 7.5 * demand[2]
                    time3 = (traffic(durations.loc[dist_centre][n3])) / 60
                # If route contain two nodes
                elif n2 != 0:
                    time1_2 = (traffic(durations.loc[n2][n1])) / 60 + 7.5 * demand[1]
                    time2_3 = 0
                    time3 = (traffic(durations.loc[n2][dist_centre])) / 60
                # If route contains only one node
                else:
                    time1_2 = 0
                    time2_3 = 0
                    time3 = (traffic(durations.loc[dist_centre][n1])) / 60
                    
                # Calculate the total time for route 1
                total_time = time0_1 + time1_2 + time2_3 + time3
                # If the total time is less than or equal to 4h calculate cost using standard truck pricing
                if total_time <= 240:
                    cost = total_time * 3.75
                # Otherwise calculate cost with overtime pricing
                else:
                    overtime = total_time - 240
                    cost = 240 * 3.75 + overtime * 4.583
                # add cost to total for this route
                costs[j] += cost
    return costs

def simulate_weekends(routes, n, df, a=2):
    """
    Inputs: Routes = array of routes that have been selected
            n = number of simulations to be done
            df = dataframe of generated routes to pull selected routes from
            a = standard deviation for demands, calculated from given data to be approximately 1.5
    Output: costs = array of costs of routing for each simulation, length n

    """
    # distribution centre index is 56
    dist_centre = 56
    # initialise cost array 
    costs = [0]*n
    
    # run n simulations
    for j in range(n):
        # for each route in the selected routes
        for route in routes:
            # pull stores from each route
            nodes = df.iloc[route,]

            # Initialise demand array
            demand = [0]*len(nodes)
            total_demand = 0

            # generate variation in demand
            for i in range(len(nodes)):
                if nodes[i] != 0:
                    demand[i] = round(np.random.normal(weekend_demands.loc[nodes[i]]['Average'], a))
                    total_demand += demand[i]
                    # demand array will now hold the time it takes to unload the respective demands
                    demand[i] = demand[i]*7.5

            # each store in node
            n1 = int(nodes[0])
            n2 = int(nodes[1])
            n3 = int(nodes[2])
            n4 = int(nodes[3])

            # if total demand on a route > 26:
            if total_demand > 26:
                # Construct route durations from each node to every other
                time0_1 = (traffic(durations.loc[n1][dist_centre])) / 60
                time0_2 = (traffic(durations.loc[n2][dist_centre])) / 60
                time0_3 = (traffic(durations.loc[n3][dist_centre])) / 60
                time0_4 = (traffic(durations.loc[n4][dist_centre])) / 60
                time1_2 = (traffic(durations.loc[n2][n1])) / 60
                time2_3 = (traffic(durations.loc[n3][n2])) / 60
                time3_4 = (traffic(durations.loc[n4][n3])) / 60
                time1_3 = (traffic(durations.loc[n3][n1])) / 60
                time1_4 = (traffic(durations.loc[n4][n1])) / 60
                time2_4 = (traffic(durations.loc[n4][n1])) / 60

                total_time = [[0, 0]]*3 
                # Did originally have it total_time = [[0, 0]]*7 to include routes commented out below
                # Changed bc cannot guarantee that the three nodes visited also have a demand < 26
                # and the change did not impact the results very much
                # If changed back, would also have to change c = [0]*14
                c = [0]*6

                # Find variations of nodes visited on route and by extra truck
                #total_time[0] = [time0_1+demand[0]+time1_2+demand[1]+time2_3+demand[2]+time0_3, time0_4*2+demand[3]]
                #total_time[1] = [time0_1+demand[0]+time1_2+demand[1]+time2_4+demand[3]+time0_4, time0_3*2+demand[2]]
                #total_time[2] = [time0_1+demand[0]+time1_3+demand[2]+time3_4+demand[3]+time0_4, time0_2*2+demand[1]]
                #total_time[3] = [time0_2+demand[1]+time2_3+demand[2]+time3_4+demand[3]+time0_4, time0_1*2+demand[0]]
                total_time[0] = [time0_1+demand[0]+time1_2+demand[1]+time0_2, time0_3+demand[2]+time3_4+demand[3]+time0_4]
                total_time[1] = [time0_1+demand[0]+time1_3+demand[2]+time0_3, time0_2+demand[1]+time2_4+demand[3]+time0_4]
                total_time[2] = [time0_1+demand[0]+time1_4+demand[3]+time0_4, time0_2+demand[1]+time2_3+demand[2]+time0_3]

                for i in range(len(total_time)):
                    # find cost of all routes, with different nodes in 
                    if total_time[i][0] <= 240:
                        # cost of normal truck
                        c[i] = total_time[i][0] * 3.75
                        # cost of extra truck
                        c[i+len(total_time)] += 2000
                        if total_time[i][1] <= 240:
                            c[i+len(total_time)] = total_time[i][1] * 3.75
                            c[i] += 2000
                        else:
                            overtime = total_time[i][1] - 240
                            c[i+len(total_time)] += 240 * 3.75 + overtime * 4.583
                            c[i] += 4000
                     # Otherwise calculate cost with overtime pricing
                    else:
                        overtime = total_time[i][0] - 240
                        c[i] = 240 * 3.75 + overtime * 4.583
                        # if over time, extra cost will be $4000
                        c[i+len(total_time)] += 4000
                        if total_time[i][1] <= 240:
                            c[i+len(total_time)] += total_time[i][1] * 3.75
                            c[i] += 2000
                        else:
                            overtime = total_time[i][1] - 240
                            c[i+len(total_time)] += 240 * 3.75 + overtime * 4.583
                            c[i] += 4000
                    # choose least expensive route
                    costs[j] += min(c)

            else:
                # Distribution centre to node1, constant despite other nodes
                time0_1 = (durations.loc[n1][dist_centre]) / 60 + demand[0]
                
                # If four nodes are visited en route
                if n4!= 0 & n3 != 0 & n2 !=0:
                    time1_2 = (traffic(durations.loc[n2][n1])) / 60 + demand[1]
                    time2_3 = (traffic(durations.loc[n3][n2])) / 60 + demand[2]
                    time3_4 = (traffic(durations.loc[n4][n3])) / 60 + demand[3]
                    # Node4 back to distribution centre
                    time4 = (traffic(durations.loc[dist_centre][n4])) / 60
                elif n3 != 0: # If route has three nodes
                    time1_2 = (traffic(durations.loc[n2][n1])) / 60 + demand[1]
                    time2_3 = (traffic(durations.loc[n3][n2])) / 60 + demand[2]
                    time3_4 = 0
                    # Node3 back to distribution centre
                    time4 = (traffic(durations.loc[dist_centre][n3])) / 60
                elif n2 != 0: # If route contains two nodes
                    time1_2 = (traffic(durations.loc[n2][n1])) / 60 + demand[1]
                    time2_3 = 0
                    time3_4 = 0
                    # Node2 back to distribution centre
                    time4 = (traffic(durations.loc[dist_centre][n2])) / 60
                else: # If only one node in the route
                    time1_2 = 0
                    time2_3 = 0
                    time3_4 = 0
                    # Node1 back to distribution centre
                    time4 = (traffic(durations.loc[dist_centre][n1])) / 60

                # Calculate the total time for route
                total_time = time0_1 + time1_2 + time2_3 + time3_4 + time4
                # If the total time is less than or equal to 4h calculate cost using standard truck pricing
                if total_time <= 240:
                    cost = total_time * 3.75
                # Otherwise calculate cost with overtime pricing
                else:
                    overtime = total_time - 240
                    cost = 240 * 3.75 + overtime * 4.583
                # Add cost to array
                costs[j] += cost
        
    return costs

def main():
    weekday_feasible_routes, weekday_costs, weekday_extra_costs = weekday_routes()
    weekdays = pd.DataFrame(data=weekday_feasible_routes, columns=['Store 1', 'Store 2', 'Store 3'])
    weekday_cost = pd.Series(weekday_costs)
    weekday_extra = pd.Series(weekday_extra_costs)
    weekend_feasible_routes, weekend_costs, weekend_extra_costs = weekend_routes()
    weekends = pd.DataFrame(data=weekend_feasible_routes, columns=['Store 1', 'Store 2', 'Store 3', 'Store 4'])
    weekend_cost = pd.Series(weekend_costs)
    weekend_extra = pd.Series(weekend_extra_costs)
    LP_weekend()
    LP_weekday()
    
    # Hardcode selected routes 
    weekday_selected = [1082, 1247, 1507, 1531, 1744, 189, 1944, 2031, 21, 2382, 2573, 2628, 2676, 2977, 3155, 3243, 332, 3383, 3417, 3534, 354, 3660, 465, 602, 609, 747, 833, 853]
    weekend_selected = [1449, 1468, 2008, 209, 211, 212, 2433, 249, 2529, 2797, 3116, 423, 431, 437, 49, 514, 852]

    
    costs_weekday = simulate_weekdays(weekday_selected, 1000, weekdays)
    print(np.mean(costs_weekday))

    costs_weekend = simulate_weekends(weekend_selected, 1000, weekends)
    print(np.mean(costs_weekend))
    
    # can then find mean, ttest, 95% int, error rate


if __name__ == "__main__":
    main()
