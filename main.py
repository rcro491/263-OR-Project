import numpy as np
import pandas as pd
import folium
import openrouteservice as ors

# read location data
locations = pd.read_csv("WoolworthsLocations.csv")

# make coordinates into list of lists
coords = locations[['Long', 'Lat']]
coords = coords.to_numpy().tolist()

m = folium.Map(location=list(reversed(coords[2])), zoom_start=10)
folium.Marker(list(reversed(coords[0])), popup=locations.Store[0], icon=folium.Icon(color='black')).add_to(m)

for i in range(1, len(coords)):
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
m.save('map.html')
