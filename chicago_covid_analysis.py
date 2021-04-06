#!/usr/bin/env python
# coding: utf-8

# # Chicago ZIP codes and COVID-19 cases

# In this notebook, we will explore Chicago area (by ZIP code) and apply ML techniques for clustering depending on the category of nearby venues provided by the Foursquare API. Additionally, we will try to find some correlation between each cluster and the number of COVID-19 cases detected.

# First, let's import *Pandas* and *Numpy*

# In[1]:


import pandas as pd
import numpy as np


# ## 1. Importing and preparing data

# ### 1.1 Chicago COVID-19 data by ZIP code

# We will first import data of COVID-19 cases in the area of Chicago by ZIP code

# In[2]:


# Defining the path to COVID-19 data for the city of chicago
url = "https://data.cityofchicago.org/api/views/yhhz-zm2v/rows.csv"


# In[3]:


# Reading data using Pandas
covid_data = pd.read_csv(url)
covid_data.head()


# New, let's do some cleaning and prepare data for processing

# In[4]:


# Removing "unknown data due to missing ZIP codes"
covid_data = covid_data[~covid_data['ZIP Code'].str.contains("Unknown")]
covid_data.reset_index(inplace=True,drop=True)
covid_data.shape


# In[5]:


# Display dataframe columns
covid_data.columns


# For our needs, we will only use cumulative data from COVID cases, and especially the rate per 100,000 population

# In[6]:


# Droping unnecessary columns for our analysis
todrop = ['Week Number','Week Start', 'Week End' ,'Population','Cases - Weekly', 'Cases - Cumulative', 'Case Rate - Weekly' , 'Tests - Weekly', 'Tests - Cumulative', 'Test Rate - Weekly',
       'Test Rate - Cumulative', 'Percent Tested Positive - Weekly',
       'Percent Tested Positive - Cumulative', 'Deaths - Weekly',
       'Deaths - Cumulative', 'Death Rate - Weekly', 'Death Rate - Cumulative','Row ID']

covid_data = covid_data.drop(todrop,axis=1)
covid_data.head()


# In[7]:


# Renaming columns for consistency
covid_data = covid_data.rename(columns={'ZIP Code' : 'ZipCode',
                              'Case Rate - Cumulative' : 'CaseRateCumulative' , 
                               'ZIP Code Location' : 'Coordinates'})
covid_data.head()


# Let's extract Latitudes and Longitudes from the "Coordinates" colum :

# In[8]:


#Extracting Latitudes and Longitudes from coordinates
covid_data.Coordinates = covid_data.Coordinates.str.extract(r'([-+]\d*[.]\d*\s\d*[.]\d*)')

#Creating Latitudes column
covid_data['Latitude'] = covid_data.Coordinates.str.extract(r'(\s\d*[.]\d*)')

#Creating Longitudes column
covid_data['Longitude'] = covid_data.Coordinates.str.extract(r'([-+]\d*[.]\d*)')

#Removing blank spaces from Latitudes data
covid_data['Latitude'] = covid_data['Latitude'].str.replace(" ","")

#Droping the Coordinates column
covid_data = covid_data.drop(['Coordinates'],axis=1)

covid_data.head()


# In[9]:


#Droping rows that contain one or more NaN values
covid_data = covid_data.dropna()
covid_data = covid_data.reset_index(drop=True)


# Let's group the data by ZIP code. We will take the "max" of the cumulative rate of cases to obtain the total number of cases up until now

# In[10]:


#Grouping data by ZIP code
covid_data_grouped = covid_data.groupby(['ZipCode']).max()
covid_data_grouped.reset_index(inplace=True)
covid_data_grouped.head()


# Let's get the geographical coordinates of Chicago

# In[11]:


# Coverting coordinates to floats
covid_data_grouped['Latitude'] = covid_data_grouped['Latitude'].astype("float")
covid_data_grouped['Longitude'] = covid_data_grouped['Longitude'].astype("float")


# In[12]:


# Getting Chicago's Latitude and Longitude
from geopy.geocoders import Nominatim

# City for which we want the Latitude and Longitude
address = 'Chicago, IL'
geolocator = Nominatim(user_agent="ny_explorer")

# Retrieving latitude and longitude
location = geolocator.geocode(address)
chic_latitude = location.latitude
chic_longitude = location.longitude

# Printing results
print('The geograpical coordinate of Chicago are :\n\nLatitude : {}, Longitude : {}.'.format(chic_latitude, chic_longitude))


# Now let's create a map of Chicago using it's coordinates and the coordinates of each ZIP code

# In[13]:


#Importing necessary libraries

import matplotlib.cm as cm
import matplotlib.colors as colors
import folium
import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe


# In[14]:


# Creating map of Chicago using latitude and longitude values
map_chicago = folium.Map(location=[chic_latitude, chic_longitude], zoom_start=11)

# Adding markers to map to show ZIP codes
for lat, lng, label in zip(covid_data_grouped['Latitude'], covid_data_grouped['Longitude'], covid_data_grouped['ZipCode']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_chicago)  
    
# Displaying the map
map_chicago


# ### 1.2 Chicago venues data using Foursquare API

# The Foursquare API allows to explore venues nearby a specific location (given it's coordinates) within a desired radius. First, we need to define the credentials :

# In[15]:


# Setting Foursquare API credentials and parameters 

CLIENT_ID = '' # your Foursquare ID
CLIENT_SECRET = '' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version
LIMIT = 100 # A default Foursquare API limit value
radius = 500 # perimeter to lookup venues (in meters)
venues_cat = {'4d4b7104d754a06370d81259' : 'Arts & Entertainment',
              '4d4b7105d754a06372d81259' : 'College & University',
              '4d4b7105d754a06373d81259' : 'Event',
              '4d4b7105d754a06374d81259' : 'Food',
              '4d4b7105d754a06376d81259' : 'Nightlife Spot',
              '4d4b7105d754a06377d81259' : 'Outdoors & Recreation',
              '4d4b7105d754a06375d81259' : 'Professional & Other Places',
              '4e67e38e036454776db1fb3a' : 'Residence',
              '4d4b7105d754a06378d81259' : 'Shop & Service',
              '4d4b7105d754a06379d81259' : 'Travel & Transport'} 


# The following function allows to make a series of calls to retrieve a number of venues and their data around specified coordinates

# Let's use the defined function to return a dataframe with the venues around Chicago's ZIP Codes

# In[16]:



def getNearbyVenues(zipcodes, latitudes, longitudes, radius=500):
    
    venues_list=[]
    
    for zipcode, lat, lng in zip(zipcodes, latitudes, longitudes):
    
        for cat_id,cat_label in venues_cat.items():

            # create the API request URL
            url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&categoryId={}'.format(
                CLIENT_ID, 
                CLIENT_SECRET, 
                VERSION, 
                lat, 
                lng, 
                radius, 
                cat_id)

            # make the GET request
            results = requests.get(url).json()["response"]['groups'][0]['items']

            # return only relevant information for each nearby venue
            venues_list.append([(
                zipcode, 
                lat, 
                lng, 
                v['venue']['name'], 
                v['venue']['location']['lat'], 
                v['venue']['location']['lng'],  
                cat_label) for v in results])

        nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
        nearby_venues.columns = ['ZipCode', 
                      'Latitude', 
                      'Longitude', 
                      'VenueName', 
                      'VenueLatitude', 
                      'VenueLongitude', 
                      'VenueCategory']
    
    return(nearby_venues)


# Let's use the defined function to return a dataframe with the venues around Chicago's ZIP Codes

# In[17]:


# Retrieving venues data
chicago_venues = getNearbyVenues(zipcodes=covid_data_grouped['ZipCode'],
                                   latitudes=covid_data_grouped['Latitude'],
                                   longitudes=covid_data_grouped['Longitude'])

chicago_venues.head(10)


# Let's check the size of the resulting datframe

# In[18]:


# Venues dataframe size
print(chicago_venues.shape)


# In[22]:


# Getting the number of venues per ZIP Code
chicago_venues.groupby('ZipCode').count()


# Now, let's check how many venues were returned for each ZipCode

# In[23]:


# Number of unique categories 

print('There are {} uniques categories.'.format(len(chicago_venues['VenueCategory'].unique())))


# ## 2. Analysis of each ZIP Code

# In order to exploit the data, we will first use the one hot encoding technique for venue categories

# In[24]:


# one hot encoding
chicago_onehot = pd.get_dummies(chicago_venues[['VenueCategory']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
chicago_onehot['ZipCode'] = chicago_venues['ZipCode']

# move neighborhood column to the first column
fixed_columns = [chicago_onehot.columns[-1]] + list(chicago_onehot.columns[:-1])
chicago_onehot = chicago_onehot[fixed_columns]

chicago_onehot.head()


# Next, let's group rows by Zip Code and by taking the mean of the frequency of occurrence of each category

# In[28]:


# Group by ZIP code to visualize the total of each venue category

chicago_grouped = chicago_onehot.groupby("ZipCode").mean().reset_index()
chicago_grouped.head(20)


# It is now possible to identify the most common venues per ZIP Code

# In[29]:


# Defining a function that returns the most common venues per ZIP Code
def return_most_common_venues(row, num_top_venues):
    
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[30]:


# retrieve the top 5 venues
num_top_venues = 5

indicators = ['st', 'nd', 'rd'] # for 1st, 2nd and 3rd, otherwise it's 4th, 5th, etc...

# create columns according to number of top venues
columns = ['ZipCode']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
zipcodes_venues_sorted = pd.DataFrame(columns=columns)
zipcodes_venues_sorted['ZipCode'] = chicago_grouped['ZipCode']

for ind in np.arange(chicago_grouped.shape[0]):
    zipcodes_venues_sorted.iloc[ind, 1:] = return_most_common_venues(chicago_grouped.iloc[ind, :], num_top_venues)

zipcodes_venues_sorted.head()


# ## 3. Clustering Chicago's neighborhoods per ZIP Code

# Now that we have prepared our data, we can proceed to further analysis by clustering Chicago's neighborhoods based on the percentage of each venue category per ZIP Code. In this section, we will use the K-Means clustering algorithm

# In[31]:


#Import the necessary package

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# In[32]:


# Droping the 'ZipCode' column from the dataframe
chicago_grouped_clustering = chicago_grouped.drop('ZipCode', 1)


# First, let's plot plot the evolution of the sum of squared distances to compute the optimal number of clusters

# In[47]:


# Plot distortions for different numbers of clusters
distortions = []
K = range(3,12)

for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(chicago_grouped_clustering)
    distortions.append(km.inertia_)
    
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortions')
plt.title('Elbow Method For Optimal k')
plt.show()


# As you can see, the elbow method is unclear and doesn't allow us to define the optimal K as the "Elbow" doesn't appear clearly.

# Alternatively, we can use the Silhouette Score method. An optimal K maximizes the Silhouette Score :

# In[51]:


# Plot silhouette score for different numbers of clusters
from sklearn.metrics import silhouette_score

sil = []
kmax = 12

# dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
for k in range(3, kmax+1):
  kmeans = KMeans(n_clusters = k).fit(chicago_grouped_clustering)
  labels = kmeans.labels_
  sil.append(silhouette_score(chicago_grouped_clustering, labels, metric = 'euclidean'))

plt.plot(range(3, kmax+1), sil, 'bx-')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for optimal K')
plt.show()


# It appears clearly now that the optimal K is 4. Let's use it from now on.

# In[53]:


# run k-means clustering using 4 clusters
opt_k = 4
kmeans = KMeans(n_clusters=opt_k, random_state=0).fit(chicago_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 


# Let's create a new dataframe that includes the cluster as well as the top 10 venues for each ZIP Code

# In[54]:


# add clustering labels
zipcodes_venues_sorted.insert(0, 'ClusterLabel', kmeans.labels_)

chicago_merged = covid_data_grouped

chicago_merged = chicago_merged.join(zipcodes_venues_sorted.set_index('ZipCode'), how = 'inner' , on='ZipCode')

chicago_merged.head() # check the last columns!


# Now, let's see what caracterizes each cluster

# In[55]:


chicago_grouped.insert(0, 'ClusterLabel', kmeans.labels_)


# ### Cluster 0

# In[56]:


chicago_merged[chicago_merged['ClusterLabel'] == 0].head(10)


# In[57]:


cluster1 = chicago_grouped[ chicago_grouped['ClusterLabel'] == 0]
cluster1.describe()


# ### Cluster 1

# In[58]:


chicago_merged[chicago_merged['ClusterLabel'] == 1].head(10)


# In[59]:


cluster2 = chicago_grouped[ chicago_grouped['ClusterLabel'] == 1]
cluster2.describe()


# ### Cluster 2

# In[60]:


chicago_merged[chicago_merged['ClusterLabel'] == 2].head(10)


# In[61]:


cluster3 = chicago_grouped[ chicago_grouped['ClusterLabel'] == 2]
cluster3.describe()


# ### Cluster 3

# In[62]:


chicago_merged[chicago_merged['ClusterLabel'] == 3].head(10)


# In[63]:


cluster4 = chicago_grouped[ chicago_grouped['ClusterLabel'] == 3]
cluster4.describe()


# ### Clusters summary

# | Cluster Label | Total Zip Codes | 1st Most Common Venue | 2nd Most Common Venue | 3rd Most Common Venue| 4th Most Common Venue| 5th Most Common Venue|
# | :------ | :-:| :----------: | :--------------:|:---------------: |:--------------:|----------------:|
# |0| 30 | Shop & Service (**29.94%**)| Professional & Other Places (**26.44%**)| Food **(16.74%)**|Outdoors & Recreation (**8.21%**)|Travel & Transport (**5.45%**)|
# |1| 21 | Shop & Service (**16.22%**)| Professional & Other Places (**17.78%**)| Food (**13.41%**)|Travel & Transport (**12.80%**)|Outdoors & Recreation (**10.15%**)|
# |2| 9  | Professional & Other Places (**50.04%**)| Shop & Service (**16.22%**)| Food (**11.53%**)|Nightlife Spot (**4.72%**)|Outdoors & Recreation (**4.27%**)|
# |3| 1  | Shop & Service (**60%**)| Food (**20%**)| Travel & Transport (**20%**)|

# Know, let's visualize the resulting clusters using the Folium library

# In[109]:


# Number of clusters
kclusters = 4

# create map
map_clusters = folium.Map(location=[chic_latitude, chic_longitude], zoom_start=10)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = ['darkred','darkblue','darkgreen','black']
fill_colors = ['red','blue','green','gray']
for lat, lon, poi, cluster in zip(chicago_merged['Latitude'], chicago_merged['Longitude'], chicago_merged['ZipCode'], chicago_merged['ClusterLabel']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        location=[lat, lon],
        radius=4,
        popup=label,
        color=markers_colors[cluster],
        fill=True,
        fill_color=fill_colors[cluster],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# ## 4. Clusters VS COVID-19 Cases

# Let's visualize the profile of covid cases rate per 100K population using a histogram

# In[69]:


# visualization libraries
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot

# plot histogram
plt.pyplot.hist(chicago_merged["CaseRateCumulative"],bins=4)

# set x/y labels and plot title
plt.pyplot.xlabel("Case Rate Cumulative")
plt.pyplot.ylabel("Count")
plt.pyplot.title("Case Rates for Chicago Postal Codes")


# Now let's assign categories according to covid rates

# In[70]:


# Creating bins
bins = np.linspace(min(chicago_merged["CaseRateCumulative"]), max(chicago_merged["CaseRateCumulative"]), 5)
bins


# In[72]:


# Defining categories for the created bins
rate_cat = ['Low', 'Average Minus', 'Average Plus', 'High']


# Now, let's create a choropleth map to see how covid rates for each zip code change between clusters

# In[73]:


# Creating a 'Case Rate Categories' column in chicago_merged
chicago_merged['CaseRateCategories'] = pd.cut(chicago_merged['CaseRateCumulative'], bins, labels=rate_cat, include_lowest=True )
chicago_merged[['CaseRateCumulative','CaseRateCategories']].head()


# In[110]:


#Choropleth map with case rate per Zip Code
chicago_geo = r'chicago_zipcodes.geojson'
chicago_map = folium.Map(location = [chic_latitude, chic_longitude], zoom_start = 10)

chicago_map.choropleth(
    geo_data=chicago_geo,
    data=chicago_merged,
    columns=['ZipCode','CaseRateCumulative'],
    key_on='feature.properties.zip',
    fill_color='RdPu', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='Case Rate per 100000 citizens'
)
# add markers to the map
markers_colors = ['darkred','darkblue','darkgreen','black']
fill_colors = ['red','blue','green','gray']
for lat, lon, poi, cluster in zip(chicago_merged['Latitude'],chicago_merged['Longitude'], chicago_merged['ZipCode'], chicago_merged['ClusterLabel']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=3,
        popup=label,
        color=markers_colors[cluster],
        fill=True,
        fill_color=fill_colors[cluster],
        fill_opacity=0.7).add_to(chicago_map)
       
# display map
chicago_map


# Doing this, we notice that Cumulative Covid rates are not homogeneous withn clusters (See Cluster 0 for example). Now let's compute the average covid rate for each cluster

# In[75]:


# Creating a list with the mean for each cluster
means = []
for index,row in chicago_merged.iterrows():
    cluster = row['ClusterLabel']
    means.append(chicago_merged.groupby("ClusterLabel").mean().loc[cluster,'CaseRateCumulative'])


# In[76]:


# Creating a new column with the mean rate per cluster
chicago_merged['meanCaseRateCluster'] = pd.DataFrame(means)


# Now, let's try and visualize a similar map, but instead of using the covid rate per ZIP Code, we will display the mean covid rate per cluster

# In[117]:


#Choropleth map with average case rate per cluster

chicago_geo = r'chicago_zipcodes.geojson'
chicago_map = folium.Map(location = [chic_latitude, chic_longitude], zoom_start = 10)

chicago_map.choropleth(
    geo_data=chicago_geo,
    data=chicago_merged,
    columns=['ZipCode','meanCaseRateCluster'],
    key_on='feature.properties.zip',
    fill_color='RdPu', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='Case Rate per 100000 citizens (Average per cluster)',
)
# add markers to the map
markers_colors = ['darkred','darkblue','darkgreen','black']
fill_colors = ['red','blue','green','gray']
for lat, lon, poi, cluster in zip(chicago_merged['Latitude'],chicago_merged['Longitude'], chicago_merged['ZipCode'], chicago_merged['ClusterLabel']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=3,
        popup=label,
        color=markers_colors[cluster],
        fill=True,
        fill_color=fill_colors[cluster],
        fill_opacity=0.7).add_to(chicago_map)
       
# display map
chicago_map


# Let's try to interprete these results by grouping chicago's data by cluster 

# In[104]:


# Group by cluster
chicago_grouped_cluster = chicago_grouped.groupby("ClusterLabel").mean()
chicago_grouped_cluster.reset_index(drop=False,inplace=True)
chicago_grouped_cluster


# Now, let's retrieve the top ten venues per Cluster

# In[91]:


# retrieve the top 10 venues
num_top_venues = 10

indicators = ['st', 'nd', 'rd'] # for 1st, 2nd and 3rd, otherwise it's 4th, 5th, etc...

# create columns according to number of top venues
columns = ['ClusterLabel']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
cluster_venues_sorted = pd.DataFrame(columns=columns)
cluster_venues_sorted['ClusterLabel'] = chicago_grouped_cluster['ClusterLabel']

for ind in np.arange(chicago_grouped_cluster.shape[0]):
    cluster_venues_sorted.iloc[ind, 1:] = return_most_common_venues(chicago_grouped_cluster.iloc[ind, :], num_top_venues)

cluster_venues_sorted.head()


# Let's compare the numbers with the average case rate per cluster

# In[103]:


# join the two dataframes using cluster labels
chicago_grouped_cluster.set_index('ClusterLabel').join(chicago_merged[['ClusterLabel','meanCaseRateCluster']].set_index('ClusterLabel'),
                           how='right',rsuffix="_r").drop_duplicates()


# ## 5. Results

# Let's update the summary table for clusters

# | Cluster Label | Total Zip Codes | 1st Most Common Venue | 2nd Most Common Venue | 3rd Most Common Venue| 4th Most Common Venue| 5th Most Common Venue| Average Case Rate |
# | :------ | :-:| :----------: | :--------------:|:---------------: |:--------------:|:-:|----------------:|
# |0| 30 | Shop & Service (**29.94%**)| Professional & Other Places (**26.44%**)| Food **(16.74%)**|Outdoors & Recreation (**8.21%**)|Travel & Transport (**5.45%**)|3970.592593|
# |1| 21 | Shop & Service (**16.22%**)| Professional & Other Places (**17.78%**)| Food (**13.41%**)|Travel & Transport (**12.80%**)|Outdoors & Recreation (**10.15%**)|2788.690476|
# |2| 9  | Professional & Other Places (**50.04%**)| Shop & Service (**16.22%**)| Food (**11.53%**)|Nightlife Spot (**4.72%**)|Outdoors & Recreation (**4.27%**)|3717.877778|
# |3| 1  | Shop & Service (**60%**)| Food (**20%**)| Travel & Transport (**20%**)|||5218.000000|
