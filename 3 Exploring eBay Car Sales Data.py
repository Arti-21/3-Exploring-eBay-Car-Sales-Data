#!/usr/bin/env python
# coding: utf-8

# # Exploring eBay Car Sales Data
# 
# In this guided project, we'll work with a dataset of used cars from eBay Kleinanzeigen, a [classifieds](https://en.wikipedia.org/wiki/Classified_advertising)  section of the German eBay website.
# 
# The dataset was originally [scraped](https://en.wikipedia.org/wiki/Web_scraping) and uploaded to [Kaggle](https://www.kaggle.com/orgesleka/used-cars-database/data). The version of the dataset we are working with is a sample of 50,000 data points that was prepared by Dataquest including simulating a less-cleaned version of the data.

# The data dictionary is as follows:
# 
#  * `dateCrawled` - When this ad was first crawled. All field-values are taken from this date.
#  * `name` - Name of the car.
#  * `seller` - Whether the seller is private or a dealer.
#  * `offerType` - The type of listing
#  * `price` - The price on the ad to sell the car.
#  * `abtest` - Whether the listing is included in an A/B test.
#  * `vehicleType` - The vehicle Type.
#  * `yearOfRegistration` - The year in which which year the car was first registered.
#  * `gearbox` - The transmission type.
#  * `powerPS` - The power of the car in PS.
#  * `model` - The car model name.
#  * `kilometer` - How many kilometers the car has driven.
#  * `monthOfRegistration` - The month in which which year the car was first registered.
#  * `fuelType` - What type of fuel the car uses.
#  * `brand` - The brand of the car.
#  * `notRepairedDamage` - If the car has a damage which is not yet repaired.
#  * `dateCreated` - The date on which the eBay listing was created.
#  * `nrOfPictures` - The number of pictures in the ad.
#  * `postalCode` - The postal code for the location of the vehicle.
#  * `lastSeenOnline` - When the crawler saw this ad last online.
# 
# The aim of this project is to clean the data and analyze the included used car listings.

# ### Read the data
# 
# We'll import the NumPy and Pandas libraries and then read the CSV file into Pandas.

# In[2]:


import pandas as pd
import numpy as np


# First, we'll try to read the file without specifying any encoding (which will default to **UTF-8** which is the most common encoding) 

# In[3]:


## autos = pd.read_csv('autos.csv') Gives the UnicodeDecodeError


# Since our file has an unknown encoding, we try other common encodings:
# 
# * **Latin-1 (also known as ISO-8859-1)**
# * **Windows-1251**
# 
# until we are able to read the file without error.

# In[4]:


#reads file without error
autos = pd.read_csv('autos.csv', encoding='Latin-1') 


# In[5]:


#jupyter notebook renders the first few and last few values of any pandas object
autos


# In[6]:


autos.info() #prints information about the autos dataframe 
autos.head() #first few rows of autos


# We observe that:
# * The dataset contains 20 columns, most of which are stored as strings. 
# * Some columns have null values, but none have more than ~20% null values. 
# * There are some columns that contain dates stored as strings.
# * The column names use [camelcase](https://en.wikipedia.org/wiki/Camel_case) instead of Python's preferred [snakecase](https://en.wikipedia.org/wiki/Snake_case), which means we can't just replace spaces with underscores.

# ### Data Cleaning 
# 
# We'll start by cleaning the column names to make the data easier to work with.

# In[7]:


autos.columns #columns attribute prints an array of existing column names


# We'll make a few changes here:
# 
# * Convert the column names from camelcase to snakecase.
# * Reword some column names based on the data dictionary to make them more descriptive.

# In[8]:


autos.columns = ['date_crawled', 'name', 'seller', 'offer_type', 'price', 'ab_test',
       'vehicle_type', 'registration_year', 'gearbox', 'power_ps', 'model',
       'odometer', 'registration_month', 'fuel_type', 'brand',
       'unrepaired_damage', 'ad_created', 'num_photos', 'postal_code',
       'last_seen']


# In[9]:


autos.head() #look at the current state of the autos dataframe


# ### Initial Data Exploration and Cleaning 
# 
# We'll start by exploring the data to find obvious areas where we can clean the data.

# In[10]:


# Using `DataFrame.describe()` method to look at descriptive statistics for all columns with `include='all'` to get both categorical and numeric columns
autos.describe(include="all")


# Our initial observations:
# 
# * There are a number of text columns where all (or nearly all) of the values are the same:
#   * `seller`
#   * `offer_type`
# * The `num_photos` column looks odd, we'll need to investigate this further.
# 
# 

# In[11]:


autos["num_photos"].value_counts() #Return a Series containing counts of unique values in descending order 


# It looks like the `num_photos` column has 0 for every column. We'll drop this column.
# 
# Since, columns that have mostly one value are candidates to be dropped.
# We drop `seller` and `offer_type` columns too. 

# In[12]:


autos = autos.drop(["num_photos", "seller", "offer_type"], axis=1) #Removes rows or columns by specifying label names and corresponding axis(0 or 'index' for row, 1 or ' columns' for column )


# There are two columns, `price` and `odometer`, which are numeric values with extra characters being stored as text. We'll clean and convert these.

# In[13]:


#Removing non numeric characters from `price` column and conerting it to numeric dtype
autos["price"] = (autos["price"]
                          .str.replace("$","")
                          .str.replace(",","")
                          .astype(int)
                          )
autos["price"].head()


# In[14]:


#Removing non numeric characters from `odometer` column and conerting it to numeric dtype
autos["odometer"] = (autos["odometer"]
                             .str.replace("km","")
                             .str.replace(",","")
                             .astype(int)
                             )
#Renaming the column 
# axis=1 as column is to be renamed
# Inplace = True then dataframe copy is ignored.
autos.rename({"odometer": "odometer_km"}, axis=1, inplace=True)
autos["odometer_km"].head()


# ### Exploring odometer and price column
# 
# We'll analyze the columns using minimum and maximum values and look for any values that look unrealistically high or low (outliers) that we might want to remove.

# In[15]:


autos["odometer_km"].value_counts()


# We can see that the values in `odometer_km` field are rounded, which might indicate that sellers had to choose from pre-set options for this field. Also, there are more high mileage than low mileage vehicles.

# In[16]:


print(autos["price"].unique().shape) # to see how many unique values are there
print(autos["price"].describe()) #to view min/max/median/mean etc.
# using series.value_cunts() chained to .head() as there are lot of values
autos["price"].value_counts().head(20)


# Again, the prices in this column seem rounded, however given there are 2357 unique values in the column, that may just be people's tendency to round prices on the site.
# 
# There are 1,421 cars listed with $0 price - given that this is only 2% of the of the cars, we might consider removing these rows. The maximum price is one hundred million dollars, which seems a lot, let's look at the highest prices further.
# 

# In[17]:


# using Series.sort_index() with ascending= False to view the highest values with their counts 
autos["price"].value_counts().sort_index(ascending=False).head(20)


# In[18]:


# using Series.sort_index() with ascending= True to view the lowest values with their counts 

autos["price"].value_counts().sort_index(ascending=True).head(20)


# There are a number of listings with prices below \$30, including about 1,500 at \$0. There are also a small number of listings with very high values, including 14 at around or over $1 million.
# 
# Given that eBay is an auction site, there could legitimately be items where the opening bid is \$1. We will keep the \$1 items, but remove anything above \$350,000 since it seems that prices increase steadily to that number and then jump up to less realistic numbers.

# In[19]:


# for removing outliers, we can do df[(df["col"] > x ) & (df["col"] < y )]
# using df[df["col"].between(x,y)] for more readability
autos = autos[autos["price"].between(1,351000)]
autos["price"].describe()


# We'll now move on to the date columns and understand the date range the data covers.

# ### Exploring the date columns
# 
# There are 5 columns that should represent date values. Some of these columns were created by the crawler, some came from the website itself. We can differentiate by referring to the data dictionary:
# 
# * `date_crawled`: added by the crawler
# * `last_seen`: added by the crawler
# * `ad_created`: from the website
# * `registration_month`: from the website
# * `registration_year`: from the website
# 
# The non-registration dates i.e. `date_crawled`, `last_seen`, and `ad_created` columns are identified as string values by pandas.We'll need to convert the data of these columns into a numerical representation so we can understand it quantitatively.
# 
# The other two columns are represented as numeric values, so we can use methods like `Series.describe()` to understand the distribution without any extra data processing.
# 
# We'll explore each of these columns to learn more about the listings.

# In[20]:


# the three string columns represent full timestamp values, like so:
autos[['date_crawled','ad_created','last_seen']][0:5]


# The first 10 characters represent the day (e.g. 2016-03-12). To understand the date range, we can extract just the date values by using `Series.value_counts()` to generate a distribution, and then sort by the index.

# In[21]:


# To get relative frequencies instead of counts use nomalize=True
# To include missing values in the distribution use dropna=False
(autos["date_crawled"]
        .str[:10] #to select first 10 characters in each column
        .value_counts(normalize=True, dropna=False)  
        .sort_index()
        )


# Looks like the site was crawled daily over roughly a one month period in March and April 2016. The distribution of listings crawled on each day is roughly uniform.

# In[22]:


(autos["last_seen"]
        .str[:10]
        .value_counts(normalize=True, dropna=False)
        .sort_index()
        )


# The crawler recorded the date it last saw any listing, which allows us to determine on what day a listing was removed, presumably because the car was sold.
# 
# The last three days contain a disproportionate amount of 'last seen' values. Given that these are 6-10x the values from the previous days, it's unlikely that there was a massive spike in sales, and more likely that these values are to do with the crawling period ending and don't indicate car sales.
# 

# In[23]:


print(autos["ad_created"].str[:10].unique().shape)
(autos["ad_created"]
        .str[:10]
        .value_counts(normalize=True, dropna=False)
        .sort_index()
        )


# There is a large variety of ad created dates. Most fall within 1-2 months of the listing date, but a few are quite old, with the oldest at around 9 months.

# In[24]:


# Using Series.describe() to understand the distribution of registration_year
autos["registration_year"].describe()


# The year that the car was first registered will likely indicate the age of the car. Looking at this column, we note some odd values. The minimum value is `1000`, long before cars were invented and the maximum is `9999`, many years into the future.

# #### Dealing with Incorrect Registration Year Data
# 
# Since a car can't be first registered after the listing was seen, any vehicle with a registration year above 2016 is definitely inaccurate. Determining the earliest valid year is more difficult. Realistically, it could be somewhere in the first few decades of the 1900s.
# 
# One option is to remove the listings with these values. Let's determine what percentage of our data has invalid values (listings that fall outside the 1900 - 2016 intrval) in this column and see if it's safe to remove those rows entirely or we need more custom logic.
# 

# In[27]:


(~autos["registration_year"].between(1900,2016)).sum() / autos.shape[0]


# Given that this is less than 4% of our data, we will remove these rows.

# In[28]:


# Select rows that fall within a value range of `registration_year` column using `Series.between()`
autos = autos[autos["registration_year"].between(1900,2016)]
# Calculating distribution of remaining values
autos["registration_year"].value_counts(normalize=True).head(10)


# It appears that most of the vehicles were first registered in the past 20 years.

# When working with data on cars, it's natural to explore variations across different car brands. 
# We can use **aggregation** to understand the `brand` column.
# 
# 

# ### Exploring brand column

# In[31]:


# Obtaining the relative frequencies (normalize=True) of unique brands and excluding NA values by default.
autos["brand"].value_counts(normalize=True)


# 
# German manufacturers represent four out of the top five brands, almost 50% of the overall listings. Volkswagen is by far the most popular brand, with approximately double the cars for sale of the next two brands combined.
# 
# There are lots of brands that don't have a significant percentage of listings, so we will limit our analysis to brands representing more than 5% of total listings.

# In[34]:


brand_counts = autos["brand"].value_counts(normalize=True)
# Series.value_counts() produces a series with index labels which can be accessed by using Series.index attribute
# assigning the index of modified brand_counts to common_brands
common_brands = brand_counts[brand_counts > .05].index
print(common_brands)


# #### Exploring brand by price
# 
# Using `Series.mean()` to calculate mean price of cars of common brands obtained above.

# In[36]:


# Creating an empty dictionary to store our aggregate data
brand_mean_prices = {}

# Loop over the unique values
for brand in common_brands:
    brand_only = autos[autos["brand"] == brand] # Subset the dataframe by the unique values
    mean_price = brand_only["price"].mean() # Calculate the mean of price column
    brand_mean_prices[brand] = int(mean_price) # Assigning value of mean price to key of that brand

brand_mean_prices


# Of the top 5 brands, there is a distinct price gap:
# 
# * Audi, BMW and Mercedes Benz are more expensive
# * Ford and Opel are less expensive
# * Volkswagen is in between - this may explain its popularity, it may be a 'best of 'both worlds' option.
# 
# 

# For the top 6 brands, let's use aggregation to understand the average `mileage` for those cars and if there's any visible link with mean price.

# #### Exploring brand by mileage

# While our natural instinct may be to display both aggregated series objects and visually compare them, this has a few limitations:
# 
# * it's difficult to compare more than two aggregate series objects if we want to extend to more columns
# * we can't compare more than a few rows from each series object
# * we can only sort by the index (brand name) of both series objects so we can easily make visual comparisons
# 
# Instead, we can combine the data from both series objects into a single dataframe (with a shared index) and display the dataframe directly. 

# In[42]:


# the series constructor method uses brand_mean_prices dictionary
bmp_series = pd.Series(brand_mean_prices)
# The keys in the dictionary became the index in the series object
print(bmp_series)
# We can now create a single-column dataframe from this series object
# We need to use the columns parameter when calling the dataframe constructor (which accepts a array-like object) to specify the column name (or the column name will be set to 0 by default)
pd.DataFrame(bmp_series, columns=["mean_price"])


# In[43]:


# Creating an empty dictionary to store our aggregate data

brand_mean_mileage = {}

for brand in common_brands:
    brand_only = autos[autos["brand"] == brand]
    mean_mileage = brand_only["odometer_km"].mean()
    brand_mean_mileage[brand] = int(mean_mileage)

# Converting both dictionaries to series objects, using the series constructor    
mean_mileage = pd.Series(brand_mean_mileage).sort_values(ascending=False)
mean_prices = pd.Series(brand_mean_prices).sort_values(ascending=False)


# In[44]:


# Creating a dataframe from the first series (mean_mileage) object using the dataframe constructor
brand_info = pd.DataFrame(mean_mileage,columns=['mean_mileage'])
brand_info


# In[45]:


# Assigning the other series (mean_prices) as a new column in this dataframe.
brand_info["mean_price"] = mean_prices
brand_info


# ### Conclusion
# The range of car mileages does not vary as much as the prices do by brand, instead all falling within 10% for the top brands. There is a slight trend to the more expensive vehicles having higher mileage, with the less expensive vehicles having lower mileage.
