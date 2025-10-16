from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from zlib import crc32
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn import set_config


# This task is about predicting housing price with 10 parameters
# Firstly, we need to retrieve the data file, using the load_housing_data() method
# Then, we use head() to get all the headings of this file
# Afterwards, we use matplotlib to plot the graphs of each parameter's distribution


# Defining a method to load the data
def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")  # 1. Need to create the path to store the file first
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)  # 2. If the path doesn't exist, create one
        url = "https://github.com/ageron/data/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)  # 3. Get the file from the online link
    with tarfile.open(tarball_path) as housing_tarball:
        housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))  # Return the file


pd.set_option('display.max_columns', None)  # Set pandas to show all the columns
housing = load_housing_data()  # Get the file
print(housing.head())  # View the headings of this file


# extra code – the next 5 lines define the default font sizes
# plt.rc('font', size=14)
# plt.rc('axes', labelsize=14, titlesize=14)
# plt.rc('legend', fontsize=14)
# plt.rc('xtick', labelsize=10)
# plt.rc('ytick', labelsize=10)
#
# housing.hist(bins=50, figsize=(12, 8))
# plt.show()


# Set aside partial data as test data using random seed (method 1)
def shuffle_and_split_data(data, test_ratio):
    np.random.seed(42)  # Set the random seed, so that it will be a pseudo random process
    shuffled_indices = np.random.permutation(len(data))  # random.permutation(): randomly rearrange the data
    test_set_size = int(len(data) * test_ratio)
    # For instance, if the overall size is 1000, and test_ratio = 20%, then the test set's size should be 200
    test_indices = shuffled_indices[:test_set_size]  # Pick up the first n data as the test set
    train_indices = shuffled_indices[test_set_size:]  # Leave the rest data as the train set
    return data.iloc[train_indices], data.iloc[test_indices]  # Return the train set and test set as dataframes


# However, this method will break the next time you fetch an updated dataset.
# To have a stable train/test split even after updating the dataset, a common solution is:
# Using each instance’s identifier to decide whether it should go in the test set
# (assuming instances have unique and immutable identifiers).

# Set aside partial data as test data using identifier (method 2)
def is_id_in_test_set(identifier, test_ratio):
    return crc32(np.int64(identifier)) < test_ratio * 2 ** 32
    # Generate a hash crc32 value using the identifier
    # Note that the crc32 hash value is between 0 and 2^32
    # If this hash value is less than test ratio * 2^32, put it into the test set
    # e.g., if test_ratio = 0.2, then the hash value will be restricted between 0 and 0.2*2^32


def split_data_with_id_hash(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


# Add an index column
housing_with_id = housing.reset_index()  # adds an `index` column
# Split the data into two data sets: train and test
# train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "index")
# Use (longitude * 1000 + latitude) as the identifier
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
# With the generated identifier, split the data into two data sets: train and test
train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "id")

# Set aside partial data as test data using sklearn method (method 3, maybe the simplest)
# Split a test set (20% of the entire data set), and provide a given random seed, always generating the same indices
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# Categorize the income into 5 groups
# It is important to have sufficient instances in each stratum (each group), or it could be biased
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
# housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
# plt.xlabel("Income category")
# plt.ylabel("Number of districts")
# plt.show()


splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
# n_splits: 10 times of splits
# test_size: the test set takes 20% of total data
# random_state: the random seed
strat_splits = []
for train_index, test_index in splitter.split(housing, housing["income_cat"]):  # splitter has set the ratio as 0.2
    # Generate 10 sets of indexes, housing is the file, and housing["income_cat"] is the stratified result (5 groups)
    strat_train_set_n = housing.iloc[train_index]
    strat_test_set_n = housing.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])

strat_train_set, strat_test_set = strat_splits[0]
print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
#
# Reverting the data back to its original state
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.copy()

# A vivid graph showing the relationship between population and housing price for the first stratum
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True,
             s=housing["population"] / 100, label="population",
             c="median_house_value", cmap="jet", colorbar=True,
             legend=True, sharex=False, figsize=(10, 7))
plt.show()

# Calculate the correlation coefficient between each pair of attributes
# But first, we need to select the numeric type columns, since Strings could not be calculated
housing_numeric_preparation = housing.select_dtypes(include=['number'])  # Use this line to select numeric columns only
housing_numeric = housing_numeric_preparation.corr()
print(housing_numeric["median_house_value"].sort_values(ascending=False))

# attributes = ["median_house_value", "median_income", "total_rooms",
#               "housing_median_age"]
# housing.plot(kind="scatter", x="median_income", y="median_house_value",
#              alpha=0.1, grid=True)
# plt.show()

# Adding other attributes is also available by the following steps:
housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"] = housing["population"] / housing["households"]
housing_num2 = housing.select_dtypes(include=['number'])  # Use this line to select numeric columns only
housing_numeric = housing_num2.corr()
print(housing_numeric["median_house_value"].sort_values(ascending=False))  # 3 new attributes will be added

# Revert to a clean training set
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# Missing values: 3 methods to deal with missing values (like total_bedrooms in this case)
# housing.dropna(subset=["total_bedrooms"], inplace=True)  # Method 1, get rid of the rows with missing values
# housing.drop("total_bedrooms", axis=1)  # Method 2, get rid of the whole column
# median = housing["total_bedrooms"].median()
# housing["total_bedrooms"].fillna(median, inplace=True)  # Method 3, fill all the missing values with the median

# Generally speaking, method 3 is the least destructive method
# However, we could apply a simpler way, by using SimpleImputer
imputer = SimpleImputer(strategy="median")
housing_num3 = housing.select_dtypes(include=[np.number])
imputer.fit(housing_num3)
# Now we can use this imputer to replace the missing values with the median
# X = imputer.transform(housing_num3)
# Use pd.DataFrame to wrap the X into the dataset
# housing_tr = pd.DataFrame(X, columns=housing_num.columns,
#                           index=housing_num.index)

# Don't forget that we still have one String column in the dataset, which needs to be set to numeric type
housing_cat = housing[["ocean_proximity"]]
# ordinal_encoder = OrdinalEncoder()
# housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

# Another method which could be more suitable: OneHotEncoder
# List all the possible attribute values, e.g., near the seaside, <1h to the seaside, >1h to the seaside
# Then, create three columns for these three values
# First column: near the seaside = 1, NOT near the seaside (despite its exact value) = 0
# The second and third columns follow the same rule
# THIS METHOD IS SUITABLE FOR ARRAYS WITH LOTS OF ZEROS
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

# To scale the labels to fit the prediction better, we have two methods here:
# Method 1: Manually scaling
target_scaler = StandardScaler()  # from sklearn.preprocessing import OrdinalEncoder, StandardScaler
scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())  # Copied median house value

model = LinearRegression()  # Use linear regression
model.fit(housing[["median_income"]], scaled_labels)  # Try to find the relationship between income and house value
some_new_data = housing[["median_income"]].iloc[:5]  # pretend this is new data

# Use this model to predict "new data" (actually it's retrieved from the original dataset)
scaled_predictions = model.predict(some_new_data)
predictions_old = target_scaler.inverse_transform(scaled_predictions)

# Method 2: TransformedTargetRegressor, automatically scale the labels and train the regression model
model = TransformedTargetRegressor(LinearRegression(),
                                   transformer=StandardScaler())
model.fit(housing[["median_income"]], housing_labels)
predictions = model.predict(some_new_data)

# Since "population" is a positive feature, and it has long-tail, we could use log function to transform it
# REMEMBER THAT IT WOULD BE BETTER FOR THE MACHINE TO LEARN IF WE SET THE DATA TO A "BELL" SHAPE?
log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
log_pop = log_transformer.transform(housing[["population"]])

# Now let's build a pipeline to preprocess the numerical attributes
# The following pipeline code set all the empty units to the median
num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("standardize", StandardScaler()),
])
num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
set_config(display='diagram')
print(num_pipeline)

housing_num_prepared = num_pipeline.fit_transform(housing_num3)
print(housing_num_prepared[:2].round(2))

df_housing_num_prepared = pd.DataFrame(
    housing_num_prepared, columns=num_pipeline.get_feature_names_out(),
    index=housing_num3.index)
print(df_housing_num_prepared.head(2))  # extra code

