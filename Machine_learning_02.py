from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from zlib import crc32
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer, make_column_selector
from sklearn.preprocessing import FunctionTransformer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn import set_config
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans


# This task is about predicting housing price with 10 parameters
# Firstly, we need to retrieve the data file, using the load_housing_data() method
# Then, we use head() to get all the headings of this file
# Afterwards, we use matplotlib to plot the graphs of each parameter's distribution


# Defining a method to load the data
def load_housing_data():
    dataset_path = Path("datasets/housing.tgz")  # 1. Need to create the path to store the file first
    if not dataset_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)  # 2. If the path doesn't exist, create one
        url = "https://github.com/ageron/data/housing.tgz"
        urllib.request.urlretrieve(url, dataset_path)  # 3. Get the file from the online link
    with tarfile.open(dataset_path) as housing_price:
        housing_price.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))  # Return the file


pd.set_option('display.max_columns', None)  # Set pandas to show all the columns
housing = load_housing_data()  # Get the file
print(housing.head())  # View the headings of this file

# If we would like to learn more about the dataset, we could use info() method:
print(housing.info())

# If we would like to know the distribution of a certain attribute, we could use value_counts() method:
print(housing["ocean_proximity"].value_counts())
print(housing.describe())

# extra code – the next 5 lines define the default font sizes
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

housing.hist(bins=50, figsize=(12, 8))
plt.show()
#
#
# Set aside partial data as test data using random seed (method 1)
def shuffle_and_split_data(data, test_ratio):
    np.random.seed(42)  # Set the random seed, so that it will be a pseudo random process
    shuffled_indices = np.random.permutation(len(data))  # random.permutation(): randomly rearrange the data
    test_set_size = int(len(data) * test_ratio)
    # For instance, if the overall size is 1000, and test_ratio = 20%, then the test set's size should be 200
    test_indices = shuffled_indices[:test_set_size]  # Pick up the first n data as the test set
    train_indices = shuffled_indices[test_set_size:]  # Leave the rest data as the train set
    return data.iloc[train_indices], data.iloc[test_indices]  # Return the train set and test set as dataframes
#

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
train_set_1, test_set_1 = split_data_with_id_hash(housing_with_id, 0.2, "index")

# Now, the whole dataset has been split into a train set and a test set

# # Use (longitude * 1000 + latitude) as the identifier
# housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
# # With the generated identifier, split the data into two data sets: train and test
# train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "id")
#
# # Set aside partial data as test data using sklearn method (method 3, maybe the simplest)
# # Split a test set (20% of the entire data set), and provide a given random seed, always generating the same indices
# train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
#
# # Categorize the income into 5 groups
# # It is important to have sufficient instances in each stratum (each group), or it could be biased
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
plt.xlabel("Income category")
plt.ylabel("Number of districts")
plt.show()
#
#
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
#
strat_train_set, strat_test_set = strat_splits[0]
# print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
# #
# # Reverting the data back to its original state
# for set_ in (strat_train_set, strat_test_set):
#     set_.drop("income_cat", axis=1, inplace=True)
#
housing = strat_train_set.copy()
#
# # A vivid graph showing the relationship between population and housing price for the first stratum
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
#
attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.show()
#
housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1, grid=True)
plt.show()
#
# Adding other attributes is also available by the following steps:
housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"] = housing["population"] / housing["households"]
housing_num2 = housing.select_dtypes(include=['number'])  # Use this line to select numeric columns only
housing_numeric = housing_num2.corr()
print(housing_numeric["median_house_value"].sort_values(ascending=False))  # 3 new attributes will be added
#
# Revert to a clean training set
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
#
# # Missing values: 3 methods to deal with missing values (like total_bedrooms in this case)
# # housing.dropna(subset=["total_bedrooms"], inplace=True)  # Method 1, get rid of the rows with missing values
# # housing.drop("total_bedrooms", axis=1)  # Method 2, get rid of the whole column
# # median = housing["total_bedrooms"].median()
# # housing["total_bedrooms"].fillna(median, inplace=True)  # Method 3, fill all the missing values with the median
#
# Generally speaking, method 3 is the least destructive method
# However, we could apply a simpler way, by using SimpleImputer
imputer = SimpleImputer(strategy="median")
housing_num_imputer = housing.select_dtypes(include=[np.number])
imputer.fit(housing_num_imputer)
# Now we can use this imputer to replace the missing values with the median
X = imputer.transform(housing_num_imputer)
# Use pd.DataFrame to wrap the X into the dataset
housing_tr = pd.DataFrame(X, columns=housing_num_imputer.columns,
                          index=housing_num_imputer.index)

# # Don't forget that we still have one String column in the dataset, which needs to be set to numeric type
housing_cat = housing[["ocean_proximity"]]
print(housing_cat.head(8))
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
print(housing_cat_encoded[:8])
#
# Another method which could be more suitable: OneHotEncoder
# List all the possible attribute values, e.g., near the seaside, <1h to the seaside, >1h to the seaside
# Then, create three columns for these three values
# First column: near the seaside = 1, NOT near the seaside (despite its exact value) = 0
# The second and third columns follow the same rule
# THIS METHOD IS SUITABLE FOR ARRAYS WITH LOTS OF ZEROS
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

# We also need to scale the attribute values to improve the performance of our model
# Method 1: Normalization
min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num_imputer)
# Method 2:

std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num_imputer)

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
#
# Method 2: TransformedTargetRegressor, automatically scale the labels and train the regression model
model = TransformedTargetRegressor(LinearRegression(),
                                   transformer=StandardScaler())
model.fit(housing[["median_income"]], housing_labels)
predictions = model.predict(some_new_data)
#
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

# Of course, you don't have to name the estimators. Instead, you could use make_pipeline() method:
# num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
# Let’s call the pipeline’s fit_transform() method and look at the output’s first two rows,
# rounded to two decimal places:

housing_num_prepared = num_pipeline.fit_transform(housing_num_imputer)
print(housing_num_prepared[:2].round(2))

df_housing_num_prepared = pd.DataFrame(
    housing_num_prepared, columns=num_pipeline.get_feature_names_out(),
    index=housing_num_imputer.index)

print(df_housing_num_prepared[:2].round(2))

num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms",
               "total_bedrooms", "population", "households", "median_income"]
cat_attribs = ["ocean_proximity"]

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"))

preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])

# If we don't want to specify each column's name, we could use make_column_transformer()
from sklearn.compose import make_column_selector, make_column_transformer

full_preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline, make_column_selector(dtype_include=object)),
)

housing_prepared = preprocessing.fit_transform(housing)
feature_names = preprocessing.get_feature_names_out()

housing_prepared_df = pd.DataFrame(
    housing_prepared,
    columns=feature_names,
    index=housing.index
)

print(housing_prepared_df[:2].round(2))


# To sum up, let's build a pipeline to process the original dataset end-to-end

# The method that calculates the ratio of two columns
def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

# The method that returns the ratio column's name (here we just use "ratio")
def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]  # feature names out

# The method that creates a pipeline for calculating the ratio
def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),  # Use median value to impute the empty values
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),  # Calculate the ratio
        StandardScaler())  # Standardize the data


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]


# Make the data to its logarithm
log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler())

# Create 10 clusters, use this to determine how close is each sample with the clusters
# We use this to replace latitude and longitude
cluster_similarity = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)

# Deal with other numeric data, just use the median value to impute the empty values
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                     StandardScaler())

preprocessing_end_to_end = ColumnTransformer([
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),  # Need bedroom ratio
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),  # Need room ratio
        ("people_per_house", ratio_pipeline(), ["population", "households"]),  # Need population ratio
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                               "households", "median_income"]),  # Make these 5 attributes more evenly distributed
        ("geo", cluster_similarity, ["latitude", "longitude"]),  # Find the similarity of each sample and 10 clusters
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
        # If the type is object, use categorical transformer
        ],
        remainder=default_num_pipeline)  # one column remaining: housing_median_age, just use default method to process

# Now this is an NumPy array with 24 features:
housing_prepared_end_to_end = preprocessing_end_to_end.fit_transform(housing)

# # A QUICK SUMMARY OF THE CURRENT PROGRESS:
# # 1. We have got the data and explored it
# # 2. We have sampled the training set and the test set
# # 3. We have generated a preprocessing pipeline to automatically clean up and prepare your data for ML
#
# # Afterwards, we need to train the model, and use it to predict
# # The first attempt is to create a linear regression
lin_reg = make_pipeline(preprocessing_end_to_end, LinearRegression())
lin_reg.fit(housing, housing_labels)
housing_predictions = lin_reg.predict(housing)
print(housing_predictions[:5].round(-2))
print(housing_labels.iloc[:5].values)

lin_rmse = np.sqrt(mean_squared_error(housing_labels, housing_predictions))
print("The linear regression RMSE: ", lin_rmse)
# Output: 68300.88727399787
# This result definitely doesn't fit our requirement in accuracy

from sklearn.tree import DecisionTreeRegressor

tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(housing, housing_labels)

housing_predictions = tree_reg.predict(housing)
tree_rmse = np.sqrt(mean_squared_error(housing_labels, housing_predictions))
print("The decision tree regression RMSE: ", tree_rmse)

from sklearn.model_selection import cross_val_score

tree_rmses = -cross_val_score(tree_reg, housing, housing_labels,
                              scoring="neg_root_mean_squared_error", cv=10)
print(pd.Series(tree_rmses).describe())

# The next part of RandomForestRegressor code consumes lots of time, thus I temporarily comment it
# from sklearn.ensemble import RandomForestRegressor
# forest_reg = make_pipeline(preprocessing,
#                            RandomForestRegressor(random_state=42))
# forest_rmses = -cross_val_score(forest_reg, housing, housing_labels,
#                                 scoring="neg_root_mean_squared_error", cv=10)
# print(pd.Series(forest_rmses).describe())

from sklearn.model_selection import GridSearchCV

full_pipeline = Pipeline([
    ("preprocessing", preprocessing_end_to_end),
    ("random_forest", RandomForestRegressor(random_state=42)),
])
param_grid = [
    {'preprocessing__geo__n_clusters': [5, 8, 10],
     'random_forest__max_features': [4, 6, 8]},
    {'preprocessing__geo__n_clusters': [10, 15],
     'random_forest__max_features': [6, 8, 10]},
]
grid_search = GridSearchCV(full_pipeline, param_grid, cv=3,
                           scoring='neg_root_mean_squared_error')
grid_search.fit(housing, housing_labels)

print("The best parameter combination suggested by grid search is: ", grid_search.best_params_)

