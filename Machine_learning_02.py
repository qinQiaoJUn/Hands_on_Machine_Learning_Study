from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


# Defining a method to load the data
def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")  # 1. Need to create the path to store the file first
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)  # 2. If the path doesn't exist, create one
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)  # 3. Get the file from the online link
    with tarfile.open(tarball_path) as housing_tarball:
        housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))  # Return the file


pd.set_option('display.max_columns', None)  # Set pandas to show all the columns

housing = load_housing_data()  # Get the file
print(housing.head())  # View the headings of this file

# extra code – the next 5 lines define the default font sizes
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

housing.hist(bins=50, figsize=(12, 8))
plt.show()


# Set aside partial data as test data
def shuffle_and_split_data(data, test_ratio):
    shuffled_indices = np.random.seed(42).permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# Split a test set (20% of the entire data set), and provide a given random seed
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
# Categorize the income into 5 groups
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
plt.xlabel("Income category")
plt.ylabel("Number of districts")
plt.show()
