import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

class Dataframe_Manipulation:
    def __init__(self):
        self.dataframe = pd.read_csv(r'C:\Users\bohayes\AppData\Local\Programs\Python\Python38\Excel and Text\housing.csv')
    def Cat_Creation(self):
        # Creation of an Income Category to organize the median incomes into strata (bins) to sample from
        self.income_cat = self.dataframe['income_category'] = pd.cut(self.dataframe['median_income'],
                                      bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                      labels=[1, 2, 3, 4, 5])
        self.rooms_per_house_cat = self.dataframe['rooms_per_house'] = self.dataframe['total_rooms']/self.dataframe['households']
        self.bedrooms_per_room_cat = self.dataframe['bedrooms_per_room'] = self.dataframe['total_bedrooms']/self.dataframe['total_rooms']
        self.pop_per_house = self.dataframe['pop_per_house'] = self.dataframe['population'] / self.dataframe['households']
        return self.dataframe
    def Fill_NA(self):
        self.imputer = KNNImputer(n_neighbors=5, weights='uniform')
        self.dataframe['total_bedrooms'] = self.imputer.fit_transform(self.dataframe[['total_bedrooms']])
        self.dataframe['bedrooms_per_room'] = self.imputer.fit_transform(self.dataframe[['bedrooms_per_room']])
        return self.dataframe
    def Income_Cat_Split(self):
        self.inc_cat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for self.train_index, self.test_index in self.inc_cat_split.split(self.dataframe, self.dataframe['income_category']):
            self.strat_train_set = self.dataframe.loc[self.train_index].reset_index(drop=True)
            self.strat_test_set = self.dataframe.loc[self.test_index].reset_index(drop=True)
            # the proportion is the % of total instances and which strata they are assigned to
            self.proportions = self.strat_test_set['income_category'].value_counts() / len(self.strat_test_set)
            # Only pulling out training set!!!!!!!!!!!!!!!
            return self.strat_train_set
    def Remove_Cats(self):
        self.labels = self.strat_train_set['median_house_value'].copy()
        self.strat_train_set = self.strat_train_set.drop(['median_house_value'], axis=1)
        return self.labels
    def Encode_Transform(self):
        self.column_trans = make_column_transformer((OneHotEncoder(), ['ocean_proximity']), remainder='passthrough')
        self.training_set_encoded = self.column_trans.fit_transform(self.strat_train_set)
        return self.training_set_encoded
    def Standard_Scaler(self):
        self.scaler = StandardScaler()
        self.scale_training_set = self.scaler.fit(self.training_set_encoded)
        self.scaled_training_set = self.scaler.transform(self.training_set_encoded)
        return self.scaled_training_set
##    , self.strat_test_set
        
    
A = Dataframe_Manipulation()
B = A.Cat_Creation()
C = A.Fill_NA()
D = A.Income_Cat_Split()
E = A.Remove_Cats()
F = A.Encode_Transform()
Dataframe = A.Standard_Scaler()
##np.set_printoptions(threshold=sys.maxsize)
##print(Dataframe)

param_grid = [{'n_estimators' : np.random.randint([1, 30]), 'max_features' : np.random.randint([1, 30])}]
##              {'bootstrap' : [False], 'n_estimators' : [3, 12], 'max_features' : [2, 3, 4]}]

rand_reg = RandomForestRegressor({'bootstrap': True, 'ccp_alpha': 0.0, 'criterion': 'mse',
                                  'max_depth': None, 'max_leaf_nodes': None,
                                  'max_samples': None, 'min_impurity_decrease': 0.0,
                                  'min_impurity_split': None, 'min_samples_leaf': 1,
                                  'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0,
                                  'n_jobs': None, 'oob_score': False,
                                  'random_state': None, 'verbose': 0, 'warm_start': False})
grid_search = GridSearchCV(rand_reg, param_grid, cv=5, refit=True,
                            scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(Dataframe, E)
##print(grid_search.predict(Dataframe))
##print(grid_search.best_estimator_.get_params())

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)



##class Training_Models:
##    def __init__(self):
##        self.Dataframe = Dataframe
##        self.E = E
##    def Predictions(self):
##        self.predictor = self.Dataframe[:5]
##        self.predictor_labels = self.E[:5]
##        return self.predictor, self.predictor_labels
####    def GridSearch(self):
####        self.grd_srch = GridSearchCV()
####        self.grd_srch_fit = self.grd_srch.fit(self.predictor, self.predictor_labels)
##        
####    def Lin_Reg(self):
####        self.lin_reg = LinearRegression()
####        self.lin_reg_fit = self.lin_reg.fit(self.Dataframe, self.E)
####        return self.lin_reg.predict(self.Dataframe[:5])
####    def Tree_Reg(self):
####        self.tree_reg = DecisionTreeRegressor()
####        self.tree_reg.fit(self.Dataframe, self.E)
####        self.tree_predict = self.tree_reg.predict(self.Dataframe)
####        return self.tree_predict
##    def Rand_For_Reg(self):
##
##        self.param_grid = [{'n_estimators' : [3, 10, 30], 'max_features' : [2, 4, 6, 8]},
##                      {'bootstrap' : [False], 'n_estimators' : [3, 10], 'max_features' : [2, 3, 4]}]
##        self.rand_reg = RandomForestRegressor()
##        self.grid_search = GridSearchCV(self.rand_reg, self.param_grid, refit=True
####                                        scoring='neg_mean_squared_error', return_train_score=True
##                                        )
##        self.grid_search.fit(self.Dataframe, self.E)
####        self.rand_predict = self.grid_search.predict(self.Dataframe)
##                      
##        return print(self.grid_search.best_params_), print(self.grid_search.best_estimator_)
####    print("Prediction", self.rand_predict), 
####    def Mean_Squared_Error(self):
####        self.tree_mse = mean_squared_error(self.E, self.tree_predict)
####        self.tree_rmse = np.sqrt(self.tree_mse)
####        self.rand_mse = mean_squared_error(self.E, self.rand_predict)
####        self.rand_rmse = np.sqrt(self.rand_mse)
####        return self.tree_mse, self.rand_mse
####    def Cross_Validation(self):
######        self.lin_scores = cross_val_score(self.lin_reg, self.Dataframe, self.E, scoring="neg_mean_squared_error", cv=10)
######        self.lin_rmse_scores = np.sqrt(-self.lin_scores)
######        self.tree_scores = cross_val_score(self.tree_reg, self.Dataframe, self.E, scoring="neg_mean_squared_error", cv=10)
######        self.tree_rmse_scores = np.sqrt(-self.tree_scores)
######        self.rand_scores = cross_val_score(self.grid_search, self.Dataframe, self.E, scoring="neg_mean_squared_error", cv=10)
######        self.rand_rmse_scores = np.sqrt(-self.rand_scores)
######        return self.rand_rmse_scores
######    self.tree_rmse_scores,
######    self.lin_rmse_scores,
####    
####    def Display_Scores(self):
######        print("Tree_Scores", self.tree_rmse_scores)
######        print("Tree_Mean", self.tree_rmse_scores.mean())
######        print("Tree_Standard Deviation", self.tree_rmse_scores.std())
######        print("Lin_Scores", self.lin_rmse_scores)
######        print("Lin_Mean", self.lin_rmse_scores.mean())
######        print("Lin_STD", self.lin_rmse_scores.std())
####        print("Forest_Scores", self.rand_rmse_scores)
####        print("Forest_Mean", self.rand_rmse_scores.mean())
####        print("forest_STD", self.rand_rmse_scores.std())
####              
####
##A = Training_Models()
##B = A.Predictions()
####C = A.Lin_Reg()
####D = A.Tree_Reg()
##E = A.Rand_For_Reg()
######F = A.Mean_Squared_Error()
####G = A.Cross_Validation()
####H = A.Display_Scores()
####print(H)

"""
NOTES

Above:
After some intense reseach and a lot of changes I am making this update so that I may understand more about
what I am doing here.

From how I understand it: any changes made to the original dataframe constitutes as a Transformation
I have combine all of the changes into a single class and named it DataFrame_Manipulation

There is still more edits to come uncluding feature scaling and others before I am ready to utilize a
Algorithm at this time.

First I define the dataframe within the class,

I then move on to category creation, where I collect and add Data from other columns together to create new
data that will approve the accuracy of my solution. Here I am also creating a stratified
income category from which all of the median_income values will be parsed into bins so that later,
I may be able to get a more accurate prediction, based off of which income category a feature falls into. 

I then remove the traget attribute to ensure that I am not making any predictions on it yet.

I then fill the null cells in the dataframe using KNNImputer, a class from sklearn that I found gave the most
Accurate results as it chooses values for the null values based off of its nearest neighbors

I then separate the training and test sets with Income_Cat_Split. Here I am separating arbitrarily, but
due to the nature of stratification, I am recieveing the best suited values back in my training set.
Read more about stratification below.

I then take the training set and with the best written code on this page, I am able to apply encoding
to ONLY the 'ocean_proximity' value as I am able to use remainder='passthrough' to allow all other values to
pass through with ease, without being encoded. This is the best solution I have found so far for the
shortest lines of code possible. Any other attempt to parse the 'ocean_proximity' attribute into
each cell of the dataframe failed, but the good thing here is that Encode_Transform also takes each
row and converts it into an array, which is much easier to do work on

I have tested it and I am seing only the correct values returned and it seems to be successful.

I will now move on to feature scaling, which may mess with the layout and order of my class,

But I am inching closer and closer to feeding the training set to a ML system.

For 'ocean_proximity' the different categories available are:
['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
This means that for a row that has Ocean_proximity listed as '<1H OCEAN' the encoded value appears in the
array as:

1.0 0.0 0.0 0.0 0.0

The first value is on and the rest are off.

similarily if 'INLAND' was listed on the row the value in the array would appear as:

0.0 1.0 0.0 0.0 0.0

fit_transform allows you to both fit the datat and transform it simultaneously,
this also saves time and processing power. 


The only difference between the dataframe and the newly created "training set array" is that the ocean_proximity
encoded value appears first in the "row" rather than longitute.

to verify that this is correct, I viewed the return of self.strat_train_set.iloc[0] as :

longitude               -121.89
latitude                  37.29
housing_median_age           38
total_rooms                1568
total_bedrooms              351
population                  710
households                  339
median_income            2.7042
ocean_proximity       <1H OCEAN
income_category               2
rooms_per_house         4.62537
bedrooms_per_room      0.223852
pop_per_house            2.0944

and I was able to pull out the first value of the new training set array as:

1.0 0.0 0.0 0.0 0.0 
-121.89 
37.29 
38.0 
1568.0 
351.0 
710.0 
339.0 
2.7042
2 
4.625368731563422 
0.22385204081632654 
2.094395280235988

They are identical, the array lists longer float values and the only difference is the placement of
the encoded 'ocean_proximity' value in the order, this should not make a difference. 

-----------------------------------------------------------------------------------------------------

def train_split_test is for random sampling of which every time the function is run it
chooses a new sample at random from the dataset

You can view the proportions of the random sample by running the following:

proportions1 = dataframe['income_category'].value_counts() / len(dataframe)
print(proportions1)

Notice that the restuls of the above differ from the reults of the stratified sample proportions. 

def Income_Cat_Split is for stratified sampling, meaning that a new column is created called
Income_Category where the data from the Median_Income column is separated into bins or strata
strata is the grouping together of multiple instances within a certian range. Beccause of this the function
chooses a specific entry that best represents the data, so it will not be random everytime the function
is called. this function returns three values, but the test_strata sample is the most important.
This function preforms a classification of the incomes to better represent the group being sampled from.

You can view the classification by running the following:

orig_data['median_income'].hist()
dataframe['income_category'].hist()
plt.show()

here in orange you will see that the data is collected into strata and the original data is much more dispersed
and random in the dataframe. You can get a more accurate representation of the median incomes by collecting
the instances into categories and assigning a new column to each instance (row) that determines which instance
falls into which category

The return of the function def Income_Cat_Split is NOT random upon each call becuase the random_state=42
parameter makes sure that the data is shuffled the exact same way every time. If this parameter was
removed it would mean that each new call would yeild a new shuffle.

When random_state is set to 42 it ensures that the deck is shuffled the exact same way everytime,
On the other hand if you use random_state=some_number, then you can guarantee that the output of
Run 1 will be equal to the output of Run 2, i.e. your split will be always the same.
It doesn't matter what the actual random_state number is 42, 0, 21, ... The important thing
is that everytime you use 42, you will always get the same output the first time you make the
split. This is useful if you want reproducible results, for example in the documentation, so
that everybody can consistently see the same numbers when they run the examples. In practice
I would say, you should set the random_state to some fixed number while you test stuff, but then
remove it in production if you really need a random (and not a fixed) split.



"""
###############OLD CODE

####Code used for creation of train_split_test function
######pandas/sklrean version using from sklearn.utils import shuffle
##Test_total = shuffle(dataframe)
##Test_Total1 = Test_total.iloc[:4126].reset_index(drop=True)
##print(Test_Total1)

####numpy/pandas version of train_split_test function
##df_sample = dataframe.iloc[np.random.permutation(len(dataframe))]
##test_sample = df_sample.iloc[:4126].reset_index(drop=True)
##print(test_sample)

############Using sklearn to randomize and split the dataset. It is redundant for datasets of many sizes. 
##test_set, train_set =train_test_split(dataframe, test_size=0.2, random_state=42)
##test_set_final = test_set.reset_index(drop=True)
##train_set_final  = train_set.reset_index(drop=True)
##print(test_set_final.head(), "Test set")
##print(train_set_final.head(), "Train Set")


######## Original code used for random sampling 
##def train_split_test(dataframe):
##    df_test_sample = dataframe.iloc[np.random.permutation(len(dataframe))]
##    test_sample = df_test_sample.iloc[:4126].reset_index(drop=True)
##    train_sample = df_test_sample.iloc[4127:].reset_index(drop=True)
##    return test_sample, train_sample
##
##test_set, train_set = train_split_test(dataframe)
##print(test_set.head(), "Test set")
##print(train_set.head(), "Train Set")
##
##dataframe.hist(bins=50, figsize=(20,15))    #.hist() shows the histogram when used with plt and matplotlib
##plt.show()

######## Code from book on Stratified Sampling
##inc_cat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
##for train_index, test_index in inc_cat_split.split(dataframe, dataframe['income_category']):
##    strat_train_set = dataframe.loc[train_index]
##    strat_test_set = dataframe.loc[test_index]
##
##print(strat_test_set['income_category'].value_counts() / len(strat_test_set))
##
##for set_ in (strat_train_set, strat_test_set):
##    set_.drop("income_category", axis=1, inplace=True)
##
##print(dataframe.info())
##print(set_.info())

############ Code from Train_test_Split, original code from myself setting up different ways to split the training
############ Test sets all within one class. 
##class Test_Split:
##    def __init__(self, dataframe):
##        self.dataframe = dataframe
##        # My method of separating the traing and test sets
##    def train_split_test(self, dataframe):
##        self.df_test_sample = self.dataframe.iloc[np.random.permutation(len(self.dataframe))]
##        self.test_sample = self.df_test_sample.iloc[:4126].reset_index(drop=True) # 20%
##        self.train_sample = self.df_test_sample.iloc[4127:].reset_index(drop=True)
##        self.proportions = self.dataframe['income_category'].value_counts() / len(self.dataframe)
##        return self.test_sample, self.train_sample, self.proportions
##        # The book's method for splitting the train/test set, create a stratified column out of the median_income
##        # Values and then separate it. 
##    def Income_Cat_Split(self, dataframe):
##        self.inc_cat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
##        for train_index, test_index in self.inc_cat_split.split(self.dataframe, self.dataframe['income_category']):
##            self.strat_train_set = self.dataframe.loc[train_index].reset_index(drop=True)
##            self.strat_test_set = self.dataframe.loc[test_index].reset_index(drop=True)
##            # the proportion is the % of total instances and which strata they are assigned to
##            self.proportions1 = self.strat_test_set['income_category'].value_counts() / len(self.strat_test_set)
##            return self.strat_train_set, self.strat_test_set, self.proportions1
##    def Drop_Income_Cat(self):
##        for self.set_ in (self.strat_test_set, self.strat_train_set):
##            self.set_.drop("income_category", axis=1, inplace=True)
##            return self.set_.reset_index(drop=True)
##        
##
####Initialize the class 
##Split = Test_Split(dataframe)
#### Run first function
##print(Split.train_split_test(dataframe), "MY METHOD")
#### Run second function
##print(Split.Income_Cat_Split(dataframe), "INCOME CAT SPLIT")
#### Run third function
##print(Split.Drop_Income_Cat(), "Drop Inceme Cat")

########### Plotting the Longitude and Latitude
##X = training_set['longitude']
##y = training_set['latitude']
##plt.scatter(X, y, alpha=0.1, s=training_set['population']/100, label="population", cmap=plt.get_cmap("jet"))
##plt.show()

########### Pearsons R and comparison
# getting the standard correlation coefficients for all of the columns in the training set
##class Pearsons_R:
##    def __init__(self, training_set):
##        self.training_set = training_set
##        # Cycle through all Features and find the Correlation Coef. for all other columns to compare.
##    def cycle(self, training_set):
##        self.all_corrs = self.training_set.corr()
##        self.median_house_value = self.all_corrs['median_house_value'].sort_values(ascending=False)
##        self.median_income = self.all_corrs['median_income'].sort_values(ascending=False)
##        self.total_rooms = self.all_corrs['total_rooms'].sort_values(ascending=False)
##        self.housing_median_age = self.all_corrs['housing_median_age'].sort_values(ascending=False)
##        self.total_bedrooms = self.all_corrs['total_bedrooms'].sort_values(ascending=False)
##        self.population = self.all_corrs['population'].sort_values(ascending=False)
##        self.longitude = self.all_corrs['longitude'].sort_values(ascending=False)
##        self.latitude = self.all_corrs['latitude'].sort_values(ascending=False)
##        return self.median_house_value, self.median_income, self.total_rooms, self.housing_median_age, self.total_bedrooms, self.population, self.longitude, self.latitude
##        
##        
##R = Pearsons_R(training_set)
##print(R.cycle(training_set))

############Old cfreation of columns after train_test_split function, removed and replaced at dataframe
##training_set["rooms_per_house"] = training_set["total_rooms"]/training_set["households"]
##training_set['bedrooms_per_room'] = training_set['total_bedrooms']/training_set['total_rooms']
##training_set['pop_per_house'] = training_set['population'] / training_set['households']
##print(training_set)
##def Column_Creation(training_set):
##    training_set["rooms_per_house"] = training_set["total_rooms"] / training_set["households"]
##    training_set['bedrooms_per_room'] = training_set['total_bedrooms'] / training_set['total_rooms']
##    training_set['pop_per_house'] = training_set['population'] / training_set['households']
##    return training_set["rooms_per_house"], training_set['bedrooms_per_room'], training_set['pop_per_house']
##Activate = Column_Creation(training_set)
##print(Activate)

##############Ploting median_house_value against bedrooms_per_room
##X = training_set["median_house_value"]
##Y = training_set["bedrooms_per_room"]
##plt.scatter(X, Y, alpha=0.1)
##plt.show()

##############Ploting all attributes against eachother
##Sample = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
##scatter_matrix(training_set[Sample], figsize=(12, 8))
##plt.plot()
##plt.show()

############## Creating categories (cats) and replacing missing values with medians
##dataframe['bedrooms_per_room'] = dataframe['total_bedrooms']/dataframe['total_rooms']
##dataframe['pop_per_house'] = dataframe['population'] / dataframe['households']
##
##
##dataframe1 = dataframe.drop(['median_house_value'], axis=1)
##median_total_bedrooms = dataframe1['total_bedrooms'].median()
##median_bedrooms_per_room = dataframe1['bedrooms_per_room'].median()
##dataframe1['total_bedrooms'].fillna(median_total_bedrooms, inplace=True)
##dataframe1['bedrooms_per_room'].fillna(median_bedrooms_per_room, inplace=True)
##        
##############Removing the original variable for opening the cxv
##dataframe1 = pd.read_csv(r'C:\Users\bohayes\AppData\Local\Programs\Python\Python38\Excel and Text\housing.csv')

##############Removing previous soilution for Fill_NA function where missing values are computed
##        self.median_total_bedrooms = self.dataframe['total_bedrooms'].median()
##        self.median_bedrooms_per_room = self.dataframe['bedrooms_per_room'].median()
##        self.total_bedrooms_fill = self.dataframe['total_bedrooms'].fillna(self.median_total_bedrooms, inplace=True)
##        self.bedrooms_per_room_fill = self.dataframe['bedrooms_per_room'].fillna(self.median_bedrooms_per_room, inplace=True)

##############Simple Imputer, replacing missing values with median values.
##from sklearn.impute import SimpleImputer
##        self.imputer = SimpleImputer(strategy='median')
##        self.dataframe['total_bedrooms'] = self.imputer.fit_transform(self.dataframe[['total_bedrooms']])
##        self.dataframe['bedrooms_per_room'] = self.imputer.fit_transform(self.dataframe[['bedrooms_per_room']])

##############My preferred method for imputation, however KNNImputation was suggested and I am now using it in Fill_NA
##        self.dataframe['total_bedrooms'].fillna(method='ffill', inplace=True)
##        self.dataframe['bedrooms_per_room'].fillna(method='ffill', inplace=True)

##############Return only null data and print
##null_data = training_set[training_set.isnull().any(axis=1)]
##print(null_data)

############## Code from book that reiterates the creation of the new columns in a different format
##rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
##class CombindAttributesAdder(BaseEstimator, TransformerMixin):
##    def __init__(self, add_bedrooms_per_room = True):
##        self.add_bedrooms_per_room = add_bedrooms_per_room
##    def fit(self, X, y=None):
##        return self
##    def transform(self, X, y=None):
##        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
##        population_per_household = X[:, population_ix] / X[:, households_ix]
##        if self.add_bedrooms_per_room:
##            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
##            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
##        else:
##            return np.c_[X, rooms_per_household, population_per_household]
##
##attr_adder = CombindAttributesAdder(add_bedrooms_per_room=False)
##housing_extra_attribs = attr_adder.transform(training_set.values)
####np.set_printoptions(threshold=sys.maxsize)
##print(housing_extra_attribs)

##############Took out this class and added the operations to the main class as
## all actions preformed on the dataframe are considered transformations. 
###### Separating the function from the class and the test set from the training set and plotting
##class Training_Set_Manipulation:
##    def __init__(self, Dataframe):
##        self.Dataframe = Dataframe
##    def Income_Cat_Split(self):
##        self.inc_cat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
##        for self.train_index, self.test_index in self.inc_cat_split.split(self.Dataframe, self.Dataframe['income_category']):
##            self.strat_train_set = self.Dataframe.loc[self.train_index].reset_index(drop=True)
##            self.strat_test_set = self.Dataframe.loc[self.test_index].reset_index(drop=True)
##            # the proportion is the % of total instances and which strata they are assigned to
##            self.proportions1 = self.strat_test_set['income_category'].value_counts() / len(self.strat_test_set)
##            # Only pulling out training set!!!!!!!!!!!!!!!
##            return self.strat_train_set
##    def Encoding(self):
##        self.housing_cat = self.strat_train_set[["ocean_proximity"]]
##        self.cat_encoder = OneHotEncoder()
##        self.housing_cat_1hot = self.cat_encoder.fit_transform(self.housing_cat)
##        self.ocean_proximity = self.housing_cat_1hot.toarray()
##        # Not returning these values and not adding self.cat_encoder to dataframe
##        # Could not disperse array over dataframe accurately
##        return self.ocean_proximity, self.cat_encoder.categories_
##
##
##A = Training_Set_Manipulation(Dataframe)
##B = A.Income_Cat_Split()
##C = A.Encoding()
##training_set = B
####pd.set_option("display.max_rows", None)
##print(training_set.iloc[0])
##

############## Code for column transformation, endocing one attributer and passing through the rest
##column_trans = make_column_transformer((OneHotEncoder(), ['ocean_proximity']), remainder='passthrough')
##print(column_trans.fit_transform(Dataframe))
##print(Dataframe)

############## Understanging Encoding
##ohe = OneHotEncoder(sparse=False)
##print(ohe.fit_transform(Dataframe[['ocean_proximity']]))
##print(ohe.categories_)
