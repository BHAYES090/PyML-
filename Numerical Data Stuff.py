import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import FunctionTransformer
from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import make_blobs
from sklearn.preprocessing import Binarizer
from sklearn.cluster import KMeans


#Rescale the values of a numerical feature to be betweeen two values

#Use scikit-learn MinMaxScaler to rescale a feature array

# Create feature
feature = np.array([[-500.5],
                    [-100.1],
                    [0],
                    [100.1],
                    [900.9]])

# Create scalar
minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))

# Scale feature
scaled_feature = minmax_scale.fit_transform(feature)

#Show feature
print(scaled_feature, "Feature scaled to range of 0, 1")


##Rescaling is a common preprocessing task in machine learning.
##Many of the algorithms described later in this book will assume
##all features are on the same scale, typically 0 to 1 or –1 to 1.
##There are a number of rescaling techniques, but one of the simplest
##is called min-max scaling. Min-max scaling uses the minimum and
##maximum values of a feature to rescale values to within a range.
##In our example, we can see from the outputted array that the
##feature has been successfully rescaled to between 0 and 1:

##scikit-learn’s MinMaxScaler offers two options to rescale a
##feature. One option is to use fit to calculate the minimum
##and maximum values of the feature, then use transform to rescale
##the feature. The second option is to use fit_transform to do both
##operations at once. There is no mathematical difference between the
##two options, but there is sometimes a practical benefit to keeping
##the operations separate because it allows us to
##apply the same transformation to different sets of the data.


############################################################################


#Preform a transform on a feature to have a mean of 0 and a standard deviation of 1

#scikit-learn’s StandardScaler performs both transformations:

# Create feature
x = np.array([[-1000.1],
              [-200.2],
              [500.5],
              [600.6],
              [9000.9]])

# Create scaler
scaler = preprocessing.StandardScaler()

# Transform the feature
standardized = scaler.fit_transform(x)

# Show feature
print(standardized)

##A common alternative to min-max scaling is rescaling of features
##to be approximately standard normally distributed. To achieve this,
##we use standardization to transform the data such that it has a mean,
##x̄, of 0 and a standard deviation, σ, of 1.

##The transformed feature represents the number of standard deviations
##the original value is away from the feature’s mean value
##(also called a z-score in statistics).

##Standardization is a common go-to scaling method for machine
##learning preprocessing and in my experience is used more than
##min-max scaling. However, it depends on the learning algorithm.
##For example, principal component analysis often works better using
##standardization, while min-max scaling is often recommended for neural
##networks (both algorithms are discussed later in this book).

##As a general
##rule, I’d recommend defaulting to standardization unless you have a
##specific reason to use an alternative.

##We can see the effect of standardization by looking at the mean
##and standard deviation of our solution’s output:

# Print mean and standard deviation
print("Mean:", round(standardized.mean()))
print("Standard deviation:", standardized.std())

##If our data has significant outliers, it can negatively
##impact our standardization by affecting the feature’s mean
##and variance. In this scenario, it is often helpful to instead
##rescale the feature using the median and quartile range.
##In scikit-learn, we do this using the RobustScaler method:

# Create scaler
robust_scaler = preprocessing.RobustScaler()

# Transform feature
print(robust_scaler.fit_transform(x), "rescale data using robust_scaler")


############################################################################


# rescale feature values of observations to have a unit norm (total lenght of 1.0)
# the following relies on from sklearn.preprocessing import Normalizer

# Use normalizer with a norm argument

# Create feature matrix
features = np.array([[0.5, 0.5],
                     [1.1, 3.4],
                     [1.5, 20.2],
                     [1.63, 34.4],
                     [10.9, 3.3]])

# Create normalizer
normalizer = Normalizer(norm="l2")

# Transform feature matrix
print(normalizer.transform(features), "use norm to tansform features")

##Many rescaling methods (e.g., min-max scaling and standardization)
##operate on features; however, we can also rescale across individual
##observations. Normalizer rescales the values on individual observations
##to have unit norm (the sum of their lengths is 1). This type of rescaling
##is often used when we have many equivalent features (e.g., text classification
##when every word or n-word group is a feature).

##Normalizer provides three norm options with Euclidean norm (often called L2)
##being the default argument.

# Transform feature matrix
features_l2_norm = Normalizer(norm="l2").transform(features)

# Show feature matrix
print(features_l2_norm, "Euclidean norm")

#Alternatively, we can specify Manhattan norm (L1):

# Transform feature matrix
features_l1_norm = Normalizer(norm="l1").transform(features)

# Show feature matrix
print(features_l1_norm, "Manhattan norm")

##Practically, notice that norm='l1' rescales an observation’s values
##so they sum to 1, which can sometimes be a desirable quality:

# Print sum
print("Sum of the first observation\'s values:",
   features_l1_norm[0, 0] + features_l1_norm[0, 1])

##Sum of the first observation's values: 1.0


############################################################################


# Create polynomial and interaction features

# this section relies on from sklearn.preprocessing import PolynomialFeatures

##Even though some choose to create polynomial and interaction features manually,
##scikit-learn offers a built-in method:

# Create feature matrix
features = np.array([[2, 3],
                     [2, 3],
                     [2, 3]])

# Create PolynomialFeatures object
polynomial_interaction = PolynomialFeatures(degree=2, include_bias=False)

# Create polynomial features
print(polynomial_interaction.fit_transform(features), "Polynomial raised to the 2nd degree")

##The degree parameter determines the maximum degree of the polynomial.
##For example, degree=2 will create new features raised to the second power.

##while degree=3 will create new features raised to the second and third power.

##Furthermore, by default PolynomialFeatures includes interaction features.

##We can restrict the features created to only interaction features by setting
##interaction_only to True:

interaction = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
print(interaction.fit_transform(features), "Interaction set to true")

##Polynomial features are often created when we want to include the notion
##that there exists a nonlinear relationship between the features and the target.
##For example, we might suspect that the effect of age on the probability of
##having a major medical condition is not constant over time but increases as
##age increases. We can encode that nonconstant effect in a feature, x, by
##generating that feature’s higher-order forms (x2, x3, etc.).

##Additionally, often we run into situations where the effect of one feature
##is dependent on another feature. A simple example would be if we were trying
##to predict whether or not our coffee was sweet and we had two features: 1)
##whether or not the coffee was stirred and 2) if we added sugar. Individually,
##each feature does not predict coffee sweetness, but the combination of their
##effects does. That is, a coffee would only be sweet if the coffee had sugar and
##was stirred. The effects of each feature on the target (sweetness) are dependent
##on each other. We can encode that relationship by including an interaction feature
##that is the product of the individual features.


############################################################################


#create a custom transformation to one or more features

#this section relies on from sklearn.preprocessing import FunctionTransformer

#In scikit-learn, use FunctionTransformer to apply a function to a set of features:

# Create feature matrix
features1 = np.array([[2, 3],
                     [2, 3],
                     [2, 3]])

# Define a simple function
def add_ten(x):
    return x + 10

# Create transformer
ten_transformer = FunctionTransformer(add_ten)

# Transform feature matrix
print(ten_transformer.transform(features1), "transform the features by 10")

# we can create the same transformation in pandas using apply

import pandas as pd

#set each column in features1 to be their own column in pandas
df = pd.DataFrame(features1, columns=['feature_1', 'feature_2'])

#Use apply function
print(df.apply(add_ten), "use pandas to apply function to features1")

##It is common to want to make some custom transformations to
##one or more features. For example, we might want to create a
##feature that is the natural log of the values of the different
##feature. We can do this by creating a function and then mapping
##it to features using either scikit-learn’s FunctionTransformer or
##pandas' apply. In the solution we created a very simple function,
##add_ten, which added 10 to each input, but there is no
##reason we could not define a much more complex function.

#testing
def exp(x):
    return np.exp(x)

exp_transformer = FunctionTransformer(exp)

print(exp_transformer.transform(features1), "using scikit to transform to e")

df1 = pd.DataFrame(features1, columns=['feature_1', 'feature_2'])

print(df.apply(exp), "using pandas to apply function of e")


############################################################################


# identify extreme observations and Outliers

#This section relies on from sklearn.covariance import EllipticEnvelope
#AND
#from sklearn.datasets import make_blobs

##Detecting outliers is unfortunately more of an art than a science.
##However, a common method is to assume the data is normally distributed
##and based on that assumption "draw" an ellipse around the data,
##classifying any observation inside the ellipse as an inlier (labeled as 1) and any
##observation outside the ellipse as an outlier (labeled as -1):

# Create simulated data
features2, _ = make_blobs(n_samples = 10,
                         n_features = 2,
                         centers = 1,
                         random_state = 1)

# Replace the first observation's values with extreme values
features2[0,0] = 10000
features2[0,1] = 10000

# Create detector
outlier_detector = EllipticEnvelope(contamination=.1)

# Fit detector
outlier_detector.fit(features2)

# Predict outliers
print(outlier_detector.predict(features2), "detect outliers with ellipse")

##A major limitation of this approach is the need to specify a contamination
##parameter, which is the proportion of observations that are outliers—​a value
##that we don’t know. Think of contamination as our estimate of the cleanliness
##of our data. If we expect our data to have few outliers, we can set contamination
##to something small. However, if we believe that the data is very likely to have
##outliers, we can set it to a higher value.

##Instead of looking at observations as a whole, we can instead look at individual
##features and identify extreme values in those features using interquartile range (IQR):

# Create one feature
feature = features[:,0]

# Create a function to return index of outliers
def indicies_of_outliers(x):
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    return np.where((x > upper_bound) | (x < lower_bound))

# Run function
print(indicies_of_outliers(feature), "Interquartile range")

##IQR is the difference between the first and third quartile
##of a set of data. You can think of IQR as the spread of the
##bulk of the data, with outliers being observations far from
##the main concentration of data. Outliers are commonly
##defined as any value 1.5 IQRs less than the first
##quartile or 1.5 IQRs greater than the third quartile.

##There is no single best technique for detecting outliers. Instead, we
##have a collection of techniques all with their own advantages and disadvantages.
##Our best strategy is often trying multiple techniques
##(e.g., both EllipticEnvelope and IQR-based detection) and looking at the results as a whole.

##If at all possible, we should take a look at observations we
##detect as outliers and try to understand them. For example,
##if we have a dataset of houses and one feature is number of
##rooms, is an outlier with 100 rooms really a house or is it
##actually a hotel that has been misclassified?


############################################################################


#Handling outliers

#typically we have three strategies we can use to handle outliers
# Load library
import pandas as pd

# Create DataFrame
houses = pd.DataFrame()
houses['Price'] = [534433, 392333, 293222, 4322032]
houses['Bathrooms'] = [2, 3.5, 2, 116]
houses['Square_Feet'] = [1500, 2500, 1500, 48000]

# Filter observations
print(houses[houses['Bathrooms'] < 20], "Houses dataset")

# Load library
import numpy as np

# Create feature based on boolean condition
houses["Outlier"] = np.where(houses["Bathrooms"] < 20, 0, 1)

# Show data
print(houses, "filtered for new data")

#Make a transform to dampen the effect of of the outlier

#Log Feature
houses["Log_Of_Square_Feet"] = [np.log(x) for x in houses["Square_Feet"]]

#Show data
print(houses, "dampened outliers with log")

##Similar to detecting outliers, there is no hard-and-fast rule for handling them.
##How we handle them should be based on two aspects. First, we should consider what
##makes them an outlier. If we believe they are errors in the data such as from a
##broken sensor or a miscoded value, then we might drop the observation or replace
##outlier values with NaN since we can’t believe those values. However, if we
##believe the outliers are genuine extreme values (e.g., a house [mansion] with 200 bathrooms),
##then marking them as outliers or transforming their values is more appropriate.

##Second, how we handle outliers should be based on our goal for machine learning.
##For example, if we want to predict house prices based on features of the house,
##we might reasonably assume the price for mansions with over 100 bathrooms is driven
##by a different dynamic than regular family homes. Furthermore, if we are training a
##model to use as part of an online home loan web application, we might assume that
##our potential users will not include billionaires looking to buy a mansion.

##So what should we do if we have outliers? Think about why they are outliers,
##have an end goal in mind for the data, and, most importantly, remember that not
##making a decision to address outliers is itself a decision with implications.

##One additional point: if you do have outliers standardization might not be appropriate
##because the mean and variance might be highly influenced by the outliers. In this case,
##use a rescaling method more robust against outliers like RobustScaler.


############################################################################


#Numerical feature to be broken into discrete bins

##Depending on how we want to break up the data, there are two techniques we can use.
##First, we can binarize the feature according to some threshold:

#This section relies uponfrom sklearn.preprocessing import Binarizer

# Create feature
age = np.array([[6],
                [12],
                [20],
                [36],
                [65]])

# Create binarizer
binarizer = Binarizer(18)

# Transform feature
print(binarizer.fit_transform(age), "Transform feature with binarizer")

# Break up data into multiple thresholds

# Bin feature
print(np.digitize(age, bins=[20,30,64]), "Broken into thresholds")

##Note that the arguments for the bins parameter denote the left edge of each bin.
##For example, the 20 argument does not include the element with the value of 20,
##only the two values smaller than 20. We can switch this behavior by
##setting the parameter right to True:

# Bin feature
print(np.digitize(age, bins=[20,30,64], right=True), "added right=True to include array(20)")

##Discretization can be a fruitful strategy when we have reason to believe
##that a numerical feature should behave more like a categorical feature.
##For example, we might believe there is very little difference in the spending
##habits of 19- and 20-year-olds, but a significant difference between 20- and
##21-year-olds (the age in the United States when young adults can consume alcohol).
##In that example, it could be useful to break up individuals in our data into those
##who can drink alcohol and those who cannot. Similarly, in other cases it might be
##useful to discretize our data into three or more bins.

##In the solution, we saw two methods of discretization—​scikit-learn’s
##Binarizer for two bins and NumPy’s digitize for three or more bins—​however,
##we can also use digitize to binarize features like Binarizer by only
##specifying a single threshold:

# Bin feature
print(np.digitize(age, bins=[18]), "using digitize to binarize")


############################################################################


#Cluster observations so that similar observations are grouped together

#If you know that you have k groups, you can use k-means clustering to group together
#similar observations and output a new feature containing each observations group membership

#This section relies upon from sklearn.cluster import KMeans

# Make simulated feature matrix
features, _ = make_blobs(n_samples = 50,
                         n_features = 2,
                         centers = 3,
                         random_state = 1)

# Create DataFrame
dataframe = pd.DataFrame(features, columns=["feature_1", "feature_2"])

# Make k-means clusterer
clusterer = KMeans(3, random_state=0)

# Fit clusterer
clusterer.fit(features)

# Predict values
dataframe["group"] = clusterer.predict(features)

# View first few observations
print(dataframe.head(5), "clustered with K_Means")

##We are jumping ahead of ourselves a bit and will go much more in depth
##about clustering algorithms later in the book. However, I wanted to point
##out that we can use clustering as a preprocessing step. Specifically,
##we use unsupervised learning algorithms like k-means to cluster observations
##into groups. The end result is a categorical feature with similar
##observations being members of the same group.

##Don’t worry if you did not understand all of that right now: just file
##away the idea that clustering can be used in preprocessing.


############################################################################


#Delete observations containing missing values

#Use numpy to delete observations with missing values

# Create feature matrix
features = np.array([[1.1, 11.1],
                     [2.2, 22.2],
                     [3.3, 33.3],
                     [4.4, 44.4],
                     [np.nan, 55]])

# Keep only observations that are not (denoted by ~) missing
print(features[~np.isnan(features).any(axis=1)], "Keep only observations that are non missing")

#USe pandas to drop missing values

# Load data
dataframe = pd.DataFrame(features, columns=["feature_1", "feature_2"])

# Remove observations with missing values
print(dataframe.dropna(), "dropping with pandas")

##Most machine learning algorithms cannot handle any missing values in the target
##and feature arrays. For this reason, we cannot ignore missing values in our data
##and must address the issue during preprocessing.

##The simplest solution is to delete every observation that contains
##one or more missing values, a task quickly and easily accomplished using NumPy or pandas.

##That said, we should be very reluctant to delete observations with missing values.
##Deleting them is the nuclear option, since our algorithm loses access to the information
##contained in the observation’s non-missing values.

##Just as important, depending on the cause of the missing values,
##deleting observations can introduce bias into our data. There are three types of missing data:

##Missing Completely At Random (MCAR): The probability that a value is missing is
##independent of everything. For example, a survey respondent rolls a die before
##answering a question: if she rolls a six, she skips that question.

##Missing At Random (MAR): The probability that a value is missing is not completely
##random, but depends on the information captured in other features. For example,
##a survey asks about gender identity and annual salary and women are more likely to
##skip the salary question; however, their nonresponse depends only on information we
##have captured in our gender identity feature.

##Missing Not At Random (MNAR): The probability that a value is missing is not
##random and depends on information not captured in our features. For example,
##a survey asks about gender identity and women are more likely to skip the salary
##question, and we do not have a gender identity feature in our data.

##It is sometimes acceptable to delete observations if they are MCAR or MAR.
##However, if the value is MNAR, the fact that a value is missing is itself
##information. Deleting MNAR observations can inject bias into our data because
##we are removing observations produced by some unobserved systematic effect.
