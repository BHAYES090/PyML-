import pandas as pd

#Create a new dataframe

# Create DataFrame
dataframe = pd.DataFrame()

# Add columns
# Creating the dataframe objects independently
dataframe['Name'] = ['Jacky Jackson', 'Steven Stevenson']
dataframe['Age'] = [38, 25]
dataframe['Driver'] = [True, False]

# Show DataFrame
#print(dataframe, "creating a dataframe")

# Create extra row
new_person = pd.Series(['Molly Mooney', 40, True], index=['Name', 'Age', 'Driver'])

#Add new person to dataframe
# Must set dataframe = dataframe(sub_data)
dataframe = dataframe.append(new_person, ignore_index=True)

#show updated dataframe
print(dataframe, "Create and add new person")

##pandas offers what can feel like an infinite number of ways to
##create a DataFrame. In the real world, creating an empty DataFrame
##and then populating it will almost never happen. Instead, our
##DataFrames will be created from real data we have loading from
##other sources (e.g., a CSV file or database).


################################################################


# View some characteristics of a DataFrame
url = 'https://tinyurl.com/titanic-csv'

#Load data
dataframe1 = pd.read_csv(url)

#Show first two rows
print(dataframe1.head(2), "Show frist two rows")

# Show last two rows
print(dataframe1.tail(2), "Show last two rows")

#show dimensions
print(dataframe1.shape, "Show dimensions (Rows and columns)")

#Show statistics
print(dataframe1.describe(), "Show stats")

##It is worth noting that summary statistics of describe() do not always tell the full story.
##For example, pandas treats the columns Survived and SexCode as numeric columns
##because they contain 1s and 0s. However, in this case the numerical values represent
##categories. For example, if Survived equals 1, it indicates that the passenger
##survived the disaster. For this reason, some of the summary statistics provided don’t
##make sense, such as the standard deviation of the SexCode column (an indicator of the passenger’s gender)


##################################################################


# Select individual slices from the dataframe
print(dataframe1.iloc[0], "Select the 0th element with iloc")

#Select a series of rows
print(dataframe1.iloc[1:4], "Select a series of rows")

#You can use the : to define a slice of rows you want to view
#Select everything up to the 4th row
print(dataframe1.iloc[:4], "Select everything up to the 4th row")

#Change the index
dataframe1 = dataframe1.set_index(dataframe1['Name'])

#Show row
print(dataframe1.loc['Allen, Miss Elisabeth Walton'], "Changed index to name and locate")

##All rows in a pandas DataFrame have a unique index value.
##By default, this index is an integer indicating the row position
##in the DataFrame; however, it does not have to be. DataFrame indexes
##can be set to be unique alphanumeric strings or customer numbers.
##To select individual rows and slices of rows, pandas provides two methods:
##
##loc is useful when the index of the DataFrame is a label (e.g., a string).
##
##iloc works by looking for the position in the DataFrame.
##For example, iloc[0] will return the first row regardless of whether the index is an integer or a label.
##
##It is useful to be comfortable with both loc and iloc since they will come up a lot during data cleaning.


##################################################################


#Select dataFrame rows based on conditions

#Show top two rows where column 'sex' is 'female'
print(dataframe1[dataframe1['Sex'] == 'female'].head(2), "Select first two rows where Sex is female")

##Take a second and look at the format of this solution. dataframe['Sex'] == 'female'
##is our conditional statement; by wrapping that in dataframe[] we are telling pandas
##to "select all the rows in the DataFrame where the value of dataframe['Sex'] is 'female'.

# Select all the rows where the passenger is female and 65 yo or older
print(dataframe1[(dataframe1['Sex'] == 'female') & (dataframe1['Age'] >= 65)],
      "Select first two rows where sex is F and age >= 65")

##Conditionally selecting and filtering data is one of the most common tasks
##in data wrangling. You rarely want all the raw data from the source; instead,
##you are interested in only some subsection of it. For example, you might only
##be interested in stores in certain states or the records of patients over a certain age.


####################################################################


#Replace the values in a DataFrame
print(dataframe1['Sex'].replace("female", "Woman").head(2), "Replace values")

#Replace female and male with Woman and Man
print(dataframe1['Sex'].replace(["female", "male"], ["Woman", "Man"]).head(5),
      "Replace values at the same time")

#Replace values across the entire dataframe
print(dataframe1.replace(1, "One").head(2), "Replace values across the entire dataframe")

#Replace also has regular expressions
print(dataframe1.replace(r"1st", "First", regex=True).head(2), "Use regular expression")

##replace is a tool we use to replace values that is simple and yet
##has the powerful ability to accept regular expressions.


####################################################################


#renaming columns
print(dataframe1.rename(columns={'PClass': 'Passenger Class'}).head(2), "renaming columns")

#Rename can accept a dictonary as a parameter, chnage multiple column names at once
print(dataframe1.rename(columns={'PClass': 'Passenger Class', 'SexCode': 'Gender'}).head(2),
      "renaming multiple columns")

##Using rename with a dictionary as an argument to the columns parameter
##is my preferred way to rename columns because it works with any number of columns.
##If we want to rename all columns at once, this helpful snippet of code creates a dictionary
##with the old column names as keys and empty strings as values:

# Load library
import collections

# Create Dictionary
column_names = collections.defaultdict(str)

# Create keys
for name in dataframe1.columns:
    column_names[name]

#show dictionary
print(column_names)


####################################################################


#Find the minimum, maximum, sum, average or count of a numerical column

#Calculate Statistics
print('Maximum', dataframe1['Age'].max())
print('Minimum', dataframe1['Age'].min())
print('Mean', dataframe1['Age'].mean())
print('Sum', dataframe1['Age'].sum())
print('Count', dataframe1['Age'].count())
print('Variance', dataframe1['Age'].var())
print('Standard Deviation', dataframe1['Age'].std())
print('Kurtosis', dataframe1['Age'].kurt())
print('Skewness', dataframe1['Age'].skew())
print('Standard Error of Mean', dataframe1['Age'].sem())
print('Mode', dataframe1['Age'].mode(), "End")
print('Median', dataframe1['Age'].median(), "END")

#apply the calulations to the entire dataframe
print(dataframe.count(), "Apply changes to entire dataframe")


####################################################################


#select unique values in a column
print(dataframe1['Sex'].unique(), "Select unique values from each column")

#Show counts
print(dataframe1['Sex'].value_counts(),
      "Display unique values with the number of times each value appears")

##Both unique and value_counts are useful for manipulating and exploring categorical columns.
##Very often in categorical columns there will be classes that need to be handled in the data
##wrangling phase. For example, in the Titanic dataset, PClass is a column indicating the class
##of a passenger’s ticket. There were three classes on the Titanic; however, if we use
##value_counts we can see a problem:

#Show counts
print(dataframe1['PClass'].value_counts(), "Using value_counts on PClass")

##While almost all passengers belong to one of three classes as expected, a single passenger
##has the class *. There are a number of strategies for handling this type of issue, but for
##now just realize that "extra" classes are common in categorical data and should not be ignored.
##Finally, if we simply want to count the number of unique values, we can use nunique:

# Show number of unique values
print(dataframe1['PClass'].nunique(), "Show number of unique values")


######################################################################


#Select missing values in a DataFrame
print(dataframe1[dataframe1['Age'].isnull()].head(2), "Select missing values")

# Attempt to replace values with NaN, returns an error becuase NaN is not defined
#print(dataframe1['Sex'] == dataframe1['Sex'].replace('male', NaN), "Replace missing valyes with NaN")

#import numpy to have full functionality with NaN
import numpy as np

#Run command again
print(dataframe1['Sex'] == dataframe1['Sex'].replace('male', np.nan),
      "Replace missing valyes with NaN after numpy import")

##Oftentimes a dataset uses a specific value to denote a missing observation,
##such as NONE, -999, or .. pandas' read_csv includes a parameter allowing us to
##specify the values used to indicate missing values:

# Load data, set missing values
dataframe1 = pd.read_csv(url, na_values=[np.nan, 'NONE', -999])


######################################################################


#Delete a column from your DataFrame

#delete a column
print(dataframe1.drop('Age', axis=1).head(2), "Delete a column")

#Use a list of column names to drop multiple columns at a time
print(dataframe1.drop(['Age', 'Sex'], axis=1).head(2), "Delete multiple columns at once")

#If a column does not have a name you can drop it using its index
print(dataframe1.drop(dataframe1.columns[1], axis=1).head(2), "Drop a column based off of its index")

##drop is the idiomatic method of deleting a column. An alternative method
##is del dataframe['Age'], which works most of the time but is not recommended
##because of how it is called within pandas (the details of which are
##outside the scope of this book).

##One habit I recommend learning is to never use pandas' inplace=True
##argument. Many pandas methods include an inplace parameter, which when
##True edits the DataFrame directly. This can lead to problems in more
##complex data processing pipelines because we are treating the DataFrames
##as mutable objects (which they technically are).
##I recommend treating DataFrames as immutable objects. For example:

#create a new dataframe
dataframe_name_dropped = dataframe.drop(dataframe.columns[0], axis=1)

print(dataframe_name_dropped, "dataframe name dropped")

##In this example, we are not mutating the DataFrame dataframe
##but instead are making a new DataFrame that is an altered version
##of dataframe called dataframe_name_dropped. If you treat your
##DataFrames as immutable objects, you will save yourself a lot of headaches down the road.


######################################################################


#Delete one or more rows from a dataframe

#Delete rows, show first two rows of output
print(dataframe1[dataframe1['Sex'] != 'male'].head(2), "Delete rows")

##While technically you can use the drop method (for example,
##df.drop([0, 1], axis=0) to drop the first two rows), a more
##practical method is simply to wrap a boolean condition inside
##df[]. The reason is because we can use the power of conditionals to delete
##either a single row or (far more likely) many rows at once.

#Delete row, show first two rows of output
print(dataframe1[dataframe1['Name'] != 'Allison, Miss Helen Loraine'].head(2), "Use boolean to delete rows")

#It may be more efficient to delete a row based off of index
# Delete row, show first two rows of output
print(dataframe1[dataframe1.index != 0].head(2), "Delete row on index")


######################################################################


#Drop duplicate rows from your DataFrame

#drop duplicates, show first two rows of output
print(dataframe1.drop_duplicates().head(2), "Dropping duplicates")

#Show number of rows
print("Number of rows in the original dataframe: ",
      len(dataframe1))
print("Number of rows after duplicating: ",
      len(dataframe1.drop_duplicates()))

#Every row of the dataframe is unique from the next, so no duplication drop actually occured
#You can add parameters to drop_duplicates to specify a column or subset

print(dataframe1.drop_duplicates(subset=['Sex']), "Dropping duplicates in Sex")

#drop_duplicates chooses only the first two rows by default, using the keep parameter this can be edited
print(dataframe1.drop_duplicates(subset=['Sex'], keep='last'), "use keep parameter to edit the output")


######################################################################


#Group individual rows according to some shared value

#group rows by the values of the column "Sex", calculate mean of each group

print(dataframe1.groupby('Sex').mean(), "use groupby to group sex and take mean")

##groupby is where data wrangling really starts to take shape.
##It is very common to have a DataFrame where each row is a person
##or an event and we want to group them according to some criterion
##and then calculate a statistic. For example, you can imagine a
##DataFrame where each row is an individual sale at a national
##restaurant chain and we want the total sales per restaurant.
##We can accomplish this by grouping rows by individual resturants
##and then calculating the sum of each group.
##Users new to groupby often write a line like this and are confused by what is returned:

print(dataframe1.groupby('Sex'), "Returns nothing useful")

#groupby needs to be paired with some operation we want to apply to each group, such as taking a statistic

#Group rows, count rows
print(dataframe1.groupby('Survived')['Name'].count(), "Preforming groupby with operations")

# Name is added after groupby because particular summary statistics are only meaningful to certian
#types of data

#We can also group by a first column, then group that grouping by a second column:

#Group rows, calculate mean
print(dataframe1.groupby(['Sex', 'Survived'])['Age'].mean(), "grouping rows and calculating mean")


######################################################################


#Load Libraries
import numpy as np

#Create time range
time_index = pd.date_range('06/06/2017', periods=100000, freq='30S')

#Create Dataframe
dataframe2 = pd.DataFrame(index=time_index)

#Create column of random values
dataframe2['Sale_Amount'] = np.random.randint(1, 10, 100000)

#Group rows by week, calculate sum per week
print(dataframe2.resample('W').sum(), "Dataframe2 sample time")

#Show first three rows
print(dataframe2.head(3), "Show first three rows")

##Notice that the date and time of each sale is the index of the DataFrame; this is
##because resample requires the index to be datetime-like values.

#Group by two weeks, calculate mean
print(dataframe2.resample('2W').mean(), "Group by 2 weeks and calculate mean")

#Group by Month, count rows
print(dataframe2.resample('M').count(), "group by month and count")

#Group by month, count rows
print(dataframe2.resample('M', label='left').count(), "group by month and count ROWS")


######################################################################


#Iterate over every element in a column and apply some action

for name in dataframe1['Name'][0:2]:
    print(name.upper(), "Iterate through a column and capitalize")

#You can also use list comprehenions

print([name.upper() for name in dataframe1['Name'][0:2]], "Use list comprehension")


######################################################################


#Apply some function over all elements in a column

#Create function
def uppercase(x):
    return x.upper()

#Apply function, show two rows

print(dataframe1['Name'].apply(uppercase)[0:2], "Apply function to column in dataframe")

##apply is a great way to do data cleaning and wrangling.
##It is common to write a function to perform some useful
##operation (separate first and last names, convert strings to floats, etc.)
##and then map that function to every element in a column.


####################################################################


#Once grouping rows with groupby, apply a funciton to each group

#Group rows, apply function to groups
print(dataframe1.groupby('Sex').apply(lambda x: x.count()), "Preforming apply on groupby groups")

##In Applying a Function Over All Elements in a
##Column I mentioned apply. apply is particularly
##useful when you want to apply a function to groups.
##By combining groupby and apply we can calculate custom statistics
##or apply any function to each group separately.


####################################################################


#Concatenate two Dataframes

#Create a NEW dataframe
data_a = {'id': ['1', '2', '3'],
          'first': ['Alex', 'Amy', 'Allen'],
          'last': ['Anderson', 'Ackerman', 'Ali']}
dataframe_a = pd.DataFrame(data_a, columns = ['id', 'first', 'last'])

# Create DataFrame
data_b = {'id': ['4', '5', '6'],
          'first': ['Billy', 'Brian', 'Bran'],
          'last': ['Bonder', 'Black', 'Balwner']}
dataframe_b = pd.DataFrame(data_b, columns = ['id', 'first', 'last'])

# Concatenate DataFrames by rows
print(pd.concat([dataframe_a, dataframe_b], axis=0), "Concatenate two dataframes by row")

#Use axis=1 to concatenate by columns
print(pd.concat([dataframe_a, dataframe_b], axis=1), "Concatenate by columns")

##Concatenating is not a word you hear much outside of computer science
##and programming, so if you have not heard it before, do not worry.
##The informal definition of concatenate is to glue two objects together.
##In the solution we glued together two small DataFrames using the axis
##parameter to indicate whether we wanted to stack the two DataFrames on top
##of each other or place them side by side.
##Alternatively we can use append to add a new row to a DataFrame:

#create a new row for sample dataframe
row = pd.Series([10, 'Chris', 'Chillon'], index=['id', 'first', 'last'])

#Append Row
print(dataframe_a.append(row, ignore_index=True), "Append dataframe with new data")


##################################################################



#Merge two dataframes

# Create DataFrame
employee_data = {'employee_id': ['1', '2', '3', '4'],
                 'name': ['Amy Jones', 'Allen Keys', 'Alice Bees',
                 'Tim Horton']}
dataframe_employees = pd.DataFrame(employee_data, columns = ['employee_id',
                                                              'name'])

# Create DataFrame
sales_data = {'employee_id': ['3', '4', '5', '6'],
              'total_sales': [23456, 2512, 2345, 1455]}
dataframe_sales = pd.DataFrame(sales_data, columns = ['employee_id',
                                                      'total_sales'])

# Merge DataFrames how not specified
print(pd.merge(dataframe_employees, dataframe_sales, on='employee_id'), "Merge two dataframes")

# Merge DataFrames how='outer'
print(pd.merge(dataframe_employees, dataframe_sales, on='employee_id', how='outer'), "Merge along outer join")

# Merge DataFrames how='left'
print(pd.merge(dataframe_employees, dataframe_sales, on='employee_id', how='left'), "Merge along left join")

# Merge DataFrames specified columns
print(pd.merge(dataframe_employees,
         dataframe_sales,
         left_on='employee_id',
         right_on='employee_id'), "Merge along columns")

##If instead of merging on two columns we want to merge on the indexes of
##each DataFrame, we can replace the left_on and right_on parameters with
##right_index=True and left_index=True.

##Oftentimes, the data we need to use is complex; it doesn’t always come
##in one piece. Instead in the real world, we’re usually faced with
##disparate datasets, from multiple database queries or files. To get
##all that data into one place, we can load each data query or data file
##into pandas as individual DataFrames and then merge them together into
##a single DataFrame.

##This process might be familiar to anyone who has used SQL, a popular
##language for doing merging operations (called joins). While the exact
##parameters used by pandas will be different, they follow the same
##general patterns used by other software languages and tools.

##There are three aspects to specify with any merge operation.
##First, we have to specify the two DataFrames we want to merge
##together. In the solution we named them dataframe_employees
##and dataframe_sales. Second, we have to specify the name(s)
##of the columns to merge on—​that is, the columns whose values
##are shared between the two DataFrames. For example, in our
##solution both DataFrames have a column named employee_id.
##To merge the two DataFrames we will match up the values in
##each DataFrame’s employee_id column with each other. If these
##two columns use the same name, we can use the on parameter.
##However, if they have different names we can use left_on and right_on.

##What is the left and right DataFrame? The simple answer is that
##the left DataFrame is the first one we specified in merge and
##the right DataFrame is the second one. This language comes up
##again in the next sets of parameters we will need.

##The last aspect, and most difficult for some people to grasp,
##is the type of merge operation we want to conduct. This is
##specified by the how parameter. merge supports the four main types of joins:

##Inner

##Return only the rows that match in both DataFrames
##(e.g., return any row with an employee_id value appearing
## in both dataframe_employees and dataframe_sales).

##Outer

##Return all rows in both DataFrames. If a row exists
##in one DataFrame but not in the other DataFrame, fill
##NaN values for the missing values (e.g., return all rows in
##both employee_id and dataframe_sales).

##Left

##Return all rows from the left DataFrame but only rows
##from the right DataFrame that matched with the left DataFrame.
##Fill NaN values for the missing values (e.g., return all rows
##from dataframe_employees but only rows from dataframe_sales
##that have a value for employee_id that appears in dataframe_employees).

##Right

##Return all rows from the right DataFrame but only rows from
##the left DataFrame that matched with the right DataFrame.
##Fill NaN values for the missing values (e.g., return all rows
##from dataframe_sales but only rows from dataframe_employees
##that have a value for employee_id that appears in dataframe_sales).

##If you did not understand all of that right now, I encourage
##you to play around with the how parameter in your code and see
##how it affects what merge returns.
