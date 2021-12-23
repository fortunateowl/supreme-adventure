import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from DataClean import DataClean as DC

# Get the HVC data from the UCI machine learning archive

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00571/hcvdat0.csv"
Blood = pd.read_csv(url)

# Columns, except the first one are already named, so change the name of
# Column 1 to PatientID

Blood.rename(columns={'Unnamed: 0': 'PatientID'}, inplace=True)


# Create a DataFrame to compare the original data versus the cleanded data.
Compare = pd.DataFrame(columns = ['Variable', 'Missing_Values', 'Outliers', 'STDV_Before', 'STDV_After'])
Compare.loc[:,'Variable'] = Blood.columns.values
Compare.loc[:, 'Missing_Values'] = np.array(Blood.isnull().sum())


# Drop rows with multiple missing values
Blood.dropna(thresh=13, inplace=True)



d = DC()



for i in range (0, Blood.shape[1]):
    if pd.api.types.is_numeric_dtype(Blood.iloc[:, i]) == 1:
        Blood.iloc[:, i] = d.fill_median(Blood.iloc[:, i])



Blood_Compare = Blood.copy()


# replace outliers with the median
# using replace_outlier function in DataClean class.
# The third column "Age" is excluded from the process even though it is
# numeric.  The minimum age of 19 and the maximum age of 77 are not unreasonable.


for i in range (0,Blood.shape[1]):
    
    if pd.api.types.is_numeric_dtype(Blood.iloc[:, i]) == 1 and i != 2:
        Blood.iloc[:, i] = d.replace_outlier(Blood.iloc[:, i])
    
    # Counts the outliers in each variable
    X = np.where(Blood_Compare.iloc[:, i] != Blood.iloc[:, i], True, False)
    Compare.iloc[i, 2] = X.sum()

# show the histograms of each numeric variable, excluding PatientID
# as that is a unique identifier for each observation.

for i in range (0,Blood.shape[1]):
  
    if i != 0 and pd.api.types.is_numeric_dtype(Blood.iloc[:, i]) == 1:
        plt.hist(Blood.iloc[:,i])
        plt.xlabel(Blood.columns.values[i])
        plt.ylabel("Frequency")
        plt.show()
        
# Create a scatter plot matrix for the dataframe        

scatter_matrix(Blood,figsize=[20,20], s=1000) 
plt.show()    


# Finds the standard deviation of the variables before removing outliers
for i in range (0,Blood_Compare.shape[1]):
    
    if i != 0 and pd.api.types.is_numeric_dtype(Blood_Compare.iloc[:, i]) == 1:
        Compare.iloc[i, 3] = Blood_Compare.iloc[:,i].std()




# Print the standard deviations of each numeric variable, excluding PatientID
# as that is a unique identifier for each observation.

for i in range (0,Blood.shape[1]):
    
    if i != 0 and pd.api.types.is_numeric_dtype(Blood.iloc[:, i]) == 1:
        Z = Blood.iloc[:,i].std()
        Compare.iloc[i, 4] = Z
        
print ('\n Summary Table')
print("\n", Compare)


# For this assignment I used the data from the UCI Machine Learning Archive
# hcvdat0.csv.  The file contains 14 attributes and 615 observations. The Data
# Frame "Compare," printed in your console, shows some of the information for
# hvcdat0.  
#
# First, there are a total of 31 missing values, most being in two attributes,
# ALP anbd CHOL.  I removed rows with multiple missing values, resulting
# in three fewer observation.  Second I replaced the remaining missing values
# with the median of the attrbute.  
#
# Second, I replaced outliers with the median values of the attribute.
# Outliers were defined using the mean plus or minus two standard deviations.
# The column "Outliers" in Compare, shows the number of replacements in each
# attribute.
#
# The final two columns in Compare show the standard deviations before and
# after the outliers are removed.
#
# Histograms were created of all numeric data, excluding PatiendID.  PROT, CREA,
# CGOL, CHE, and ALP, show roughly gaussian or normal distributions, while, CGT,
# shows exponetial decay and BIL, AST, and ALT, have right-skewed distributions.
#
# Caveats:
# The Age attribute technically has outliers, but the min and max ages are
# not unreasonable, so I did not make any changes
#
# The ALP attribute has by far the most missing values, dropping the column
# or the all of the rows containing these missing values is not advised because
# all of the missing values occur in categories other than the default
# categore of "Blood Doner." It is at least suggested that the values are
# missing non-randomly.  While I assigned the median value as the lesson
# suggests, I would be more comfortable assigning conditional means or medians
# based on category or groups of categories.
