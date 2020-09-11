
# Import libraries necessary for this project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display # Allows the use of display() for DataFrames

# Pretty display for notebooks
%matplotlib inline
plt.style.use('fivethirtyeight')

# Load the accepted loan dataset 
# low_memory and skiprows in read_csv because the file is large and it leads to the Lending Club website
try:
    loan_data = pd.read_csv("LoanStats3a.csv", low_memory = False, skiprows = 1)
    print("The loan dataset has {} samples with {} features.".format(*loan_data.shape))
except:
    print("The loan dataset could not be loaded. Is the dataset missing?")


loan_data.head()
loan_data = loan_data.drop(['desc', 'url'],axis=1)
loan_data.describe()

# count half point of the dataset.
half_point = len(loan_data) / 2
loan_data = loan_data.dropna(thresh=half_point, axis=1)
# we save the new file
loan_data.to_csv('loan_data.csv', index=False)
loan_data = pd.read_csv('loan_data.csv', low_memory = False)
loan_data.drop_duplicates()

loan_data.iloc[0]

loan_data.shape[1]

first_entry = loan_data.iloc[0]
first_entry.to_csv('first_entry.csv', index = True)

#We drop the columns enumerated in the cell above.
loan_data = loan_data.drop(['id', 'member_id', 'funded_amnt', 'funded_amnt_inv', 
                              'grade', 'sub_grade', 'emp_title'], axis =1)

loan_data = loan_data.drop(['out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv',
                              'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee'], axis =1)

loan_data = loan_data.drop(['recoveries', 'collection_recovery_fee', 
                              'last_pymnt_d', 'last_pymnt_amnt'], axis =1)
loan_data = loan_data.drop(['issue_d','zip_code'], axis =1)
loan_data.shape


loan_data['loan_status'].value_counts()

loan_data['loan_status'].value_counts().plot(kind= 'barh', color = 'orange', title = 'Possible Loan Status', alpha = 0.75)
plt.show()
loan_data = loan_data[(loan_data['loan_status'] == "Fully Paid") | (loan_data['loan_status'] == "Charged Off")]


status_replace = {
    "loan_status" : {
        "Fully Paid": 1,
        "Charged Off": 0,
    }
}
loan_data = loan_data.replace(status_replace)

loan_data['loan_status'].value_counts()

loan_data.shape

orig_columns = loan_data.columns
drop_columns = []
for col in orig_columns:
    col_series = loan_data[col].dropna().unique()
    if len(col_series) == 1:
        drop_columns.append(col)
loan_data = loan_data.drop(drop_columns, axis = 1)
drop_columns

loan_data.shape
null_counts = loan_data.isnull().sum()
null_counts
loan_data = loan_data.drop("pub_rec_bankruptcies", axis=1)
loan_data = loan_data.dropna(axis=0)

loan_data.shape

print(loan_data.dtypes.value_counts())
object_columns_df = loan_data.select_dtypes(include=["object"])
print( object_columns_df.iloc[0])

columns = ['term', 'emp_length', 'home_ownership', 'verification_status', 'addr_state']
for col in columns:
    print( loan_data[col].value_counts())
    print (" ")
    
    print( loan_data["purpose"].value_counts())
print (" ")
print (loan_data["title"].value_counts())

mapping_dict = {
    "emp_length": {
        "10+ years": 10,
        "9 years": 9,
        "8 years": 8,
        "7 years": 7,
        "6 years": 6,
        "5 years": 5,
        "4 years": 4,
        "3 years": 3,
        "2 years": 2,
        "1 year": 1,
        "< 1 year": 0,
        "n/a": 0
    }
}
loan_data = loan_data.drop(["last_credit_pull_d", "earliest_cr_line", "addr_state", "title"], axis=1)
loan_data["int_rate"] = loan_data["int_rate"].str.rstrip("%").astype("float")
loan_data["revol_util"] = loan_data["revol_util"].str.rstrip("%").astype("float")
loan_data = loan_data.replace(mapping_dict)

categorical_columns = ["home_ownership", "verification_status", "emp_length", "purpose", "term"]
dummy_df = pd.get_dummies(loan_data[categorical_columns])
loan_data = pd.concat([loan_data, dummy_df], axis=1)
loan_data = loan_data.drop(categorical_columns, axis=1)

loan_data.head()

# cleaned and filtered data to csv
loan_data.to_csv('clean_loan_data.csv', index = False)
loan_data = pd.read_csv('clean_loan_data.csv')


predictions = pd.Series(np.ones(loan_data.shape[0]))

false_positive_filter = (predictions == 1) & (loan_data['loan_status'] == 0)
false_positive = len(predictions[false_positive_filter])

true_positive_filter = (predictions == 1) & (loan_data['loan_status'] == 1)
true_positive = len(predictions[true_positive_filter])

false_negative_filter = (predictions == 0) & (loan_data['loan_status'] == 1)
false_negative = len(predictions[false_negative_filter])

true_negative_filter = (predictions == 0) & (loan_data['loan_status'] == 0)
true_negative = len(predictions[true_negative_filter])

true_positive_rate = true_positive / (true_positive + false_negative)
false_positive_rate = false_positive / (false_positive + true_negative)

print (float(true_positive_rate) )
print (float(false_positive_rate))

accuracy = float(true_positive + true_negative)/float(true_positive + false_positive+ false_negative + true_negative)
accuracy
precision = float(true_positive)/float(true_positive + false_positive)
precision

# Data to plot
labels = 'False Positive', 'True Positive'
sizes = [1-precision, precision]
colors = ['lightcoral', 'lightblue'] 
# Plot
plt.figure(figsize=(4,4))
plt.pie(sizes, colors=colors, autopct='%1.2f%%', shadow=False, startangle=0)
plt.title('Precision Of Lending', fontsize=12) 
plt.legend(labels, loc='lower left', fontsize=10)
plt.axis('equal')
plt.show()

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

cols = loan_data.columns
train_cols = cols.drop('loan_status')

features = loan_data[train_cols]

target = loan_data['loan_status']

lr.fit(features, target)
predictions = lr.predict(features)

from sklearn.cross_validation import cross_val_predict, KFold
lr = LogisticRegression()
kf = KFold(features.shape[0], random_state=42)
predictions = cross_val_predict(lr, features, target, cv=kf)
predictions = pd.Series(predictions)

false_positive_filter = (predictions == 1) & (loan_data['loan_status'] == 0)
false_positive = len(predictions[false_positive_filter])

true_positive_filter = (predictions == 1) & (loan_data['loan_status'] == 1)
true_positive = len(predictions[true_positive_filter])

false_negative_filter = (predictions == 0) & (loan_data['loan_status'] == 1)
false_negative = len(predictions[false_negative_filter])

true_negative_filter = (predictions == 0) & (loan_data['loan_status'] == 0)
true_negative = len(predictions[true_negative_filter])

true_positive_rate = float(true_positive)/float((true_positive + false_negative))
false_positive_rate = float(false_positive)/float((false_positive + true_negative))

print (float(true_positive_rate) )
print (float(false_positive_rate))

precision = float(true_positive)/float(true_positive + false_positive)
precision

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_predict

rf = RandomForestClassifier(class_weight="balanced", random_state=1)
kf = KFold(features.shape[0], random_state=42)

predictions = cross_val_predict(rf, features, target, cv=kf)
predictions = pd.Series(predictions)

false_positive_filter = (predictions == 1) & (loan_data['loan_status'] == 0)
false_positive = len(predictions[false_positive_filter])

true_positive_filter = (predictions == 1) & (loan_data['loan_status'] == 1)
true_positive = len(predictions[true_positive_filter])

false_negative_filter = (predictions == 0) & (loan_data['loan_status'] == 1)
false_negative = len(predictions[false_negative_filter])

true_negative_filter = (predictions == 0) & (loan_data['loan_status'] == 0)
true_negative = len(predictions[true_negative_filter])

true_positive_rate = float(true_positive)/float((true_positive + false_negative))
false_positive_rate = float(false_positive)/float((false_positive + true_negative))

print (float(true_positive_rate) )
print (float(false_positive_rate))
# Data to plot
labels = 'False Positive', 'True Positive'
sizes = [1-precision, precision]
colors = ['lightcoral', 'lightblue'] 
# Plot
plt.figure(figsize=(4,4))
plt.pie(sizes, colors=colors, autopct='%1.2f%%', shadow=False, startangle=0)
plt.title('Precision of Random Forest', fontsize=12) 
plt.legend(labels, loc='lower left', fontsize=10)
plt.axis('equal')
plt.show()