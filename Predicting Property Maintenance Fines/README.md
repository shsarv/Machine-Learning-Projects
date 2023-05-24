<div class="cell markdown">

# Predicting Property Maintenance Fines

This is based on a data challenge from the Michigan Data Science Team
([MDST](http://midas.umich.edu/mdst/)).

The Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/))
and the Michigan Student Symposium for Interdisciplinary Statistical
Sciences ([MSSISS](https://sites.lsa.umich.edu/mssiss/)) have partnered
with the City of Detroit to help solve one of the most pressing problems
facing Detroit - blight. [Blight
violations](http://www.detroitmi.gov/How-Do-I/Report/Blight-Complaint-FAQs)
are issued by the city to individuals who allow their properties to
remain in a deteriorated condition. Every year, the city of Detroit
issues millions of dollars in fines to residents and every year, many of
these fines remain unpaid. Enforcing unpaid blight fines is a costly and
tedious process, so the city wants to know: how can we increase blight
ticket compliance?

The first step in answering this question is understanding when and why
a resident might fail to comply with a blight ticket. This is where
predictive modeling comes in. For this assignment, your task is to
predict whether a given blight ticket will be paid on time.

All data for this assignment has been provided to us through the
[Detroit Open Data Portal](https://data.detroitmi.gov/). Nonetheless, we
encourage you to look into data from other Detroit datasets to help
inform feature creation and model selection. We recommend taking a look
at the following related datasets:

  - [Building
    Permits](https://data.detroitmi.gov/Property-Parcels/Building-Permits/xw2a-a7tf)
  - [Trades
    Permits](https://data.detroitmi.gov/Property-Parcels/Trades-Permits/635b-dsgv)
  - [Improve Detroit: Submitted
    Issues](https://data.detroitmi.gov/Government/Improve-Detroit-Submitted-Issues/fwz3-w3yn)
  - [DPD: Citizen
    Complaints](https://data.detroitmi.gov/Public-Safety/DPD-Citizen-Complaints-2016/kahe-efs3)
  - [Parcel
    Map](https://data.detroitmi.gov/Property-Parcels/Parcel-Map/fxkw-udwf)

-----

We provide you with two data files for use in training and validating
your models: train.csv and test.csv. Each row in these two files
corresponds to a single blight ticket, and includes information about
when, why, and to whom each ticket was issued. The target variable is
compliance, which is True if the ticket was paid early, on time, or
within one month of the hearing data, False if the ticket was paid after
the hearing date or not at all, and Null if the violator was found not
responsible. Compliance, as well as a handful of other variables that
will not be available at test-time, are only included in train.csv.

Note: All tickets where the violators were found not responsible are not
considered during evaluation. They are included in the training set as
an additional source of data for visualization, and to enable
unsupervised and semi-supervised approaches. However, they are not
included in the test set.

<br>

**File descriptions** (Use only this data for training your model\!)

    train.csv - the training set (all tickets issued 2004-2011)
    test.csv - the test set (all tickets issued 2012-2016)
    addresses.csv & latlons.csv - mapping from ticket id to addresses, and from addresses to lat/lon coordinates. 
     Note: misspelled addresses may be incorrectly geolocated.

<br>

**Data fields**

train.csv & test.csv

    ticket_id - unique identifier for tickets
    agency_name - Agency that issued the ticket
    inspector_name - Name of inspector that issued the ticket
    violator_name - Name of the person/organization that the ticket was issued to
    violation_street_number, violation_street_name, violation_zip_code - Address where the violation occurred
    mailing_address_str_number, mailing_address_str_name, city, state, zip_code, non_us_str_code, country - Mailing address of the violator
    ticket_issued_date - Date and time the ticket was issued
    hearing_date - Date and time the violator's hearing was scheduled
    violation_code, violation_description - Type of violation
    disposition - Judgment and judgement type
    fine_amount - Violation fine amount, excluding fees
    admin_fee - $20 fee assigned to responsible judgments

state\_fee - $10 fee assigned to responsible judgments late\_fee - 10%
fee assigned to responsible judgments discount\_amount - discount
applied, if any clean\_up\_cost - DPW clean-up or graffiti removal cost
judgment\_amount - Sum of all fines and fees grafitti\_status - Flag for
graffiti violations

train.csv only

    payment_amount - Amount paid, if any
    payment_date - Date payment was made, if it was received
    payment_status - Current payment status as of Feb 1 2017
    balance_due - Fines and fees still owed
    collection_status - Flag for payments in collections
    compliance [target variable for prediction] 
     Null = Not responsible
     0 = Responsible, non-compliant
     1 = Responsible, compliant
    compliance_detail - More information on why each ticket was marked compliant or non-compliant

-----

## Evaluation

Your predictions will be given as the probability that the corresponding
blight ticket will be paid on time.

The evaluation metric for this assignment is the Area Under the ROC
Curve (AUC). \_\_\_

For this assignment, create a function that trains a model to predict
blight ticket compliance in Detroit using `train.csv`. Using this model,
return a series of length 61001 with the data being the probability that
each corresponding ticket from `test.csv` will be paid, and the index
being the ticket\_id.

Example:

    ticket_id
       284932    0.531842
       285362    0.401958
       285361    0.105928
       285338    0.018572
                 ...
       376499    0.208567
       376500    0.818759
       369851    0.018528
       Name: compliance, dtype: float32

</div>

<div class="cell markdown">

-----

## A Quick Summary

Using Python, the data was imported and then processed using several
techniques. For instance:

    Specific features were dropped due to either data leakage or inconsistencies
    Violation address was converted to the respected lat/lon coordinate pairs
    Categorical data was 'one-hot-encoded' with a defined frequency threshold
    Feature engineering was performed on two datetime features by taking the day difference
    Models were trained with the roc_auc scoring, as opposed to just accuracy (allow for skewedness in data!)
    Two classifier algorithms were fitted, Gradient Boosting and Logistic Regression
    'Some' fine tuning was applied to increase the auc score

Gradient Boosting performed the best, with a 'fine-tuned' score on a
5-fold CV of 0.8057 (rounded). To summarize:

    Gradient Boosting Roc_Auc Score = 0.805666
    Logistic Regression Roc_Auc Score = 0.787156

</div>

<div class="cell markdown" data-collapsed="true">

# Import the data

</div>

<div class="cell code" data-execution_count="198" data-collapsed="true">

``` python
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier  #GB algorithm
from sklearn.linear_model import LogisticRegression # LR algorithm
from sklearn.model_selection import cross_val_score, GridSearchCV # Additional scklearn functions
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc # Scoring metrics to be used

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
%matplotlib notebook
```

</div>

<div class="cell markdown">

After importing the libraries, we can read in the data. Note that the
dataframes for address.csv and latlons.csv are merged on the ticket\_id
with the train dataframe. We will use the lat/lon data as a replacement
for the violation street name and street number (note that the violation
zip code was dropped from the set as the majority is NaN).

</div>

<div class="cell code" data-execution_count="2">

``` python
## Read in data
dtypes = {'ticket_issued_date': 'str', 'hearing_date': 'str'} # set known date labels to strings for conversion to dt
parse_dates = ['ticket_issued_date', 'hearing_date'] # create list of date labels

df_train = pd.read_csv('train.csv',encoding = "ISO-8859-1", 
                        dtype = dtypes, parse_dates = parse_dates) # Read in train.csv
df_test = pd.read_csv('test.csv',encoding = "ISO-8859-1", 
                       dtype = dtypes, parse_dates = parse_dates) # Read in test.csv

# Let's import addresses and accompanying lat/lons and merge on address
df_address = pd.read_csv('addresses.csv', encoding = "ISO-8859-1") # Read in addresses.csv (locations of violations in Detroit)
df_latlons = pd.read_csv('latlons.csv', encoding = "ISO-8859-1") # Lat/lons of violation locactions
df_id_latlons = pd.merge(df_address, df_latlons, how='inner', on='address') # Merge the address and lat/lons on ticket_id

# Drop address label now that it's merged
df_id_latlons.drop('address', axis = 1, inplace = True)
```

<div class="output stream stderr">

    C:\Users\Michael\Anaconda3\lib\site-packages\IPython\core\interactiveshell.py:2698: DtypeWarning: Columns (11,12,31) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)

</div>

</div>

<div class="cell markdown">

## Drop obvious labels

Any labels that appear to cause either data leakage, inconsistency, or
have a majoirty of NaN's are removed from the dataset. Furthermore, we
will also drop sampels corresponding to a target value of NaN (blight
offenders that were found not responsible).

</div>

<div class="cell code" data-execution_count="3" data-collapsed="true">

``` python
## Drop data labels that should not be used in analysis
#1 Get labels that are inconsistent with test
inconsistent_labels = ['payment_date', 'payment_status', 'collection_status', 
               'compliance_detail', 'balance_due', 'payment_amount']

#2 Get data leakage labels
data_leak_labels = ['violator_name', 'inspector_name']

#3 Get NaN's labels (col's with majority of NaN's)
NaN_labels = (df_train.isnull().sum() / len(df_train)) <= 0.50
maj_NaN_labels = NaN_labels[NaN_labels==False].index.tolist()

#4 Combine labels and drop from train
labels_to_remove = []
labels_to_remove.extend(inconsistent_labels + data_leak_labels + maj_NaN_labels)
df_train.drop(labels_to_remove, axis=1, 
        inplace = True)

#5 Remove NaN's from compliance label (just keep targets)
compliance_to_keep = df_train.compliance.notnull()
df_train = df_train.loc[compliance_to_keep, :]
```

</div>

<div class="cell markdown">

After performing steps 1-5 above, the labels that remain in the training
set are as such:

</div>

<div class="cell code" data-execution_count="4">

``` python
df_train.columns.tolist()
```

<div class="output execute_result" data-execution_count="4">

    ['ticket_id',
     'agency_name',
     'violation_street_number',
     'violation_street_name',
     'mailing_address_str_number',
     'mailing_address_str_name',
     'city',
     'state',
     'zip_code',
     'country',
     'ticket_issued_date',
     'hearing_date',
     'violation_code',
     'violation_description',
     'disposition',
     'fine_amount',
     'admin_fee',
     'state_fee',
     'late_fee',
     'discount_amount',
     'clean_up_cost',
     'judgment_amount',
     'compliance']

</div>

</div>

<div class="cell markdown">

Note that before steps 1-5 were taken, we had:

    ticket_id
    agency_name
    inspector_name
    violator_name
    violation_street_number, violation_street_name, violation_zip_code
    mailing_address_str_number, mailing_address_str_name, city, state, zip_code, non_us_str_code, country
    ticket_issued_date 
    hearing_date 
    violation_code, violation_description 
    disposition 
    fine_amount
    admin_fee 
    state_fee 
    late_fee 
    discount_amount
    clean_up_cost
    judgment_amount
    grafitti_status
    payment_amount 
    payment_date
    payment_status
    balance_due
    collection_status
    compliance_detail

Moving forward, we now have only 22 labels, as opposed to the original
33 (excluding compliance, our target label). Note that the removal fo
the 11 labels was due to the preprocessing of the data in steps 1-5
above.

</div>

<div class="cell markdown">

## Process the data

To begin, we will remove any remaining labels that we do not want to
include in the analysis. We chose the mailing location of the violator
to not be included:

    ['mailing_address_str_number', 'mailing_address_str_name', 
                  'city', 'state', 'zip_code', 'country']

It is possible that this data can indeed have a positive impact of the
AUC train score, but it is assumed that this data can vary too greatly
in the test set (and future test sets), and thus our model will not be
generalized well enough. However, it is worth considering and should be
included in the model when feature engineering becomes a must.

Secondly, we will replace the violation street name / number with the
corresponding lat/lon and fill the lat/lon NaN's with the most frequent
value.

Third, we'll splice out the columns that are purely objects, and call
our own Categories function to convert said columns into categories,
specify a frequency threshold, and bucket categories below the threshold
as **'unknown'**.

Finally, we'll call get\_dummies to complete our one-hot-encoding
process of the object data. Note that each label was carefully examined
to ensure that one-hot-encoding was the best approach to categorize the
data. We noted that each object label were strings that were *not
sorted*. If any object was indeed a sorted label (e.g. (low, medium,
high), then we would have simply called the method cat.codes and not the
function get\_dummies.

</div>

<div class="cell code" data-execution_count="5" data-collapsed="true">

``` python
## Process data
#1 Drop remaining address columns
labels_address = ['mailing_address_str_number', 'mailing_address_str_name', 
                  'city', 'state', 'zip_code', 'country']
df_train.drop(labels_address, axis = 1, 
        inplace = True)

#2 Let's merge violation street name / number with corresponding lat/lons
# and fill na's with most frequent of each column
df_id_latlons = df_id_latlons.apply(lambda x:x.fillna(x.value_counts().index[0]))
df_train = pd.merge(df_train, df_id_latlons, how='inner', on='ticket_id')
df_train.drop(['violation_street_number', 'violation_street_name'], axis = 1, 
              inplace = True)

#3 Get labels of objects for one-hot encoding
col_obj = df_train.dtypes[df_train.dtypes == 'object'].index.tolist()

# Convert object data into categories (fill less than threshold with 'unknown' cat)
def Categories(series):
    threshold = 100 # frequency of categories
    unknown_cat = '<unknown>' # name of additional 'unknown' category
    for series in series:
        count = df_train[series].value_counts()
        categories_to_keep = count[count > threshold].index.tolist()
        df_train[series] = pd.Categorical(df_train[series], 
                categories = categories_to_keep, ordered=True)
        df_train[series] = df_train[series].cat.add_categories(unknown_cat).fillna(unknown_cat)

Categories(col_obj) # Update dataframe with categorical data

#4 Call get_dummies
df_train = pd.get_dummies(df_train, columns = col_obj) # One hot encoding
```

</div>

<div class="cell markdown">

It's worth noting here that the threshold of categorical frequency was
taken 100. This was a design choice and has obvious consequences on the
test score. Simply reducing the threshold to allow for categories that
have less than 100 appearances will lead to basically empty feature sets
when get\_dummies is called. Therefore, this will only add computational
time and resources, all for a score that won't see much change.

However, one must take note on the number of categories present in the
feature. If the majority are equal or approximate to the threshold, then
the threshold should either not be used at all, or altered. *The idea*
is to take the majority of the categories that are in the feature, and
bin the remaining categories into one 'unknown' category.

</div>

<div class="cell markdown">

## Feature Engineering

Next, we will engineer a new feature by taking the difference between
ticket\_issued\_date and the hearing\_date. We update the new label with
the difference in days. Furthermore, we perform a last check on any
columns that contain NaN, and fill those NaN's with the respected mean
value of the column.

Note that we find the columns that contain NaN's to speed up the
fillna() process (as opposed to calculating the mean for every column).

</div>

<div class="cell code" data-execution_count="6">

``` python
## Feature engineering
# Remove the two date labels, and engineer a new feature.
# This feature will simply be the time between the hearing data and the 
# issue ticket date.
col_date = ['hearing_date', 'ticket_issued_date']
df_train['hearing_issued_date_diff'] = (df_train[col_date[0]]
        - df_train[col_date[1]]).dt.days
df_train.drop(col_date, axis=1, 
        inplace = True)

NaN_in_labels = df_train.columns[df_train.isnull().any()].tolist()
df_train.fillna(df_train[NaN_in_labels].mean(), inplace=True)
```

<div class="output execute_result" data-execution_count="6">

``` 
        ticket_id  fine_amount  admin_fee  state_fee  late_fee  \
0           22056        250.0       20.0       10.0      25.0   
1           27586        750.0       20.0       10.0      75.0   
2           22046        250.0       20.0       10.0      25.0   
3           18738        750.0       20.0       10.0      75.0   
4           18735        100.0       20.0       10.0      10.0   
5           18733        100.0       20.0       10.0      10.0   
6           28204        750.0       20.0       10.0      75.0   
7           18743        750.0       20.0       10.0      75.0   
8           18741        750.0       20.0       10.0      75.0   
9           18978        750.0       20.0       10.0      75.0   
10          18746        100.0       20.0       10.0      10.0   
11          18744        100.0       20.0       10.0      10.0   
12          26846        750.0       20.0       10.0      75.0   
13          26848        750.0       20.0       10.0      75.0   
14          28209        750.0       20.0       10.0      75.0   
15          19950        100.0       20.0       10.0       0.0   
16          18645        250.0       20.0       10.0      25.0   
17          18651        250.0       20.0       10.0      25.0   
18          18649        250.0       20.0       10.0      25.0   
19          18664        250.0       20.0       10.0      25.0   
20          18646        250.0       20.0       10.0      25.0   
21          18661        250.0       20.0       10.0      25.0   
22          18657        250.0       20.0       10.0      25.0   
23          18652        250.0       20.0       10.0      25.0   
24          18665        250.0       20.0       10.0      25.0   
25          18650        250.0       20.0       10.0      25.0   
26          18653        250.0       20.0       10.0       0.0   
27          18658        250.0       20.0       10.0       0.0   
28          18666        250.0       20.0       10.0      25.0   
29          18655        250.0       20.0       10.0       0.0   
...           ...          ...        ...        ...       ...   
159850     268081        750.0       20.0       10.0      75.0   
159851     267964        250.0       20.0       10.0      25.0   
159852     267966        250.0       20.0       10.0       0.0   
159853     267974        500.0       20.0       10.0      50.0   
159854     267985        250.0       20.0       10.0      25.0   
159855     267970        250.0       20.0       10.0      25.0   
159856     284870       1000.0       20.0       10.0     100.0   
159857     284873       1000.0       20.0       10.0     100.0   
159858     284871       1000.0       20.0       10.0     100.0   
159859     284875        100.0       20.0       10.0      10.0   
159860     284874        100.0       20.0       10.0      10.0   
159861     285091         50.0       20.0       10.0       5.0   
159862     285508          0.0       20.0       10.0       0.0   
159863     285093         50.0       20.0       10.0       5.0   
159864     285095         50.0       20.0       10.0       5.0   
159865     285121         50.0       20.0       10.0       5.0   
159866     285122         50.0       20.0       10.0       5.0   
159867     285120         50.0       20.0       10.0       5.0   
159868     285123         50.0       20.0       10.0       5.0   
159869     285092         50.0       20.0       10.0       5.0   
159870     285094         50.0       20.0       10.0       5.0   
159871     285096       1000.0       20.0       10.0     100.0   
159872     285036         50.0       20.0       10.0       5.0   
159873     285037        100.0       20.0       10.0      10.0   
159874     285034        500.0       20.0       10.0      50.0   
159875     285106        200.0       20.0       10.0      20.0   
159876     284650       1000.0       20.0       10.0     100.0   
159877     285125        500.0       20.0       10.0      50.0   
159878     284881        200.0       20.0       10.0       0.0   
159879     284333        200.0       20.0       10.0      20.0   

        discount_amount  clean_up_cost  judgment_amount  compliance  \
0                   0.0            0.0            305.0         0.0   
1                   0.0            0.0            855.0         1.0   
2                   0.0            0.0            305.0         0.0   
3                   0.0            0.0            855.0         0.0   
4                   0.0            0.0            140.0         0.0   
5                   0.0            0.0            140.0         0.0   
6                   0.0            0.0            855.0         0.0   
7                   0.0            0.0            855.0         0.0   
8                   0.0            0.0            855.0         0.0   
9                   0.0            0.0            855.0         0.0   
10                  0.0            0.0            140.0         1.0   
11                  0.0            0.0            140.0         1.0   
12                  0.0            0.0            855.0         0.0   
13                  0.0            0.0            855.0         0.0   
14                  0.0            0.0            855.0         0.0   
15                  0.0            0.0            130.0         0.0   
16                  0.0            0.0            305.0         0.0   
17                  0.0            0.0            305.0         0.0   
18                  0.0            0.0            305.0         0.0   
19                  0.0            0.0            305.0         0.0   
20                  0.0            0.0            305.0         0.0   
21                  0.0            0.0            305.0         0.0   
22                  0.0            0.0            305.0         1.0   
23                  0.0            0.0            305.0         0.0   
24                  0.0            0.0            305.0         0.0   
25                  0.0            0.0            305.0         0.0   
26                 25.0            0.0            280.0         1.0   
27                  0.0            0.0            280.0         0.0   
28                  0.0            0.0            305.0         0.0   
29                 25.0            0.0            280.0         1.0   
...                 ...            ...              ...         ...   
159850              0.0            0.0            855.0         0.0   
159851              0.0            0.0            305.0         0.0   
159852              0.0            0.0            280.0         0.0   
159853              0.0            0.0            580.0         0.0   
159854              0.0            0.0            305.0         0.0   
159855              0.0            0.0            305.0         0.0   
159856              0.0            0.0           1130.0         0.0   
159857              0.0            0.0           1130.0         0.0   
159858              0.0            0.0           1130.0         0.0   
159859              0.0            0.0            140.0         0.0   
159860              0.0            0.0            140.0         0.0   
159861              0.0            0.0             85.0         0.0   
159862              0.0            0.0              0.0         1.0   
159863              0.0            0.0             85.0         0.0   
159864              0.0            0.0             85.0         0.0   
159865              0.0            0.0             85.0         0.0   
159866              0.0            0.0             85.0         0.0   
159867              0.0            0.0             85.0         0.0   
159868              0.0            0.0             85.0         0.0   
159869              0.0            0.0             85.0         0.0   
159870              0.0            0.0             85.0         0.0   
159871              0.0            0.0           1130.0         0.0   
159872              0.0            0.0             85.0         0.0   
159873              0.0            0.0            140.0         0.0   
159874              0.0            0.0            580.0         0.0   
159875              0.0            0.0            250.0         0.0   
159876              0.0            0.0           1130.0         0.0   
159877              0.0            0.0            580.0         0.0   
159878              0.0            0.0            230.0         1.0   
159879              0.0            0.0            250.0         0.0   

              lat            ...             \
0       42.390729            ...              
1       42.326937            ...              
2       42.145257            ...              
3       42.433466            ...              
4       42.388641            ...              
5       42.388641            ...              
6       42.435773            ...              
7       42.395765            ...              
8       42.440190            ...              
9       42.399222            ...              
10      42.360836            ...              
11      42.360836            ...              
12      42.341729            ...              
13      42.341620            ...              
14      42.435592            ...              
15      42.385741            ...              
16      42.383385            ...              
17      42.389290            ...              
18      42.393440            ...              
19      42.335224            ...              
20      42.383422            ...              
21      42.335128            ...              
22      42.388282            ...              
23      42.374155            ...              
24      42.335913            ...              
25      42.389689            ...              
26      42.339391            ...              
27      42.371792            ...              
28      42.335846            ...              
29      42.382392            ...              
...           ...            ...              
159850  42.357540            ...              
159851  42.385938            ...              
159852  42.369046            ...              
159853  42.368735            ...              
159854  42.368735            ...              
159855  42.368735            ...              
159856  42.424312            ...              
159857  42.439131            ...              
159858  42.439143            ...              
159859  42.439131            ...              
159860  42.439131            ...              
159861  42.394788            ...              
159862  42.393397            ...              
159863  42.387270            ...              
159864  42.387270            ...              
159865  42.387270            ...              
159866  42.387270            ...              
159867  42.387270            ...              
159868  42.387270            ...              
159869  42.387270            ...              
159870  42.387270            ...              
159871  42.387270            ...              
159872  42.387270            ...              
159873  42.387270            ...              
159874  42.387270            ...              
159875  42.440228            ...              
159876  42.406293            ...              
159877  42.366529            ...              
159878  42.422081            ...              
159879  42.438867            ...              

        violation_description_Failure to maintain exterior of one- or two-family dwelling, building, premises or commercial structure in good repair, structurally sound or in a sanitary condition to prevent threat to the public health, safety or welfare  \
0                                                       0                                                                                                                                                                                                       
1                                                       0                                                                                                                                                                                                       
2                                                       0                                                                                                                                                                                                       
3                                                       0                                                                                                                                                                                                       
4                                                       0                                                                                                                                                                                                       
5                                                       0                                                                                                                                                                                                       
6                                                       0                                                                                                                                                                                                       
7                                                       0                                                                                                                                                                                                       
8                                                       0                                                                                                                                                                                                       
9                                                       0                                                                                                                                                                                                       
10                                                      0                                                                                                                                                                                                       
11                                                      0                                                                                                                                                                                                       
12                                                      0                                                                                                                                                                                                       
13                                                      0                                                                                                                                                                                                       
14                                                      0                                                                                                                                                                                                       
15                                                      0                                                                                                                                                                                                       
16                                                      0                                                                                                                                                                                                       
17                                                      0                                                                                                                                                                                                       
18                                                      0                                                                                                                                                                                                       
19                                                      0                                                                                                                                                                                                       
20                                                      0                                                                                                                                                                                                       
21                                                      0                                                                                                                                                                                                       
22                                                      0                                                                                                                                                                                                       
23                                                      0                                                                                                                                                                                                       
24                                                      0                                                                                                                                                                                                       
25                                                      0                                                                                                                                                                                                       
26                                                      0                                                                                                                                                                                                       
27                                                      0                                                                                                                                                                                                       
28                                                      0                                                                                                                                                                                                       
29                                                      0                                                                                                                                                                                                       
...                                                   ...                                                                                                                                                                                                       
159850                                                  0                                                                                                                                                                                                       
159851                                                  0                                                                                                                                                                                                       
159852                                                  0                                                                                                                                                                                                       
159853                                                  0                                                                                                                                                                                                       
159854                                                  0                                                                                                                                                                                                       
159855                                                  0                                                                                                                                                                                                       
159856                                                  0                                                                                                                                                                                                       
159857                                                  0                                                                                                                                                                                                       
159858                                                  0                                                                                                                                                                                                       
159859                                                  0                                                                                                                                                                                                       
159860                                                  0                                                                                                                                                                                                       
159861                                                  0                                                                                                                                                                                                       
159862                                                  0                                                                                                                                                                                                       
159863                                                  0                                                                                                                                                                                                       
159864                                                  0                                                                                                                                                                                                       
159865                                                  0                                                                                                                                                                                                       
159866                                                  0                                                                                                                                                                                                       
159867                                                  0                                                                                                                                                                                                       
159868                                                  0                                                                                                                                                                                                       
159869                                                  0                                                                                                                                                                                                       
159870                                                  0                                                                                                                                                                                                       
159871                                                  0                                                                                                                                                                                                       
159872                                                  0                                                                                                                                                                                                       
159873                                                  0                                                                                                                                                                                                       
159874                                                  0                                                                                                                                                                                                       
159875                                                  0                                                                                                                                                                                                       
159876                                                  0                                                                                                                                                                                                       
159877                                                  0                                                                                                                                                                                                       
159878                                                  0                                                                                                                                                                                                       
159879                                                  0                                                                                                                                                                                                       

        violation_description_Failing to secure City or Private solid waste collection containers and services  \
0                                                       0                                                        
1                                                       0                                                        
2                                                       0                                                        
3                                                       0                                                        
4                                                       0                                                        
5                                                       0                                                        
6                                                       0                                                        
7                                                       0                                                        
8                                                       0                                                        
9                                                       0                                                        
10                                                      0                                                        
11                                                      0                                                        
12                                                      0                                                        
13                                                      0                                                        
14                                                      0                                                        
15                                                      0                                                        
16                                                      0                                                        
17                                                      0                                                        
18                                                      0                                                        
19                                                      0                                                        
20                                                      0                                                        
21                                                      0                                                        
22                                                      0                                                        
23                                                      0                                                        
24                                                      0                                                        
25                                                      0                                                        
26                                                      0                                                        
27                                                      0                                                        
28                                                      0                                                        
29                                                      0                                                        
...                                                   ...                                                        
159850                                                  0                                                        
159851                                                  0                                                        
159852                                                  0                                                        
159853                                                  0                                                        
159854                                                  0                                                        
159855                                                  0                                                        
159856                                                  0                                                        
159857                                                  0                                                        
159858                                                  0                                                        
159859                                                  0                                                        
159860                                                  0                                                        
159861                                                  0                                                        
159862                                                  0                                                        
159863                                                  0                                                        
159864                                                  0                                                        
159865                                                  0                                                        
159866                                                  0                                                        
159867                                                  0                                                        
159868                                                  0                                                        
159869                                                  0                                                        
159870                                                  0                                                        
159871                                                  0                                                        
159872                                                  0                                                        
159873                                                  0                                                        
159874                                                  0                                                        
159875                                                  0                                                        
159876                                                  0                                                        
159877                                                  0                                                        
159878                                                  0                                                        
159879                                                  0                                                        

        violation_description_Defective exterior wall(s) one- or two-family dwelling or commercial building  \
0                                                       0                                                     
1                                                       0                                                     
2                                                       0                                                     
3                                                       0                                                     
4                                                       0                                                     
5                                                       0                                                     
6                                                       0                                                     
7                                                       0                                                     
8                                                       0                                                     
9                                                       0                                                     
10                                                      0                                                     
11                                                      0                                                     
12                                                      0                                                     
13                                                      0                                                     
14                                                      0                                                     
15                                                      0                                                     
16                                                      0                                                     
17                                                      0                                                     
18                                                      0                                                     
19                                                      0                                                     
20                                                      0                                                     
21                                                      0                                                     
22                                                      0                                                     
23                                                      0                                                     
24                                                      0                                                     
25                                                      0                                                     
26                                                      0                                                     
27                                                      0                                                     
28                                                      0                                                     
29                                                      0                                                     
...                                                   ...                                                     
159850                                                  0                                                     
159851                                                  0                                                     
159852                                                  0                                                     
159853                                                  0                                                     
159854                                                  0                                                     
159855                                                  0                                                     
159856                                                  0                                                     
159857                                                  0                                                     
159858                                                  0                                                     
159859                                                  0                                                     
159860                                                  0                                                     
159861                                                  0                                                     
159862                                                  0                                                     
159863                                                  0                                                     
159864                                                  0                                                     
159865                                                  0                                                     
159866                                                  0                                                     
159867                                                  0                                                     
159868                                                  0                                                     
159869                                                  0                                                     
159870                                                  0                                                     
159871                                                  0                                                     
159872                                                  0                                                     
159873                                                  0                                                     
159874                                                  0                                                     
159875                                                  0                                                     
159876                                                  0                                                     
159877                                                  0                                                     
159878                                                  0                                                     
159879                                                  0                                                     

        violation_description_<unknown>  disposition_Responsible by Default  \
0                                     0                                   1   
1                                     1                                   0   
2                                     0                                   1   
3                                     1                                   1   
4                                     1                                   1   
5                                     1                                   1   
6                                     1                                   1   
7                                     1                                   1   
8                                     1                                   1   
9                                     1                                   1   
10                                    1                                   0   
11                                    1                                   0   
12                                    1                                   1   
13                                    1                                   1   
14                                    1                                   1   
15                                    0                                   0   
16                                    0                                   1   
17                                    0                                   1   
18                                    0                                   1   
19                                    0                                   1   
20                                    0                                   1   
21                                    0                                   1   
22                                    0                                   0   
23                                    0                                   1   
24                                    0                                   1   
25                                    0                                   1   
26                                    0                                   0   
27                                    0                                   0   
28                                    0                                   1   
29                                    0                                   1   
...                                 ...                                 ...   
159850                                1                                   1   
159851                                0                                   1   
159852                                0                                   0   
159853                                0                                   1   
159854                                0                                   1   
159855                                0                                   1   
159856                                1                                   1   
159857                                1                                   1   
159858                                1                                   1   
159859                                0                                   1   
159860                                0                                   1   
159861                                0                                   1   
159862                                1                                   0   
159863                                0                                   1   
159864                                0                                   1   
159865                                0                                   1   
159866                                0                                   1   
159867                                0                                   1   
159868                                0                                   1   
159869                                0                                   1   
159870                                0                                   1   
159871                                0                                   1   
159872                                0                                   1   
159873                                0                                   1   
159874                                0                                   1   
159875                                0                                   1   
159876                                0                                   1   
159877                                0                                   1   
159878                                0                                   0   
159879                                0                                   1   

        disposition_Responsible by Admission  \
0                                          0   
1                                          0   
2                                          0   
3                                          0   
4                                          0   
5                                          0   
6                                          0   
7                                          0   
8                                          0   
9                                          0   
10                                         0   
11                                         0   
12                                         0   
13                                         0   
14                                         0   
15                                         1   
16                                         0   
17                                         0   
18                                         0   
19                                         0   
20                                         0   
21                                         0   
22                                         0   
23                                         0   
24                                         0   
25                                         0   
26                                         1   
27                                         1   
28                                         0   
29                                         0   
...                                      ...   
159850                                     0   
159851                                     0   
159852                                     1   
159853                                     0   
159854                                     0   
159855                                     0   
159856                                     0   
159857                                     0   
159858                                     0   
159859                                     0   
159860                                     0   
159861                                     0   
159862                                     0   
159863                                     0   
159864                                     0   
159865                                     0   
159866                                     0   
159867                                     0   
159868                                     0   
159869                                     0   
159870                                     0   
159871                                     0   
159872                                     0   
159873                                     0   
159874                                     0   
159875                                     0   
159876                                     0   
159877                                     0   
159878                                     0   
159879                                     0   

        disposition_Responsible by Determination  \
0                                              0   
1                                              1   
2                                              0   
3                                              0   
4                                              0   
5                                              0   
6                                              0   
7                                              0   
8                                              0   
9                                              0   
10                                             1   
11                                             1   
12                                             0   
13                                             0   
14                                             0   
15                                             0   
16                                             0   
17                                             0   
18                                             0   
19                                             0   
20                                             0   
21                                             0   
22                                             1   
23                                             0   
24                                             0   
25                                             0   
26                                             0   
27                                             0   
28                                             0   
29                                             0   
...                                          ...   
159850                                         0   
159851                                         0   
159852                                         0   
159853                                         0   
159854                                         0   
159855                                         0   
159856                                         0   
159857                                         0   
159858                                         0   
159859                                         0   
159860                                         0   
159861                                         0   
159862                                         0   
159863                                         0   
159864                                         0   
159865                                         0   
159866                                         0   
159867                                         0   
159868                                         0   
159869                                         0   
159870                                         0   
159871                                         0   
159872                                         0   
159873                                         0   
159874                                         0   
159875                                         0   
159876                                         0   
159877                                         0   
159878                                         1   
159879                                         0   

        disposition_Responsible (Fine Waived) by Deter  disposition_<unknown>  \
0                                                    0                      0   
1                                                    0                      0   
2                                                    0                      0   
3                                                    0                      0   
4                                                    0                      0   
5                                                    0                      0   
6                                                    0                      0   
7                                                    0                      0   
8                                                    0                      0   
9                                                    0                      0   
10                                                   0                      0   
11                                                   0                      0   
12                                                   0                      0   
13                                                   0                      0   
14                                                   0                      0   
15                                                   0                      0   
16                                                   0                      0   
17                                                   0                      0   
18                                                   0                      0   
19                                                   0                      0   
20                                                   0                      0   
21                                                   0                      0   
22                                                   0                      0   
23                                                   0                      0   
24                                                   0                      0   
25                                                   0                      0   
26                                                   0                      0   
27                                                   0                      0   
28                                                   0                      0   
29                                                   0                      0   
...                                                ...                    ...   
159850                                               0                      0   
159851                                               0                      0   
159852                                               0                      0   
159853                                               0                      0   
159854                                               0                      0   
159855                                               0                      0   
159856                                               0                      0   
159857                                               0                      0   
159858                                               0                      0   
159859                                               0                      0   
159860                                               0                      0   
159861                                               0                      0   
159862                                               1                      0   
159863                                               0                      0   
159864                                               0                      0   
159865                                               0                      0   
159866                                               0                      0   
159867                                               0                      0   
159868                                               0                      0   
159869                                               0                      0   
159870                                               0                      0   
159871                                               0                      0   
159872                                               0                      0   
159873                                               0                      0   
159874                                               0                      0   
159875                                               0                      0   
159876                                               0                      0   
159877                                               0                      0   
159878                                               0                      0   
159879                                               0                      0   

        hearing_issued_date_diff  
0                          369.0  
1                          378.0  
2                          323.0  
3                          253.0  
4                          251.0  
5                          251.0  
6                          323.0  
7                          209.0  
8                          201.0  
9                          189.0  
10                         138.0  
11                         138.0  
12                         190.0  
13                         189.0  
14                         215.0  
15                          40.0  
16                          24.0  
17                          30.0  
18                          30.0  
19                          68.0  
20                          31.0  
21                          35.0  
22                          35.0  
23                          30.0  
24                          35.0  
25                          30.0  
26                          39.0  
27                          21.0  
28                          35.0  
29                          30.0  
...                          ...  
159850                    -280.0  
159851                    -224.0  
159852                    -295.0  
159853                    -280.0  
159854                    -280.0  
159855                    -294.0  
159856                      13.0  
159857                      19.0  
159858                      19.0  
159859                      19.0  
159860                      19.0  
159861                      30.0  
159862                      44.0  
159863                      30.0  
159864                      30.0  
159865                      37.0  
159866                      37.0  
159867                      37.0  
159868                      37.0  
159869                      30.0  
159870                      30.0  
159871                      30.0  
159872                      29.0  
159873                      29.0  
159874                      29.0  
159875                      37.0  
159876                      10.0  
159877                      26.0  
159878                      40.0  
159879                       5.0  

[159880 rows x 88 columns]
```

</div>

</div>

<div class="cell markdown">

We can show that there are zero NaN values in our set, and that we are
ready to start training.

</div>

<div class="cell code" data-execution_count="10">

``` python
df_train.isnull().any().any()
```

<div class="output execute_result" data-execution_count="10">

    False

</div>

</div>

<div class="cell markdown">

## Begin training the model

We can define a function that performs the fit, caluclates the accuracy
and roc\_auc score of the training data alone, and then performs
cross-validation on the training set to return the averaged roc\_auc
score of the left-out data.

It's a good idea to make note of the accuracy and roc\_auc score of the
entire training set.

**Accuracy Score on training set**: This metric is helpful in evaluating
just how good the model can predict the label of the postitive class. Of
course, this metric alone is not a good indicator of the performance of
the model, as this metric assumes that any probability above 0.5 should
be labeled the positive class. This alone neglects the possibility for
skewedness in the target values, but is useful in evauluating
variance/bias.

**ROC\_AUC Score on training set**: Looking at the roc\_auc score, in
conjunction with accuracy, can help define how well the model is
performing. The 'auc score' basically takes into account the skewedness
of the data by evaulating over many thresholds (not just 0.5) when
classifying the data. The area under the roc curve (roc\_auc) is
essentually a number that explains how much, over a range of thresholds,
does the true positives reflect the false positives. A higher roc\_auc
means that more of the threshold range corresponds to a high true
positive rate.

\*\* ROC\_AUC Score on cross-validation set\*\*: Looking at just the
training data is again not a good practice, as one does not know if
there is a bias/variance problem of the model. One must analyze the
hold-out set from the CV results to ensure that the model is
generalizing well enough.

Of course, using the CV score isn't enough either, as we'll have to
ensure that unseen, 'new' data still generalizes well to the model (i.e.
test.csv).

</div>

<div class="cell code" data-execution_count="291" data-collapsed="true">

``` python
def modelCV(alg, dtrain, predictors, performCV=True, cv_folds=5):      
    # Predict training set
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
    # Perform cross-validation
    if performCV:
        cv_score = cross_val_score(alg, dtrain[predictors], 
                                                    dtrain[target], 
                                                    cv=cv_folds, 
                                                    scoring='roc_auc')
    
    # Print model report
    print("\nModel Report")
    print("Accuracy : %.4g" % accuracy_score(dtrain[target].values, 
          dtrain_predictions))
    print("AUC Score (Train): %f" % roc_auc_score(dtrain[target], 
          dtrain_predprob))
    
    if performCV:
        print("AUC_CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" 
              % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),
                 np.max(cv_score)))
        
def plotFeatureImp(alg, predictors):
    #Plot feature importance
    feature = 10
    feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)[:feature]
    ax = feat_imp.plot(kind = 'barh', title="Feature Importance [Top 10]")
    plt.xlabel('Feature Importance Score')
    plt.box(on=None)
    plt.tight_layout()
    
def plotROCAUC(alg, dtrain):
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    fpr, tpr, _ = roc_curve(dtrain[target], dtrain_predprob)
    roc_auc = roc_auc_score(dtrain[target], dtrain_predprob)
    
    plt.figure()
    plt.xlim([-0.01, 1.00])
    plt.ylim([-0.01, 1.01])
    plt.plot(fpr, tpr, lw=3, label='ROC curve (area = {:0.2f})'.format(roc_auc))
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC curve (Binary Classifier)', fontsize=16)
    plt.legend(loc='lower right', fontsize=13)
    plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    plt.axes().set_aspect('equal')
```

</div>

<div class="cell markdown">

Let's identify our id and target column, get our features that we'd like
to include, identify our classifier, fit, then call the modelCV function
to give us a modest 'model report'.

</div>

<div class="cell markdown">

### Gradient Boosting Classifier

</div>

<div class="cell code" data-execution_count="33">

``` python
#Choose all predictors except target & IDcols 
target = 'compliance'
IDcol = 'ticket_id'
predictors = [x for x in df_train.columns if x not in [target, IDcol]]
gbm0 = GradientBoostingClassifier(random_state=10)
gbm0.fit(df_train[predictors], df_train[target])

modelCV(gbm0, df_train, predictors)
```

<div class="output stream stdout">

``` 

Model Report
Accuracy : 0.944
AUC Score (Train): 0.819741
AUC_CV Score : Mean - 0.8023197 | Std - 0.02520005 | Min - 0.7763131 | Max - 0.8469616
```

</div>

</div>

<div class="cell code" data-execution_count="292">

``` python
#Plot feature importance
plt.style.use('seaborn-pastel')
plotFeatureImp(gbm0, predictors)
```

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

<div class="output display_data">

    <IPython.core.display.HTML object>

</div>

</div>

<div class="cell markdown" data-collapsed="true">

The above score for a default Gradient Boosting Classifier was about
0.8023 (cv mean test score). We can also note the 10 most important
features that contributed to the predictions made by the classifier in
Figure 1. We can futher optimize and tune the model by performing a grid
search over a range of params (GridSearchCV). We perform a grid search
and vary the 'n\_estimators' parameter for GBC.

</div>

<div class="cell code" data-execution_count="20">

``` python
param_test1 = {'n_estimators': np.arange(20, 41, 10).tolist()}
gsearch1 = GridSearchCV(GradientBoostingClassifier(learning_rate=0.05,min_samples_split=500,
                                                   min_samples_leaf=50,max_depth=8,
                                                   max_features='sqrt',subsample=0.8,random_state=10), 
                        param_grid = param_test1, scoring='roc_auc', iid=False, cv=5)

gsearch1.fit(df_train[predictors],df_train[target])
```

<div class="output execute_result" data-execution_count="20">

    GridSearchCV(cv=5, error_score='raise',
           estimator=GradientBoostingClassifier(criterion='friedman_mse', init=None,
                  learning_rate=0.05, loss='deviance', max_depth=8,
                  max_features='sqrt', max_leaf_nodes=None,
                  min_impurity_decrease=0.0, min_impurity_split=None,
                  min_samples_leaf=50, min_samples_split=500,
                  min_weight_fraction_leaf=0.0, n_estimators=100,
                  presort='auto', random_state=10, subsample=0.8, verbose=0,
                  warm_start=False),
           fit_params=None, iid=False, n_jobs=1,
           param_grid={'n_estimators': [20, 30, 40]}, pre_dispatch='2*n_jobs',
           refit=True, return_train_score=True, scoring='roc_auc', verbose=0)

</div>

</div>

<div class="cell markdown">

We can now call the gridsearch object and print the best parameter (in
this case, just n\_estimator) and the highest mean roc\_auc score on the
cv test set.

</div>

<div class="cell code" data-execution_count="21">

``` python
gsearch1.best_params_, gsearch1.best_score_
```

<div class="output execute_result" data-execution_count="21">

    ({'n_estimators': 20}, 0.80566598467260619)

</div>

</div>

<div class="cell markdown">

Two things to note here. The n\_estimators parameter reached the extreme
(20) on the range chosen (20, 30, 40). We should increase the lower end
of the param range to see if the parameter lies below the current
extreme. Also, the test score increased from 0.8023 to 0.8057. Let's
continue to optimize by first extending the range for n\_estimators.

</div>

<div class="cell code" data-execution_count="22">

``` python
param_test1 = {'n_estimators': np.arange(10, 31, 10).tolist()}
gsearch1 = GridSearchCV(GradientBoostingClassifier(learning_rate=0.05,min_samples_split=500,
                                                   min_samples_leaf=50,max_depth=8,
                                                   max_features='sqrt',subsample=0.8,random_state=10), 
                        param_grid = param_test1, scoring='roc_auc', iid=False, cv=5)

gsearch1.fit(df_train[predictors],df_train[target])
```

<div class="output execute_result" data-execution_count="22">

    GridSearchCV(cv=5, error_score='raise',
           estimator=GradientBoostingClassifier(criterion='friedman_mse', init=None,
                  learning_rate=0.05, loss='deviance', max_depth=8,
                  max_features='sqrt', max_leaf_nodes=None,
                  min_impurity_decrease=0.0, min_impurity_split=None,
                  min_samples_leaf=50, min_samples_split=500,
                  min_weight_fraction_leaf=0.0, n_estimators=100,
                  presort='auto', random_state=10, subsample=0.8, verbose=0,
                  warm_start=False),
           fit_params=None, iid=False, n_jobs=1,
           param_grid={'n_estimators': [10, 20, 30]}, pre_dispatch='2*n_jobs',
           refit=True, return_train_score=True, scoring='roc_auc', verbose=0)

</div>

</div>

<div class="cell code" data-execution_count="23">

``` python
gsearch1.best_params_, gsearch1.best_score_
```

<div class="output execute_result" data-execution_count="23">

    ({'n_estimators': 20}, 0.80566598467260619)

</div>

</div>

<div class="cell markdown">

We can see that lowering the range does not help, as the best value for
n\_estimators is set at 20. Let's perform another grid search over the
follow:

    max_depth [5-15 at int of 2]
    min_samples_split [50-200 at int of 50]

We'll make sure to update n\_estimators to 20 from the previous
gridsearch (gsearch1).

</div>

<div class="cell code" data-execution_count="192">

``` python
param_test2 = {'max_depth': np.arange(5,11,1).tolist(), 'min_samples_split': np.arange(100,1501,100).tolist()}

gsearch2 = GridSearchCV(GradientBoostingClassifier(learning_rate=0.05, n_estimators=20, 
                                                   max_features='sqrt', subsample=0.8, 
                                                   random_state=10), 
                        param_grid = param_test2, scoring='roc_auc', iid=False, cv=5)

gsearch2.fit(df_train[predictors],df_train[target])
gsearch2.best_params_, gsearch2.best_score_
```

<div class="output execute_result" data-execution_count="192">

    ({'max_depth': 8, 'min_samples_split': 900}, 0.80559233621530102)

</div>

</div>

<div class="cell markdown">

We can see that the auc score generated here is lower than the previous.
It however can be assumed that gridsearching over other parameters
besides 'max\_depth' and 'min\_samples\_split' will lead to a higher
score. Increasing the number of trees ('n\_estimators') and lowering the
'learning\_rate' can lead to significant gains in score.

Once the fine tuning is complete, the results can be analyed against the
unseen test set. The test set will have to be processed in the same
manner as the training set. One must be careful not to remove samples
from the test set when dealing with NaN's, as samples of a test set
should be treated as real world samples that hold significance. Of
course, we're expecing the test score to be slightly below the finalized
cv score.

Let's run our 'final' model of gradient boosting with:

    n_estimators: 20
    min_samples_split: 500
    max_depth: 8

Then, let's plot our feature importances.

</div>

<div class="cell code" data-execution_count="300">

``` python
gbm2 = GradientBoostingClassifier(learning_rate=0.05,n_estimators = 20, 
                                               min_samples_split=500, min_samples_leaf=50,
                                               max_depth=8, max_features='sqrt',
                                               subsample=0.8, random_state=10)

gbm2.fit(df_train[predictors], df_train[target])
modelCV(gbm2, df_train, predictors)
#Plot feature importance
plt.style.use('seaborn-pastel')
plotFeatureImp(gbm2, predictors)
```

<div class="output stream stdout">

``` 

Model Report
Accuracy : 0.9362
AUC Score (Train): 0.815307
AUC_CV Score : Mean - 0.805666 | Std - 0.02358854 | Min - 0.7823296 | Max - 0.8497655
```

</div>

<div class="output display_data">

    <IPython.core.display.Javascript object>

</div>

<div class="output display_data">

    <IPython.core.display.HTML object>

</div>

</div>

<div class="cell markdown">

One can note that the feature late\_fee is still the most important,
followed by several categories for the original 'disposition' feature.

</div>

<div class="cell markdown">

### Logistic Regression

Let's model using logistic regression. We're expecting quicker results,
but not as much accuracy as we'd like. Furthermore, we're going to only
transform the data once for scaling purposes (note that this should be
done in a pipeline with each change in cv fold).

</div>

<div class="cell code" data-execution_count="185">

``` python
lr0 = LogisticRegression(random_state=10)

# Scale the data from 0 to 1.0
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(df_train) # transform df_train values. Result is array
df_train_scaled = pd.DataFrame(train_scaled, columns=df_train.columns) # put back into df with correct columns

# Now fit the data
lr0.fit(df_train_scaled[predictors], df_train_scaled[target])

modelCV(lr0, df_train_scaled, predictors)
```

<div class="output stream stdout">

``` 

Model Report
Accuracy : 0.9326
AUC Score (Train): 0.794789
AUC_CV Score : Mean - 0.7855392 | Std - 0.02741742 | Min - 0.7552199 | Max - 0.8372791
```

</div>

</div>

<div class="cell markdown">

Running a logistic regression classifier with the default params, we can
obtain a cv auc\_score of 0.7855. This is lower than the first go-around
with GBC (0.8023). Also, the accuracy of 0.9326 is slightly lower than
GBC's to 0.9440. Unlike GBC, we can adjust the regularization parameter
and see what we can produce.

</div>

<div class="cell code" data-execution_count="189">

``` python
param_test1 = {'C': np.arange(30,201,10).tolist()}
lrsearch1 = GridSearchCV(LogisticRegression(random_state=10), 
                        param_grid = param_test1, scoring='roc_auc',
                        iid=False, cv=5)
lrsearch1.fit(df_train_scaled[predictors],df_train_scaled[target])
lrsearch1.best_params_, lrsearch1.best_score_
```

<div class="output execute_result" data-execution_count="189">

    ({'C': 40}, 0.78715651694138755)

</div>

</div>

<div class="cell markdown">

We can see that a C value of 40 optimizes the model a little more, with
an auc score of 0.7872. For most cases, optimizing C (or trying
different penalty scenarios with L1 and L2 against C) is enough for an
LR classifier. Note that even after optimizing for regularization, the
auc score is still below GBC. We will not continue to optimize this
classifier as there was little gain with change in C vs the auc score
being produced with GBC.

</div>
