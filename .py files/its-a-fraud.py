
import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import math
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import warnings
warnings.simplefilter("ignore")
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

##################### Importing the data #############################
train_df = pd.read_csv('../input/its-a-fraud/train.csv')
test_df = pd.read_csv('../input/its-a-fraud/test.csv')
print("size of training dataset:" + str(train_df.shape))
print("size of testing dataset:" + str(test_df.shape))

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

##################### handling the columns having too many null values #############################
nan1 = train_df.isna().sum()
nans = []
for a,b in nan1.items():
    if(b<10000):
        nans.append(a)
len(nans)


train_df = train_df.dropna(subset=nans)

y_train = train_df.iloc[:,1]
train_df = train_df.drop(['isFraud'], axis = 1)

bignans = []
bignans = train_df.columns[train_df.isnull().mean() > 0.90]

##################### Analysing the y column #############################
y_train.value_counts()

print("Percentage of a transaction not being fraudulent: " + str((427408/(427408 + 15497))*100) + "%")
print("Percentage of a transaction being fraudulent: " + str((15497/(427408 + 15497))*100) + "%")

sns.countplot(y_train)
plt.show()

##################### Undersampling and oversampling #############################
over = RandomOverSampler(sampling_strategy=0.5)
under = RandomUnderSampler(sampling_strategy=0.8)

# first performing oversampling to minority class
X_over, y_over = over.fit_resample(train_df, y_train)
# now to comine under sampling 

x_under, y_under = under.fit_resample(X_over, y_over)

x_under.shape

y_train = y_under
train_df = x_under

sns.countplot(y_train)
plt.show()

df = train_df.append(test_df)

df = df[df.columns[df.isnull().mean() < 0.90]]

##################### Analysing card columns #############################
cards = df[['card1','card2','card3','card4','card5','card6']]

df['card6'].fillna(df['card6'].mode()[0], inplace = True )
df['card4'].fillna(df['card4'].mode()[0], inplace = True )

my_imputer = SimpleImputer()

df[['card1', 'card2', 'card3','card5']] = my_imputer.fit_transform(df[['card1', 'card2', 'card3','card5']])
corr = cards.corr()

plt.figure(figsize = (12,12))
ax = sns.heatmap(corr, annot = True, cmap = 'mako')

##################### Analysing the address columns #############################
corr1 = df[['addr1', 'addr2']].corr()
plt.figure(figsize = (12,12))
ax = sns.heatmap(corr1, annot = True, cmap = 'mako')

df[['addr1', 'addr2']].describe()

df[['addr1', 'addr2']] = my_imputer.fit_transform(df[['addr1','addr2']])

df[['addr1', 'addr2']].isna().sum()

##################### Analysing the dist1 column #############################
df[['dist1']].describe()

df[['dist1']] = my_imputer.fit_transform(df[['dist1']])

##################### Analysing the d columns #############################
d_cols = [col for col in df.columns if col[0] == 'D' and len(col) <=3]
df[d_cols].head(10)

corr2 = df[d_cols].corr()
plt.figure(figsize = (12,12))
ax = sns.heatmap(corr2, annot = True)

del df['D6']
del df['D12']
del df['D2']

df.shape

df[['D1', 'D3','D4','D5','D8','D9','D10','D11','D13','D14','D15']].isna().sum()

df[['D3','D4','D5','D8','D9','D10','D11', 'D13','D14','D15']].describe()

df[['D3','D4','D5','D8','D9','D10','D11','D13','D14','D15']].mode()

df[['D3','D4','D5','D8','D9','D10','D11','D13','D14','D15']].median()

df[['D1','D3','D4','D5','D8','D9','D10','D11','D13','D14','D15']] = my_imputer.fit_transform(df[['D1','D3','D4','D5','D8','D9','D10','D11','D13','D14','D15']])

df[['D1','D3','D4','D5','D8','D9','D10','D11','D13','D14','D15']].isna().sum()

##################### Making batches, so later it would be easier to analyze #############################
c_cols = [col for col in train_df.columns if col[0] == 'C' and len(col) <=3]
m_cols = [col for col in df.columns if col[0] == 'M' and len(col) <=2]
v_cols = [col for col in df.columns if col[0] == 'V' and len(col) <=4]
id_cols = [col for col in df.columns if col[0] == 'i' and len(col) <=5]

##################### Analysing the M columns #############################
df[m_cols].isna().sum()

df['M1'].fillna(df['M1'].mode()[0], inplace=True)
df['M2'].fillna(df['M2'].mode()[0], inplace=True)
df['M3'].fillna(df['M3'].mode()[0], inplace=True)
df['M4'].fillna(df['M4'].mode()[0], inplace=True)
df['M5'].fillna(df['M5'].mode()[0], inplace=True)
df['M6'].fillna(df['M6'].mode()[0], inplace=True)
df['M7'].fillna(df['M7'].mode()[0], inplace=True)
df['M8'].fillna(df['M8'].mode()[0], inplace=True)
df['M9'].fillna(df['M9'].mode()[0], inplace=True)

##################### Analysing the V columns #############################
df[v_cols].head(10)

nan_dict = {}
for col in v_cols:
    count = df[col].isnull().sum()
    try:
        nan_dict[count].append(col)
    except:
        nan_dict[count] = [col]   
for k,v in nan_dict.items():
    print("")
    print(f'NAN count = {k} percent: {(int(k)/df.shape[0])*100} %')
    print(v)    
print("")
print("Number of groups formed in v columns:")
print(len(nan_dict))

def reduction(grps):
    use = []
    for col in grps:
        max_unique = 0
        max_index = 0
        for i,c in enumerate(col):
            n = df[c].nunique()
            if n > max_unique:
                max_unique = n
                max_index = i
        use.append(col[max_index])
    return use

############ Group 1
g1 = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11']
corr = df[g1].corr()
plt.figure(figsize = (24,24))
ax = sns.heatmap(corr, annot = True)

############
pairs = [['V1'],['V2','V3'],['V4','V5'],['V6','V7'],['V8','V9'],['V10','V11']]
g1 = reduction(pairs)
g1

############ Group 2
g2 = ['V21', 'V22', 'V23', 'V34', 'V33', 'V32','V31', 'V30', 'V29', 'V28', 'V27', 'V25', 'V24', 'V26', 'V16', 'V15', 
      'V20', 'V14', 'V19', 'V18', 'V17', 'V12', 'V13']
corr = df[g2].corr()
plt.figure(figsize = (24,24))
ax = sns.heatmap(corr, annot = True)

############ 
pairs = [['V12','V13'],['V14'],['V15','V16','V17','V18','V21','V22','V31','V32','V33','V34'],['V19','V20'],
         ['V23','V24'],['V25','V26'],['V27','V28'],['V29','V30']]

g2 = reduction(pairs)
g2

############ Group 3
g3 = ['V35', 'V40', 'V41', 'V39', 'V38', 'V51', 'V37', 'V52', 'V36', 'V50', 'V48', 'V42',
 'V43', 'V44', 'V46', 'V47', 'V45', 'V49']

corr = df[g3].corr()
plt.figure(figsize = (24,24))
ax = sns.heatmap(corr, annot = True)

############
pairs = [['V35','V36'],['V37','V38'],['V39','V40','V42','V43','V50','V51','V52'],['V41'],
         ['V44','V45'],['V46','V47'],['V48','V49']]

g3 = reduction(pairs)
g3

############ Group 4
g4 = ['V72', 'V74', 'V73', 'V71', 'V65', 'V68', 'V58', 'V70', 'V53', 
 'V54', 'V55', 'V56', 'V57', 'V59', 'V67', 'V60', 'V61',
 'V62', 'V63', 'V64', 'V66', 'V69']

corr = df[g4].corr()
plt.figure(figsize = (24,24))
ax = sns.heatmap(corr, annot = True)

############
pairs = [['V53','V54'],['V55'],['V56'],['V57', 'V58', 'V59', 'V60', 'V63', 'V64', 'V71', 'V72', 'V73', 'V74'],['V61','V62'],
 ['V65'],['V66','V67'],['V68'],['V69','V70']]

g4 = reduction(pairs)
g4

############ Group 5
g5 = ['V80', 'V87', 'V88', 'V89', 'V90', 'V91', 'V92', 'V93', 'V94', 'V86', 'V79',
 'V85', 'V75', 'V84', 'V77', 'V83', 'V78', 'V82', 'V81', 'V76']

corr = df[g5].corr()
plt.figure(figsize = (24,24))
ax = sns.heatmap(corr, annot = True)

############
pairs = [['V75','V76'],['V77','V78'],['V79','V80','V81','V84','V85','V92','V93','V94'],['V82','V83'],['V86','V87'],
         ['V88'],['V89'],['V90','V91']]

g5 = reduction(pairs)
g5

############ Group 6
g6 = ['V104', 'V109', 'V110', 'V111', 'V112', 'V106', 'V105', 'V102', 'V103', 'V96', 'V101', 'V100',
        'V99', 'V98', 'V97', 'V95', 'V135', 'V134', 'V107', 'V133', 'V132', 'V131', 'V130', 'V129', 
        'V128', 'V127', 'V126', 'V125', 'V124', 'V123', 'V122', 'V121', 
        'V120', 'V119', 'V118', 'V117', 'V116', 'V115', 'V114', 'V113', 'V136', 'V137', 'V108']

corr = df[g6].corr()
plt.figure(figsize = (24,24))
ax = sns.heatmap(corr, annot = True)

############ Group 6.1
g6_1 = ['V95', 'V96', 'V97', 'V98', 'V99', 'V100', 'V101', 'V102', 'V103', 'V104', 'V105', 'V106']

corr = df[g6_1].corr()
plt.figure(figsize = (24,24))
ax = sns.heatmap(corr, annot = True)

############
pairs = [['V99','V100'],['V98'],['V104'],['V95','V96','V97','V101','V102','V103','V105','V106']]

g6_1 = reduction(pairs)
g6_1

############ Group 6.2
g6_2 = ['V107', 'V108', 'V109', 'V110', 'V111', 'V112', 'V113', 'V114', 'V115', 'V116', 'V117', 'V118', 'V119', 'V120', 'V121', 'V122', 'V123']

corr = df[g6_2].corr()
plt.figure(figsize = (24,24))
ax = sns.heatmap(corr, annot = True)

############
pairs = [['V107'],['V108','V109','V110','V114'],['V111','V112','V113'],['V115','V116'],['V117','V118','V119'],['V120','V122'],['V121'],['V123']]

g6_2 = reduction(pairs)
g6_2

############ Group 6.3
g6_3 = ['V124', 'V125', 'V126', 'V127', 'V128', 'V129', 'V130', 'V131', 'V132', 'V133', 'V134', 'V135', 'V136', 'V137']

corr = df[g6_3].corr()
plt.figure(figsize = (24,24))
ax = sns.heatmap(corr, annot = True)

############
pairs = [['V124','V125'],['V126','V127','V128','V132','V133','V134'],['V129'],['V130','V131'],['V135','V136','V137']]

g6_3 = reduction(pairs)
g6_3

############ Group 7
g7 = ['V142', 'V158', 'V140', 'V162', 'V141', 'V161', 'V157', 'V146', 'V156', 'V155', 'V154',
        'V153', 'V149', 'V147', 'V148', 'V163', 'V139', 'V138']

corr = df[g7].corr()
plt.figure(figsize = (24,24))
ax = sns.heatmap(corr, annot = True)

############
pairs = [['V138'],['V139','V140'],['V141','V142'],['V146','V147'],['V148','V149','V153','V154','V156','V157','V158'],['V161','V162','V163']]

g7 = reduction(pairs)
g7

############ Group 8
g8 = ['V160', 'V151', 'V152', 'V145', 'V144', 'V143', 'V159', 'V164', 'V165', 'V166', 'V150']

corr = df[g8].corr()
plt.figure(figsize = (24,24))
ax = sns.heatmap(corr, annot = True)

############ 
pairs = [['V143','V164','V165'],['V144','V145','V150','V151','V152','V159','V160'],['V166']]

g8 = reduction(pairs)
g8

############ Group 9
g9 = ['V167', 'V168', 'V172', 'V173', 'V176', 'V177', 'V178', 'V179', 'V181', 'V182', 'V183', 'V186', 
      'V187', 'V190', 'V191', 'V192', 'V193', 'V196', 'V199', 'V202', 'V203', 'V204', 'V205', 'V206', 
      'V207', 'V211', 'V212', 'V213', 'V214', 'V215', 'V216']

corr = df[g9].corr()
plt.figure(figsize = (24,24))
ax = sns.heatmap(corr, annot = True)

g9_1 = ['V167', 'V168','V172','V173','V176','V177','V178','V179','V181','V182','V183']

############
corr = df[g9_1].corr()
plt.figure(figsize = (24,24))
ax = sns.heatmap(corr, annot = True)

############ Group 9.1  
pairs = [['V167','V168','V177','V178','V179'],['V172','V176'],['V173'],['V181','V182','V183']]

g9_1 = reduction(pairs)
g9_1

############ Group 9.2
g9_2 = ['V186','V187','V190','V191','V192','V193','V196','V199','V202','V203','V204','V211','V212','V213','V205','V206','V207','V214','V215','V216']

corr = df[g9_2].corr()
plt.figure(figsize = (24,24))
ax = sns.heatmap(corr, annot = True)

############
pairs = [['V186','V187','V190','V191','V192','V193','V196','V199'],['V202','V203','V204','V211','V212','V213'],
         ['V205','V206'],['V207'],['V214','V215','V216']]

g9_2 = reduction(pairs)
g9_2

############ Group 10
g10 = ['V194', 'V200', 'V189', 'V188', 'V185', 'V184', 'V180', 'V175', 'V174', 'V171',
        'V170', 'V169', 'V195', 'V201', 'V197', 'V198', 'V209', 'V208', 'V210']

corr = df[g10].corr()
plt.figure(figsize = (24,24))
ax = sns.heatmap(corr, annot = True)

############
pairs = [['V169'],['V170','V171','V200','V201'],['V174','V175'],['V180'],['V184','V185'],['V188','V189'],['V194','V195','V197','V198'],['V208','V210'],
         ['V209']]

g10 = reduction(pairs)
g10

############ Group 11
g11 = ['V217', 'V218', 'V219', 'V223', 'V224', 'V225', 'V226', 'V228', 'V229', 'V230', 'V231',
       'V232', 'V233', 'V235', 'V236', 'V237', 'V240', 'V241', 'V242', 'V243', 'V244', 'V246', 
       'V247', 'V248', 'V249', 'V252', 'V253', 'V254', 'V257', 'V258', 'V260', 'V261', 'V262', 'V263', 'V264', 
       'V265', 'V266', 'V267', 'V268', 'V269', 'V273', 'V274', 'V275', 'V276', 'V277', 'V278']

corr = df[g11].corr()
plt.figure(figsize = (24,24))
ax = sns.heatmap(corr, annot = True)

############ Group 11.1
g11_1 = ['V217','V218','V219','V231','V232','V233','V236','V237','V223','V224','V225','V226','V228','V229','V230','V235']

corr = df[g11_1].corr()
plt.figure(figsize = (24,24))
ax = sns.heatmap(corr, annot = True)

############
pairs = [['V217','V218','V219','V231','V232','V233','V236','V237'],['V223'],['V224','V225'],['V226'],['V228'],['V229','V230'],['V235']]

g11_1 = reduction(pairs)
g11_1

############ Group 11.2
g11_2 = ['V240','V241','V242','V243','V244','V258','V246','V257','V247','V248','V249','V253','V254','V252','V260','V261','V262']

corr = df[g11_2].corr()
plt.figure(figsize = (24,24))
ax = sns.heatmap(corr, annot = True)

############
pairs = [['V240','V241'],['V242','V243','V244','V258'],['V246','V257'],['V247','V248','V249','V253','V254'],['V252'],['V260'],['V261','V262']]

g11_2 = reduction(pairs)
g11_2

############ Group 11.3
g11_3 =  ['V263','V265','V264','V266','V269','V267','V268','V273','V274','V275','V276','V277','V278']

corr = df[g11_3].corr()
plt.figure(figsize = (24,24))
ax = sns.heatmap(corr, annot = True)

############
pairs =[['V263','V265','V264'],['V266','V269'],['V267','V268'],['V273','V274','V275'],['V276','V277','V278']]

g11_3 = reduction(pairs)
g11_3

############ Group 12
g12 = ['V245', 'V271', 'V234', 'V222', 'V238', 'V239', 'V227', 'V250','V272', 'V270', 'V251', 'V220', 'V255', 'V256', 'V259', 'V221']

corr = df[g12].corr()
plt.figure(figsize = (24,24))
ax = sns.heatmap(corr, annot = True)

############
pairs = [['V220'],['V221','V222','V227','V245','V255','V256','V259'],['V234'],['V238','V239'],['V250','V251'],['V270','V271','V272']]

g12 = reduction(pairs)
g12

############ Group 13
g13 = ['V311', 'V321', 'V294', 'V306', 'V305', 'V304', 'V303', 'V302', 'V299', 'V298', 'V297', 'V295', 
        'V293', 'V308', 'V292', 'V291', 'V290', 'V287', 'V286', 'V285', 'V284', 'V280', 'V279', 
        'V320', 'V307', 'V309', 'V312', 'V316', 'V317', 'V318', 'V319', 'V310']

corr = df[g13].corr()
plt.figure(figsize = (24,24))
ax = sns.heatmap(corr, annot = True)

############ Group 13.1
g13_1 = ['V279','V280','V293','V294','V295','V298','V299','V284','V285','V287','V286','V290','V291','V292','V297']

corr = df[g13_1].corr()
plt.figure(figsize = (24,24))
ax = sns.heatmap(corr, annot = True)

############
pairs = [['V279','V280','V293','V294','V295','V298','V299'],['V284'],['V285','V287'],['V286'],['V290','V291','V292'],['V297']]

g13_1 = reduction(pairs)
g13_1

############ Group 13.2
g13_2 = ['V302','V303','V304','V305','V306','V307','V308','V316','V317','V318','V309','V311','V310','V312','V319','V320','V321']

corr = df[g13_2].corr()
plt.figure(figsize = (24,24))
ax = sns.heatmap(corr, annot = True)

############
pairs = [['V302','V303','V304'],['V305'],['V306','V307','V308','V316','V317','V318'],['V309','V311'],['V310','V312'],['V319','V320','V321']]

g13_2 = reduction(pairs)
g13_2

############ Group 14
g14 = ['V296', 'V289', 'V288', 'V283', 'V282', 'V281', 'V300', 'V301', 'V313', 'V314', 'V315']

corr = df[g14].corr()
plt.figure(figsize = (24,24))
ax = sns.heatmap(corr, annot = True)

############
pairs = [['V281'],['V282','V283'],['V288','V289'],['V296'],['V300','V301'],['V313','V314','V315']]

g14 = reduction(pairs)
g14

############ Group 15
g15 = ['V337', 'V333', 'V336', 'V335', 'V334', 'V338', 'V339', 'V324', 'V332', 'V325', 'V330', 'V329', 'V328', 'V327', 'V326', 'V322', 'V323', 'V331']

corr = df[g15].corr()
plt.figure(figsize = (24,24))
ax = sns.heatmap(corr, annot = True)

############
pairs = [['V322','V323','V324','V326','V327','V328','V329','V330','V331','V332','V333'],['V325'],['V334','V335','V336'],['V337','V338','V339']]

g15 = reduction(pairs)
g15

##################### Final reduced V cols #############################
reduced_V_cols = ['V1', 'V3', 'V4', 'V6', 'V8', 'V11', 'V13', 'V14', 'V17', 'V20', 
 'V23', 'V26', 'V27', 'V30', 'V36', 'V37', 'V40', 'V41', 'V44', 'V47', 'V48', 'V54', 'V56', 'V59', 
 'V62', 'V65', 'V67', 'V68', 'V70', 'V76', 'V78', 'V80', 'V82', 'V86', 'V88', 'V89', 'V91', 'V96', 
 'V98', 'V99', 'V104', 'V107', 'V108', 'V111', 'V115', 'V117', 'V120', 'V121', 'V123', 'V124', 'V127', 
 'V129', 'V130', 'V136', 'V138', 'V139', 'V142', 'V147', 'V156', 'V162', 'V165', 'V160', 'V166', 'V178',
 'V176', 'V173', 'V182', 'V187', 'V203', 'V205', 'V207', 'V215', 'V169', 'V171', 'V175', 'V180', 'V185', 
 'V188', 'V198', 'V210', 'V209', 'V218', 'V223', 'V224', 'V226', 'V228', 'V229', 'V235', 'V240', 'V258', 
 'V257', 'V253', 'V252', 'V260', 'V261', 'V264', 'V266', 'V267', 'V274', 'V277', 'V220', 'V221', 'V234', 
 'V238', 'V250', 'V271', 'V294', 'V284', 'V285', 'V286', 'V291',
 'V297', 'V303', 'V305', 'V307', 'V309', 'V310', 'V320', 'V281', 'V283', 'V289', 'V296', 'V301', 'V314', 'V332', 'V325', 'V335', 'V338']

##################### Dropping unnecessary V columns #############################
drop_cols = [col for col in df.columns if col[0] == 'V' and col not in reduced_V_cols]
df.drop(drop_cols, axis=1, inplace=True)
print('Dropped ' + str(len(drop_cols)) + ' columns successfully')

##################### Imputing the null values in the left over V columns #############################
df[reduced_V_cols] = my_imputer.fit_transform(df[reduced_V_cols])

##################### Analysing C columns #############################
corr5 = df[c_cols].corr()
plt.figure(figsize = (24,24))
ax = sns.heatmap(corr5, annot = True)

del df['C1']
del df['C2']
del df['C4']
del df['C6']
del df['C7']
del df['C8']
del df['C10']
del df['C11']
del df['C12']
del df['C14']

##################### Analysing id columns #############################
df[id_cols].head(10)
corr5 = df[id_cols].corr()
plt.figure(figsize = (24,24))
ax = sns.heatmap(corr5, annot = True)

df[id_cols].describe()

df[id_cols].isna().sum()

num_id_cols = df[id_cols]._get_numeric_data().columns
cat_cols = list(set(id_cols) - set(num_id_cols))
cat_cols

df[num_id_cols] = my_imputer.fit_transform(df[num_id_cols])

df['id_33'].fillna(df['id_33'].mode()[0], inplace=True)
df['id_34'].fillna(df['id_34'].mode()[0], inplace=True)
df['id_30'].fillna(df['id_30'].mode()[0], inplace=True)
df['id_12'].fillna(df['id_12'].mode()[0], inplace=True)
df['id_31'].fillna(df['id_31'].mode()[0], inplace=True)
df['id_28'].fillna(df['id_28'].mode()[0], inplace=True)
df['id_38'].fillna(df['id_38'].mode()[0], inplace=True)
df['id_37'].fillna(df['id_37'].mode()[0], inplace=True)
df['id_35'].fillna(df['id_35'].mode()[0], inplace=True)
df['id_15'].fillna(df['id_15'].mode()[0], inplace=True)
df['id_16'].fillna(df['id_16'].mode()[0], inplace=True)
df['id_29'].fillna(df['id_29'].mode()[0], inplace=True)
df['id_36'].fillna(df['id_36'].mode()[0], inplace=True)

##################### Filling the Null values in the email columns with a new class name unknown #############################
df['P_emaildomain'] = df['P_emaildomain'].fillna("Unknown")
df['R_emaildomain'] = df['R_emaildomain'].fillna("Unknown")
df['DeviceType'] = df['DeviceType'].fillna("Unknown")
df['DeviceInfo'] = df['DeviceInfo'].fillna("Unknown")

df.isna().sum()

##################### Taking care of all the categorical values #############################
cols = df.columns
num_cols = df._get_numeric_data().columns
cat_cols = list(set(cols) - set(num_cols))
cat_cols

for i in cat_cols:
    print( i+ ": " + str(df[i].nunique()))

small_cat_cols = []
big_cat_cols = []

##################### If the number of unique values in a column is less than 7 than one hot encoding that column#############################
for i in cat_cols:
    if df[i].nunique()<7:
        small_cat_cols.append(i)
    else:
        big_cat_cols.append(i)
small_cat_cols

df= pd.get_dummies(data=df, columns = small_cat_cols )
df.shape

##################### If the number of unique values in a column is more than 7 than label encoding that column#############################
for i in big_cat_cols:
    df[i] = df[i].astype('category')

for i in big_cat_cols:
    df[i] = df[i].cat.codes

cols = df.columns
num_cols = df._get_numeric_data().columns
cat_cols = list(set(cols) - set(num_cols))
cat_cols

train_df = df.iloc[:470898,:]

test_df = df.iloc[470898:,:]

pd.DataFrame(test_df).to_csv('preprocessedV1_testdata.csv')

pd.DataFrame(train_df).to_csv('preprocessedV2_traindata.csv')

pd.DataFrame(y_train).to_csv('preprocessedV3_ytrain.csv')