import pandas as pd
import numpy as np 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

#Importing our training data (3mil) and validation data (1mil)
data = pd.read_csv('SampleTraining3mil.csv',sep=',',header=0,quotechar='"')
Val = pd.read_csv('SampleVal.csv', sep=',', header=0, quotechar='"')

#Feature selection based on variable analysis done in R 
data = data[["click", "banner_pos", "site_category", "device_type", "device_conn_type", "C16", "C17", "C21"]]
Val = Val[["click", "banner_pos", "site_category", "device_type", "device_conn_type", "C16", "C17", "C21"]]

#Encoding banner position
data.banner_pos.unique()
banner_dummy = pd.get_dummies(data['banner_pos'], prefix="BannerPos")
data = pd.concat([data, banner_dummy], axis=1)

Val.banner_pos.unique()
Vbanner_dummy = pd.get_dummies(Val['banner_pos'], prefix="BannerPos")
Val = pd.concat([Val, Vbanner_dummy], axis=1)

data.drop(['banner_pos'], axis=1, inplace=True)
Val.drop(['banner_pos'], axis=1, inplace=True) #Dropping the original banner pos feature

#Site Category - relabelling values based on click through rate threshold of 17%
data['site_category'] = np.where((data['site_category'] != 'dedf689d') & (data['site_category'] != '3e814130') & (data['site_category'] != '42a36e14') & (data['site_category'] != '28905ebd') & (data['site_category'] != 'f028772b'), 'Other', data['site_category'])

Val['site_category'] = np.where((Val['site_category'] != 'dedf689d') & (Val['site_category'] != '3e814130') & (Val['site_category'] != '42a36e14') & (Val['site_category'] != '28905ebd') & (Val['site_category'] != 'f028772b'), 'Other', Val['site_category'])

#encoding site category
data.site_category.unique()
sitecat_dummy = pd.get_dummies(data['site_category'], prefix="SiteCategory")
data = pd.concat([data, sitecat_dummy], axis=1)

Vsitecat_dummy = pd.get_dummies(Val['site_category'], prefix="SiteCategory")
Val = pd.concat([Val, Vsitecat_dummy], axis=1)

data.drop(['site_category'], axis=1, inplace=True) 
Val.drop(['site_category'], axis=1, inplace=True) #dropping the original site category variable

#Device type - encoding
data.device_type.unique()
devicetype_dummy = pd.get_dummies(data['device_type'], prefix='DeviceType')
data = pd.concat([data, devicetype_dummy], axis=1)

Vdevicetype_dummy = pd.get_dummies(Val['device_type'], prefix='DeviceType')
Val = pd.concat([Val, Vdevicetype_dummy], axis=1)

data.drop(['device_type'], axis=1, inplace=True) #dropping original
data.drop(['device_conn_type'], axis=1, inplace=True)

Val.drop(['device_type'], axis=1, inplace=True) #dropping original
Val.drop(['device_conn_type'], axis=1, inplace=True)

#C16 - encoding
data.C16.unique()
C16_dummy = pd.get_dummies(data['C16'], prefix="C16")
data = pd.concat([data, C16_dummy], axis=1)

VC16_dummy = pd.get_dummies(Val['C16'], prefix="C16")
Val = pd.concat([Val, VC16_dummy], axis=1)

data.drop(['C16'], axis=1, inplace=True)
Val.drop(['C16'], axis=1, inplace=True) #dropping original

#C17 - Relabeling values based on click through rate threshold of 17%
data['C17'] = np.where((data['C17'] != 2722) & (data['C17'] != 2662) & (data['C17'] != 1694) & (data['C17'] != 2518) & (data['C17'] != 2687) & (data['C17'] != 2286) & (data['C17'] != 2295) & (data['C17'] != 2569) & (data['C17'] != 827) & (data['C17'] != 2162) & (data['C17'] != 2331) & (data['C17'] != 2284) & (data['C17'] != 1994) & (data['C17'] != 2016) & (data['C17'] != 2443) & (data['C17'] != 1272) & (data['C17'] != 1903) & (data['C17'] != 1993) & (data['C17'] != 2101) & (data['C17'] != 2663) & (data['C17'] != 2594) & (data['C17'] != 2512) & (data['C17'] != 2285) & (data['C17'] != 1939) & (data['C17'] != 2253) & (data['C17'] != 1926) & (data['C17'] != 2689) & (data['C17'] != 1447) & (data['C17'] != 1872) & (data['C17'] != 1991) & (data['C17'] != 2553) & (data['C17'] != 1974) & (data['C17'] != 2717) & (data['C17'] != 196) & (data['C17'] != 2510) & (data['C17'] != 2675) & (data['C17'] != 2647) & (data['C17'] != 2523) & (data['C17'] != 2495) & (data['C17'] != 2639) & (data['C17'] != 1526) & (data['C17'] != 2508) & (data['C17'] != 2316) & (data['C17'] != 1934) & (data['C17'] != 2673) & (data['C17'] != 1107) & (data['C17'] != 178) & (data['C17'] != 2561) & (data['C17'] != 1899) & (data['C17'] != 2711) & (data['C17'] != 1752) & (data['C17'] != 1008) & (data['C17'] != 2635) & (data['C17'] != 2225) & (data['C17'] != 2421) & (data['C17'] != 2374) & (data['C17'] != 423) & (data['C17'] != 2199) & (data['C17'] != 2218) & (data['C17'] != 2500) & (data['C17'] != 2471) & (data['C17'] != 2615) & (data['C17'] != 2299) & (data['C17'] != 2664) & (data['C17'] != 2451) & (data['C17'] != 2043) & (data['C17'] != 2702) & (data['C17'] != 1248) & (data['C17'] != 2684) & (data['C17'] != 2060) & (data['C17'] != 2435) & (data['C17'] != 2672) & (data['C17'] != 572) & (data['C17'] != 2712) & (data['C17'] != 2591) & (data['C17'] != 2600) & (data['C17'] != 1921) & (data['C17'] != 2638) & (data['C17'] != 2303) & (data['C17'] != 2533) & (data['C17'] != 832) & (data['C17'] != 2550) & (data['C17'] != 2550) & (data['C17'] != 1780) & (data['C17'] != 479) & (data['C17'] != 2306) & (data['C17'] != 2206) & (data['C17'] != 2467) & (data['C17'] != 2659) & (data['C17'] != 2545) & (data['C17'] != 2642) & (data['C17'] != 873) & (data['C17'] != 2646) & (data['C17'] != 2668) & (data['C17'] != 112) & (data['C17'] != 2616) & (data['C17'] != 2580) & (data['C17'] != 2624) & (data['C17'] != 2323) & (data['C17'] != 2551) & (data['C17'] != 2655) & (data['C17'] != 544) & (data['C17'] != 2660) & (data['C17'] != 2716) & (data['C17'] != 1401) & (data['C17'] != 2547) & (data['C17'] != 2669) & (data['C17'] != 571) & (data['C17'] != 2496) & (data['C17'] != 686) & (data['C17'] != 2703) & (data['C17'] != 2439) & (data['C17'] != 2522) & (data['C17'] != 2502) & (data['C17'] != 2708) & (data['C17'] != 2539) & (data['C17'] != 2229) & (data['C17'] != 2530) & (data['C17'] != 2605) & (data['C17'] != 2441) & (data['C17'] != 2525) & (data['C17'] != 1722) & (data['C17'] != 2671) & (data['C17'] != 1528) & (data['C17'] != 2039) & (data['C17'] != 1973) & (data['C17'] != 2450) & (data['C17'] != 1255) & (data['C17'] != 2446) & (data['C17'] != 576) & (data['C17'] != 2520) & (data['C17'] != 2226) & (data['C17'] != 2438) & (data['C17'] != 2462) & (data['C17'] != 2242) & (data['C17'] != 2617) & (data['C17'] != 2250) & (data['C17'] != 550) & (data['C17'] != 2440) & (data['C17'] != 2577) & (data['C17'] != 2676) & (data['C17'] != 1882), 'Other', data['C17'])

Val['C17'] = np.where((Val['C17'] != 2722) & (Val['C17'] != 2662) & (Val['C17'] != 1694) & (Val['C17'] != 2518) & (Val['C17'] != 2687) & (Val['C17'] != 2286) & (Val['C17'] != 2295) & (Val['C17'] != 2569) & (Val['C17'] != 827) & (Val['C17'] != 2162) & (Val['C17'] != 2331) & (Val['C17'] != 2284) & (Val['C17'] != 1994) & (Val['C17'] != 2016) & (Val['C17'] != 2443) & (Val['C17'] != 1272) & (Val['C17'] != 1903) & (Val['C17'] != 1993) & (Val['C17'] != 2101) & (Val['C17'] != 2663) & (Val['C17'] != 2594) & (Val['C17'] != 2512) & (Val['C17'] != 2285) & (Val['C17'] != 1939) & (Val['C17'] != 2253) & (Val['C17'] != 1926) & (Val['C17'] != 2689) & (Val['C17'] != 1447) & (Val['C17'] != 1872) & (Val['C17'] != 1991) & (Val['C17'] != 2553) & (Val['C17'] != 1974) & (Val['C17'] != 2717) & (Val['C17'] != 196) & (Val['C17'] != 2510) & (Val['C17'] != 2675) & (Val['C17'] != 2647) & (Val['C17'] != 2523) & (Val['C17'] != 2495) & (Val['C17'] != 2639) & (Val['C17'] != 1526) & (Val['C17'] != 2508) & (Val['C17'] != 2316) & (Val['C17'] != 1934) & (Val['C17'] != 2673) & (Val['C17'] != 1107) & (Val['C17'] != 178) & (Val['C17'] != 2561) & (Val['C17'] != 1899) & (Val['C17'] != 2711) & (Val['C17'] != 1752) & (Val['C17'] != 1008) & (Val['C17'] != 2635) & (Val['C17'] != 2225) & (Val['C17'] != 2421) & (Val['C17'] != 2374) & (Val['C17'] != 423) & (Val['C17'] != 2199) & (Val['C17'] != 2218) & (Val['C17'] != 2500) & (Val['C17'] != 2471) & (Val['C17'] != 2615) & (Val['C17'] != 2299) & (Val['C17'] != 2664) & (Val['C17'] != 2451) & (Val['C17'] != 2043) & (Val['C17'] != 2702) & (Val['C17'] != 1248) & (Val['C17'] != 2684) & (Val['C17'] != 2060) & (Val['C17'] != 2435) & (Val['C17'] != 2672) & (Val['C17'] != 572) & (Val['C17'] != 2712) & (Val['C17'] != 2591) & (Val['C17'] != 2600) & (Val['C17'] != 1921) & (Val['C17'] != 2638) & (Val['C17'] != 2303) & (Val['C17'] != 2533) & (Val['C17'] != 832) & (Val['C17'] != 2550) & (Val['C17'] != 2550) & (Val['C17'] != 1780) & (Val['C17'] != 479) & (Val['C17'] != 2306) & (Val['C17'] != 2206) & (Val['C17'] != 2467) & (Val['C17'] != 2659) & (Val['C17'] != 2545) & (Val['C17'] != 2642) & (Val['C17'] != 873) & (Val['C17'] != 2646) & (Val['C17'] != 2668) & (Val['C17'] != 112) & (Val['C17'] != 2616) & (Val['C17'] != 2580) & (Val['C17'] != 2624) & (Val['C17'] != 2323) & (Val['C17'] != 2551) & (Val['C17'] != 2655) & (Val['C17'] != 544) & (Val['C17'] != 2660) & (Val['C17'] != 2716) & (Val['C17'] != 1401) & (Val['C17'] != 2547) & (Val['C17'] != 2669) & (Val['C17'] != 571) & (Val['C17'] != 2496) & (Val['C17'] != 686) & (Val['C17'] != 2703) & (Val['C17'] != 2439) & (Val['C17'] != 2522) & (Val['C17'] != 2502) & (Val['C17'] != 2708) & (Val['C17'] != 2539) & (Val['C17'] != 2229) & (Val['C17'] != 2530) & (Val['C17'] != 2605) & (Val['C17'] != 2441) & (Val['C17'] != 2525) & (Val['C17'] != 1722) & (Val['C17'] != 2671) & (Val['C17'] != 1528) & (Val['C17'] != 2039) & (Val['C17'] != 1973) & (Val['C17'] != 2450) & (Val['C17'] != 1255) & (Val['C17'] != 2446) & (Val['C17'] != 576) & (Val['C17'] != 2520) & (Val['C17'] != 2226) & (Val['C17'] != 2438) & (Val['C17'] != 2462) & (Val['C17'] != 2242) & (Val['C17'] != 2617) & (Val['C17'] != 2250) & (Val['C17'] != 550) & (Val['C17'] != 2440) & (Val['C17'] != 2577) & (Val['C17'] != 2676) & (Val['C17'] != 1882), 'Other', Val['C17'])

#encoding C17 features
data.C17.unique()
C17_dummy = pd.get_dummies(data['C17'], prefix="C17")
data = pd.concat([data, C17_dummy], axis=1)

VC17_dummy = pd.get_dummies(Val['C17'], prefix="C17")
Val = pd.concat([Val, VC17_dummy], axis=1)

data.drop(['C17'], axis=1, inplace=True) 
Val.drop(['C17'], axis=1, inplace=True) #dropping the original

#C21 - Relabelling values using 17% CTR 
data.C21.unique()
data['C21'] = np.where((data['C21'] != 33) & (data['C21'] != 35) & (data['C21'] != 52) & (data['C21'] != 16) & (data['C21'] != 82) & (data['C21'] != 32) & (data['C21'] != 90) & (data['C21'] != 23) & (data['C21'] != 13) & (data['C21'] != 15) & (data['C21'] != 117) & (data['C21'] != 79) & (data['C21'] != 51) & (data['C21'] != 94) & (data['C21'] != 70), 'Other', data['C21'])

Val['C21'] = np.where((Val['C21'] != 33) & (Val['C21'] != 35) & (Val['C21'] != 52) & (Val['C21'] != 16) & (Val['C21'] != 82) & (Val['C21'] != 32) & (Val['C21'] != 90) & (Val['C21'] != 23) & (Val['C21'] != 13) & (Val['C21'] != 15) & (Val['C21'] != 117) & (Val['C21'] != 79) & (Val['C21'] != 51) & (Val['C21'] != 94) & (Val['C21'] != 70), 'Other', Val['C21'])

#encoding C21 features
C21_dummy = pd.get_dummies(data['C21'], prefix='C21')
data = pd.concat([data, C21_dummy], axis=1)

VC21_dummy = pd.get_dummies(Val['C21'], prefix='C21')
Val = pd.concat([Val, VC21_dummy], axis=1)

data.drop(['C21'], axis=1, inplace=True) 
Val.drop(['C21'], axis=1, inplace=True) #dropping the original

data.to_csv('Train.csv',sep=',',na_rep="NA",header=True,index=False)
Val.to_csv('Val.csv',sep=',',na_rep="NA",header=True,index=False)

#%% Model Building

X = data.iloc[:, 1:187].values #assigning features
Y = data.iloc[:, 0].values #assigning target variable - click

Val_X = Val.iloc[:, 1:187].values

# Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier(n_estimators=500, criterion='gini')
rf = rf.fit(X,Y)
ypred = rf.predict(Val_X)
phat = rf.predict_proba(Val_X)



#Val['ypred'] = ypred
#Val['PHat'] = phat[:,1]
ValRFPred = pd.concat([ypred, phat[:,1]], axis=1)

Val.to_csv('ValRFPred.csv',sep=',',na_rep="NA",header=True,index=False)

# Random Forest - through GridSearch 
param_grid = [{'n_estimators' : [100, 150, 200, 250, 300, 350, 400, 450, 500]}, {'max_features' : ['sqrt', 'log', 0.1, 0.25, 0.5, 0.75, 1]}]

gs_rf = GridSearchCV(estimator=RandomForestClassifier(criterion='gini'), param_grid = param_grid, n_jobs=4, scoring='roc_auc')
gs_rf = gs_rf.fit(X,Y)
ypred_gs_rf = gs_rf.predict_proba(Val_X)

# Logistic Regression 
from sklearn.linear_model import LogisticRegression

gs_lr = GridSearchCV(estimator=LogisticRegression(random_state=0, solver='sag'), param_grid=[{'C': [ 0.00001, 0.0001, 0.001, 0.01, 0.1 ,1 ,10 ,100, 1000, 10000, 100000, 1000000, 10000000], 'penalty':['l1','l2']}], scoring='neg_log_loss')

gs_lr = gs_lr.fit(X,Y)
ypred_lr = gs_lr.predict(Val_X)
phat_lr = gs_lr.predict_proba(Val_X)

ValLRPred = pd.concat([ypred_lr, phat_lr[:,1]], axis=1)
ValLRPred.to_csv('ValLRPred.csv',sep=',',na_rep="NA",header=True,index=False) 

# KNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 10, weights='distance', algorithm='auto', metric='minkowski')
knn = knn.fit(X,Y)
ypred_knn = knn.predict(Val_X)
phat_knn = knn.predict.proba(Val_X)

ValKNNPred = pd.concat([ypred_knn, phat_knn[:,1]], axis=1)
ValKNNPred.to_csv('ValKNNPred.csv',sep=',',na_rep="NA",header=True,index=False) 

#%% Prediction on Test Data
test = pd.read_csv('ProjectTestData.csv', sep=',', header=0, quotechar='"')
test = test[["banner_pos", "site_category", "device_type", "device_conn_type", "C16", "C17", "C21"]]

#Encoding banner position
test.banner_pos.unique()
Tbanner_dummy = pd.get_dummies(test['banner_pos'], prefix="BannerPos")
test = pd.concat([test, Tbanner_dummy], axis=1)

test.drop(['banner_pos'], axis=1, inplace=True)

#Site Category - relabelling values based on click through rate threshold of 17%
test['site_category'] = np.where((test['site_category'] != 'dedf689d') & (test['site_category'] != '3e814130') & (test['site_category'] != '42a36e14') & (test['site_category'] != '28905ebd') & (test['site_category'] != 'f028772b'), 'Other', test['site_category'])

#encoding site category
test.site_category.unique()
Tsitecat_dummy = pd.get_dummies(test['site_category'], prefix="SiteCategory")
test = pd.concat([test, Tsitecat_dummy], axis=1)

test.drop(['site_category'], axis=1, inplace=True)

#Device type - encoding
test.device_type.unique()
Tdevicetype_dummy = pd.get_dummies(test['device_type'], prefix='DeviceType')
test = pd.concat([test, Tdevicetype_dummy], axis=1)

test.drop(['device_type'], axis=1, inplace=True) #dropping original
test.drop(['device_conn_type'], axis=1, inplace=True)

#C16 - encoding
test.C16.unique()
TC16_dummy = pd.get_dummies(test['C16'], prefix="C16")
test = pd.concat([test, TC16_dummy], axis=1)

test.drop(['C16'], axis=1, inplace=True)

#C17 - Relabeling values based on click through rate threshold of 17%
test['C17'] = np.where((test['C17'] != 2722) & (test['C17'] != 2662) & (test['C17'] != 1694) & (test['C17'] != 2518) & (test['C17'] != 2687) & (test['C17'] != 2286) & (test['C17'] != 2295) & (test['C17'] != 2569) & (test['C17'] != 827) & (test['C17'] != 2162) & (test['C17'] != 2331) & (test['C17'] != 2284) & (test['C17'] != 1994) & (test['C17'] != 2016) & (test['C17'] != 2443) & (test['C17'] != 1272) & (test['C17'] != 1903) & (test['C17'] != 1993) & (test['C17'] != 2101) & (test['C17'] != 2663) & (test['C17'] != 2594) & (test['C17'] != 2512) & (test['C17'] != 2285) & (test['C17'] != 1939) & (test['C17'] != 2253) & (test['C17'] != 1926) & (test['C17'] != 2689) & (test['C17'] != 1447) & (test['C17'] != 1872) & (test['C17'] != 1991) & (test['C17'] != 2553) & (test['C17'] != 1974) & (test['C17'] != 2717) & (test['C17'] != 196) & (test['C17'] != 2510) & (test['C17'] != 2675) & (test['C17'] != 2647) & (test['C17'] != 2523) & (test['C17'] != 2495) & (test['C17'] != 2639) & (test['C17'] != 1526) & (test['C17'] != 2508) & (test['C17'] != 2316) & (test['C17'] != 1934) & (test['C17'] != 2673) & (test['C17'] != 1107) & (test['C17'] != 178) & (test['C17'] != 2561) & (test['C17'] != 1899) & (test['C17'] != 2711) & (test['C17'] != 1752) & (test['C17'] != 1008) & (test['C17'] != 2635) & (test['C17'] != 2225) & (test['C17'] != 2421) & (test['C17'] != 2374) & (test['C17'] != 423) & (test['C17'] != 2199) & (test['C17'] != 2218) & (test['C17'] != 2500) & (test['C17'] != 2471) & (test['C17'] != 2615) & (test['C17'] != 2299) & (test['C17'] != 2664) & (test['C17'] != 2451) & (test['C17'] != 2043) & (test['C17'] != 2702) & (test['C17'] != 1248) & (test['C17'] != 2684) & (test['C17'] != 2060) & (test['C17'] != 2435) & (test['C17'] != 2672) & (test['C17'] != 572) & (test['C17'] != 2712) & (test['C17'] != 2591) & (test['C17'] != 2600) & (test['C17'] != 1921) & (test['C17'] != 2638) & (test['C17'] != 2303) & (test['C17'] != 2533) & (test['C17'] != 832) & (test['C17'] != 2550) & (test['C17'] != 2550) & (test['C17'] != 1780) & (test['C17'] != 479) & (test['C17'] != 2306) & (test['C17'] != 2206) & (test['C17'] != 2467) & (test['C17'] != 2659) & (test['C17'] != 2545) & (test['C17'] != 2642) & (test['C17'] != 873) & (test['C17'] != 2646) & (test['C17'] != 2668) & (test['C17'] != 112) & (test['C17'] != 2616) & (test['C17'] != 2580) & (test['C17'] != 2624) & (test['C17'] != 2323) & (test['C17'] != 2551) & (test['C17'] != 2655) & (test['C17'] != 544) & (test['C17'] != 2660) & (test['C17'] != 2716) & (test['C17'] != 1401) & (test['C17'] != 2547) & (test['C17'] != 2669) & (test['C17'] != 571) & (test['C17'] != 2496) & (test['C17'] != 686) & (test['C17'] != 2703) & (test['C17'] != 2439) & (test['C17'] != 2522) & (test['C17'] != 2502) & (test['C17'] != 2708) & (test['C17'] != 2539) & (test['C17'] != 2229) & (test['C17'] != 2530) & (test['C17'] != 2605) & (test['C17'] != 2441) & (test['C17'] != 2525) & (test['C17'] != 1722) & (test['C17'] != 2671) & (test['C17'] != 1528) & (test['C17'] != 2039) & (test['C17'] != 1973) & (test['C17'] != 2450) & (test['C17'] != 1255) & (test['C17'] != 2446) & (test['C17'] != 576) & (test['C17'] != 2520) & (test['C17'] != 2226) & (test['C17'] != 2438) & (test['C17'] != 2462) & (test['C17'] != 2242) & (test['C17'] != 2617) & (test['C17'] != 2250) & (test['C17'] != 550) & (test['C17'] != 2440) & (test['C17'] != 2577) & (test['C17'] != 2676) & (test['C17'] != 1882), 'Other', test['C17'])

#encoding C17 features
test.C17.unique()
TC17_dummy = pd.get_dummies(test['C17'], prefix="C17")
test = pd.concat([test, TC17_dummy], axis=1)

test.drop(['C17'], axis=1, inplace=True) 

#C21 - Relabelling values using 17% CTR 
test.C21.unique()
test['C21'] = np.where((test['C21'] != 33) & (test['C21'] != 35) & (test['C21'] != 52) & (test['C21'] != 16) & (test['C21'] != 82) & (test['C21'] != 32) & (test['C21'] != 90) & (test['C21'] != 23) & (test['C21'] != 13) & (test['C21'] != 15) & (test['C21'] != 117) & (test['C21'] != 79) & (test['C21'] != 51) & (test['C21'] != 94) & (test['C21'] != 70), 'Other', test['C21'])

#encoding C21 features
TC21_dummy = pd.get_dummies(test['C21'], prefix='C21')
test = pd.concat([test, TC21_dummy], axis=1)

test.drop(['C21'], axis=1, inplace=True) 
test.head()

# Assign X variables for test
data, test = data.align(test, join='outer', axis=1, fill_value=0)

Test_X = test.iloc[:, 0:187].values

ypred_test = rf.predict(Test_X)
phat_test = rf.predict_proba(Test_X)

#gives us information about max depth of each tree in random forest 
max_depth=list()
for tree in rf.estimators_: max_depth.append(tree.tree_.max_depth)
print("avg max depth %0.1f" % (sum(max_depth) / len(max_depth)))
len(max_depth)

# Assigning column names and exporting it to import in R 
test['ypred'] = ypred_test
test['PHat'] = phat_test[:,1]
TestPred = pd.concat([ypred_test, phat_test[:,1]], axis=1)

TestPred.to_csv('TestPred.csv',sep=',',na_rep="NA",header=True,index=False)

import gc
gc.collect()
