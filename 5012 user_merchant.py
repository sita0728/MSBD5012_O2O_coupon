#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from datetime import date


"""
dataset split:
                      (date_received)                              
           dateset3: 20160701~20160731 (113640),features3 from 20160315~20160630  (off_test)
           dateset2: 20160515~20160615 (258446),features2 from 20160201~20160514  
           dateset1: 20160414~20160514 (138303),features1 from 20160101~20160413        
1.merchant related: 
      sales_use_coupon. total_coupon
      transfer_rate = sales_use_coupon/total_coupon.
      merchant_avg_distance,merchant_min_distance,merchant_max_distance of those use coupon 
      total_sales.  coupon_rate = sales_use_coupon/total_sales.  
       
2.coupon related: 
      discount_rate. discount_man. discount_jian. is_man_jian
      day_of_week,day_of_month. (date_received)
      
3.user related: 
      distance. 
      user_avg_distance, user_min_distance,user_max_distance. 
      buy_use_coupon. buy_total. coupon_received.
      buy_use_coupon/coupon_received. 
      avg_diff_date_datereceived. min_diff_date_datereceived. max_diff_date_datereceived.  
      count_merchant.  
4.user_merchant:
      times_user_buy_merchant_before.
     
5. other feature:
      this_month_user_receive_all_coupon_count
      this_month_user_receive_same_coupon_count
      this_month_user_receive_same_coupon_lastone
      this_month_user_receive_same_coupon_firstone
      this_day_user_receive_all_coupon_count
      this_day_user_receive_same_coupon_count
      day_gap_before, day_gap_after  (receive the same coupon)
"""


#1754884 record,1053282 with coupon_id,9738 coupon. date_received:20160101~20160615,date:20160101~20160630, 539438 users, 8415 merchants
off_train = pd.read_csv('C:/Users/yanta/Desktop/MSBD 5012 ML/project/ccf_offline_stage1_train/ccf_offline_stage1_train.csv',header=None,low_memory=False)
off_train.columns = ['user_id','merchant_id','coupon_id','discount_rate','distance','date_received','date']
#2050 coupon_id. date_received:20160701~20160731, 76309 users(76307 in trainset, 35965 in online_trainset), 1559 merchants(1558 in trainset)
off_test = pd.read_csv('ccf_offline_stage1_test_revised.csv',header=None,low_memory=False)
off_test.columns = ['user_id','merchant_id','coupon_id','discount_rate','distance','date_received']
#11429826 record(872357 with coupon_id),762858 user(267448 in off_train)
on_train = pd.read_csv('C:/Users/yanta/Desktop/MSBD 5012 ML/project/ccf_online_stage2_train/ccf_online_stage1_train.csv',header=None,low_memory=False)
on_train.columns = ['user_id','merchant_id','action','coupon_id','discount_rate','date_received','date']


dataset3 = off_test
feature3 = off_train[((off_train.date>='20160315')&(off_train.date<='20160630'))|((off_train.date=='null')&(off_train.date_received>='20160315')&(off_train.date_received<='20160630'))]
dataset2 = off_train[(off_train.date_received>='20160515')&(off_train.date_received<='20160615')]
feature2 = off_train[(off_train.date>='20160201')&(off_train.date<='20160514')|((off_train.date=='null')&(off_train.date_received>='20160201')&(off_train.date_received<='20160514'))]
dataset1 = off_train[(off_train.date_received>='20160414')&(off_train.date_received<='20160514')]
feature1 = off_train[(off_train.date>='20160101')&(off_train.date<='20160413')|((off_train.date=='null')&(off_train.date_received>='20160101')&(off_train.date_received<='20160413'))]


############# other feature ##################3

##################  user_merchant related feature #########################

"""
4.user_merchant:
      times_user_buy_merchant_before. 
"""
#for dataset3
all_user_merchant = feature3[['user_id','merchant_id']].copy()
all_user_merchant.drop_duplicates(inplace=True)

t = feature3[['user_id','merchant_id','date']]
t = t[t.date!='null'][['user_id','merchant_id']]
t['user_merchant_buy_total'] = 1
t = t.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t.drop_duplicates(inplace=True)

t1 = feature3[['user_id','merchant_id','coupon_id']]
t1 = t1[t1.coupon_id!='null'][['user_id','merchant_id']]
t1['user_merchant_received'] = 1
t1 = t1.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t1.drop_duplicates(inplace=True)

t2 = feature3[['user_id','merchant_id','date','date_received']]
t2 = t2[(t2.date!='null')&(t2.date_received!='null')][['user_id','merchant_id']]
t2['user_merchant_buy_use_coupon'] = 1
t2 = t2.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t2.drop_duplicates(inplace=True)

t3 = feature3[['user_id','merchant_id']].copy()
t3['user_merchant_any'] = 1
t3 = t3.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t3.drop_duplicates(inplace=True)

t4 = feature3[['user_id','merchant_id','date','coupon_id']]
t4 = t4[(t4.date!='null')&(t4.coupon_id=='null')][['user_id','merchant_id']]
t4['user_merchant_buy_common'] = 1
t4 = t4.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t4.drop_duplicates(inplace=True)

user_merchant3 = pd.merge(all_user_merchant,t,on=['user_id','merchant_id'],how='left')
user_merchant3 = pd.merge(user_merchant3,t1,on=['user_id','merchant_id'],how='left')
user_merchant3 = pd.merge(user_merchant3,t2,on=['user_id','merchant_id'],how='left')
user_merchant3 = pd.merge(user_merchant3,t3,on=['user_id','merchant_id'],how='left')
user_merchant3 = pd.merge(user_merchant3,t4,on=['user_id','merchant_id'],how='left')
user_merchant3.user_merchant_buy_use_coupon = user_merchant3.user_merchant_buy_use_coupon.replace(np.nan,0)
user_merchant3.user_merchant_buy_common = user_merchant3.user_merchant_buy_common.replace(np.nan,0)
user_merchant3['user_merchant_coupon_transfer_rate'] = user_merchant3.user_merchant_buy_use_coupon.astype('float') / user_merchant3.user_merchant_received.astype('float')
user_merchant3['user_merchant_coupon_buy_rate'] = user_merchant3.user_merchant_buy_use_coupon.astype('float') / user_merchant3.user_merchant_buy_total.astype('float')
user_merchant3['user_merchant_rate'] = user_merchant3.user_merchant_buy_total.astype('float') / user_merchant3.user_merchant_any.astype('float')
user_merchant3['user_merchant_common_buy_rate'] = user_merchant3.user_merchant_buy_common.astype('float') / user_merchant3.user_merchant_buy_total.astype('float')
user_merchant3.to_csv('C:/Users/yanta/Desktop/MSBD 5012 ML/project/user_merchant3.csv',index=None)

#for dataset2
all_user_merchant = feature2[['user_id','merchant_id']].copy()
all_user_merchant.drop_duplicates(inplace=True)

t = feature2[['user_id','merchant_id','date']]
t = t[t.date!='null'][['user_id','merchant_id']]
t['user_merchant_buy_total'] = 1
t = t.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t.drop_duplicates(inplace=True)

t1 = feature2[['user_id','merchant_id','coupon_id']]
t1 = t1[t1.coupon_id!='null'][['user_id','merchant_id']]
t1['user_merchant_received'] = 1
t1 = t1.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t1.drop_duplicates(inplace=True)

t2 = feature2[['user_id','merchant_id','date','date_received']]
t2 = t2[(t2.date!='null')&(t2.date_received!='null')][['user_id','merchant_id']]
t2['user_merchant_buy_use_coupon'] = 1
t2 = t2.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t2.drop_duplicates(inplace=True)

t3 = feature2[['user_id','merchant_id']].copy()
t3['user_merchant_any'] = 1
t3 = t3.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t3.drop_duplicates(inplace=True)

t4 = feature2[['user_id','merchant_id','date','coupon_id']]
t4 = t4[(t4.date!='null')&(t4.coupon_id=='null')][['user_id','merchant_id']]
t4['user_merchant_buy_common'] = 1
t4 = t4.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t4.drop_duplicates(inplace=True)

user_merchant2 = pd.merge(all_user_merchant,t,on=['user_id','merchant_id'],how='left')
user_merchant2 = pd.merge(user_merchant2,t1,on=['user_id','merchant_id'],how='left')
user_merchant2 = pd.merge(user_merchant2,t2,on=['user_id','merchant_id'],how='left')
user_merchant2 = pd.merge(user_merchant2,t3,on=['user_id','merchant_id'],how='left')
user_merchant2 = pd.merge(user_merchant2,t4,on=['user_id','merchant_id'],how='left')
user_merchant2.user_merchant_buy_use_coupon = user_merchant2.user_merchant_buy_use_coupon.replace(np.nan,0)
user_merchant2.user_merchant_buy_common = user_merchant2.user_merchant_buy_common.replace(np.nan,0)
user_merchant2['user_merchant_coupon_transfer_rate'] = user_merchant2.user_merchant_buy_use_coupon.astype('float') / user_merchant2.user_merchant_received.astype('float')
user_merchant2['user_merchant_coupon_buy_rate'] = user_merchant2.user_merchant_buy_use_coupon.astype('float') / user_merchant2.user_merchant_buy_total.astype('float')
user_merchant2['user_merchant_rate'] = user_merchant2.user_merchant_buy_total.astype('float') / user_merchant2.user_merchant_any.astype('float')
user_merchant2['user_merchant_common_buy_rate'] = user_merchant2.user_merchant_buy_common.astype('float') / user_merchant2.user_merchant_buy_total.astype('float')
user_merchant2.to_csv('C:/Users/yanta/Desktop/MSBD 5012 ML/project/user_merchant2.csv',index=None)

#for dataset2
all_user_merchant = feature1[['user_id','merchant_id']].copy()
all_user_merchant.drop_duplicates(inplace=True)

t = feature1[['user_id','merchant_id','date']]
t = t[t.date!='null'][['user_id','merchant_id']]
t['user_merchant_buy_total'] = 1
t = t.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t.drop_duplicates(inplace=True)

t1 = feature1[['user_id','merchant_id','coupon_id']]
t1 = t1[t1.coupon_id!='null'][['user_id','merchant_id']]
t1['user_merchant_received'] = 1
t1 = t1.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t1.drop_duplicates(inplace=True)

t2 = feature1[['user_id','merchant_id','date','date_received']]
t2 = t2[(t2.date!='null')&(t2.date_received!='null')][['user_id','merchant_id']]
t2['user_merchant_buy_use_coupon'] = 1
t2 = t2.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t2.drop_duplicates(inplace=True)

t3 = feature1[['user_id','merchant_id']].copy()
t3['user_merchant_any'] = 1
t3 = t3.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t3.drop_duplicates(inplace=True)

t4 = feature1[['user_id','merchant_id','date','coupon_id']]
t4 = t4[(t4.date!='null')&(t4.coupon_id=='null')][['user_id','merchant_id']]
t4['user_merchant_buy_common'] = 1
t4 = t4.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t4.drop_duplicates(inplace=True)

user_merchant1 = pd.merge(all_user_merchant,t,on=['user_id','merchant_id'],how='left')
user_merchant1 = pd.merge(user_merchant1,t1,on=['user_id','merchant_id'],how='left')
user_merchant1 = pd.merge(user_merchant1,t2,on=['user_id','merchant_id'],how='left')
user_merchant1 = pd.merge(user_merchant1,t3,on=['user_id','merchant_id'],how='left')
user_merchant1 = pd.merge(user_merchant1,t4,on=['user_id','merchant_id'],how='left')
user_merchant1.user_merchant_buy_use_coupon = user_merchant1.user_merchant_buy_use_coupon.replace(np.nan,0)
user_merchant1.user_merchant_buy_common = user_merchant1.user_merchant_buy_common.replace(np.nan,0)
user_merchant1['user_merchant_coupon_transfer_rate'] = user_merchant1.user_merchant_buy_use_coupon.astype('float') / user_merchant1.user_merchant_received.astype('float')
user_merchant1['user_merchant_coupon_buy_rate'] = user_merchant1.user_merchant_buy_use_coupon.astype('float') / user_merchant1.user_merchant_buy_total.astype('float')
user_merchant1['user_merchant_rate'] = user_merchant1.user_merchant_buy_total.astype('float') / user_merchant1.user_merchant_any.astype('float')
user_merchant1['user_merchant_common_buy_rate'] = user_merchant1.user_merchant_buy_common.astype('float') / user_merchant1.user_merchant_buy_total.astype('float')
user_merchant1.to_csv('C:/Users/yanta/Desktop/MSBD 5012 ML/project/user_merchant1.csv',index=None)








##################  generate training and testing set ################
def get_label(s):
    s = s.split(':')
    if s[0]=='null':
        return 0
    elif (date(int(s[0][0:4]),int(s[0][4:6]),int(s[0][6:8]))-date(int(s[1][0:4]),int(s[1][4:6]),int(s[1][6:8]))).days<=15:
        return 1
    else:
        return -1


coupon3 = pd.read_csv('data/coupon3_feature.csv')
merchant3 = pd.read_csv('data/merchant3_feature.csv')
user3 = pd.read_csv('data/user3_feature.csv')
user_merchant3 = pd.read_csv('data/user_merchant3.csv')
other_feature3 = pd.read_csv('data/other_feature3.csv')
dataset3 = pd.merge(coupon3,merchant3,on='merchant_id',how='left')
dataset3 = pd.merge(dataset3,user3,on='user_id',how='left')
dataset3 = pd.merge(dataset3,user_merchant3,on=['user_id','merchant_id'],how='left')
dataset3 = pd.merge(dataset3,other_feature3,on=['user_id','coupon_id','date_received'],how='left')
dataset3.drop_duplicates(inplace=True)
print(dataset3.shape)

dataset3.user_merchant_buy_total = dataset3.user_merchant_buy_total.replace(np.nan,0)
dataset3.user_merchant_any = dataset3.user_merchant_any.replace(np.nan,0)
dataset3.user_merchant_received = dataset3.user_merchant_received.replace(np.nan,0)
dataset3['is_weekend'] = dataset3.day_of_week.apply(lambda x:1 if x in (6,7) else 0)
weekday_dummies = pd.get_dummies(dataset3.day_of_week)
weekday_dummies.columns = ['weekday'+str(i+1) for i in range(weekday_dummies.shape[1])]
dataset3 = pd.concat([dataset3,weekday_dummies],axis=1)
dataset3.drop(['merchant_id','day_of_week','coupon_count'],axis=1,inplace=True)
dataset3 = dataset3.replace('null',np.nan)
dataset3.to_csv('data/dataset3.csv',index=None)


coupon2 = pd.read_csv('data/coupon2_feature.csv')
merchant2 = pd.read_csv('data/merchant2_feature.csv')
user2 = pd.read_csv('data/user2_feature.csv')
user_merchant2 = pd.read_csv('data/user_merchant2.csv')
other_feature2 = pd.read_csv('data/other_feature2.csv')
dataset2 = pd.merge(coupon2,merchant2,on='merchant_id',how='left')
dataset2 = pd.merge(dataset2,user2,on='user_id',how='left')
dataset2 = pd.merge(dataset2,user_merchant2,on=['user_id','merchant_id'],how='left')
dataset2 = pd.merge(dataset2,other_feature2,on=['user_id','coupon_id','date_received'],how='left')
dataset2.drop_duplicates(inplace=True)
print(dataset2.shape)

dataset2.user_merchant_buy_total = dataset2.user_merchant_buy_total.replace(np.nan,0)
dataset2.user_merchant_any = dataset2.user_merchant_any.replace(np.nan,0)
dataset2.user_merchant_received = dataset2.user_merchant_received.replace(np.nan,0)
dataset2['is_weekend'] = dataset2.day_of_week.apply(lambda x:1 if x in (6,7) else 0)
weekday_dummies = pd.get_dummies(dataset2.day_of_week)
weekday_dummies.columns = ['weekday'+str(i+1) for i in range(weekday_dummies.shape[1])]
dataset2 = pd.concat([dataset2,weekday_dummies],axis=1)
dataset2['label'] = dataset2.date.astype('str') + ':' +  dataset2.date_received.astype('str')
dataset2.label = dataset2.label.apply(get_label)
dataset2.drop(['merchant_id','day_of_week','date','date_received','coupon_id','coupon_count'],axis=1,inplace=True)
dataset2 = dataset2.replace('null',np.nan)
dataset2.to_csv('data/dataset2.csv',index=None)


coupon1 = pd.read_csv('data/coupon1_feature.csv')
merchant1 = pd.read_csv('data/merchant1_feature.csv')
user1 = pd.read_csv('data/user1_feature.csv')
user_merchant1 = pd.read_csv('data/user_merchant1.csv')
other_feature1 = pd.read_csv('data/other_feature1.csv')
dataset1 = pd.merge(coupon1,merchant1,on='merchant_id',how='left')
dataset1 = pd.merge(dataset1,user1,on='user_id',how='left')
dataset1 = pd.merge(dataset1,user_merchant1,on=['user_id','merchant_id'],how='left')
dataset1 = pd.merge(dataset1,other_feature1,on=['user_id','coupon_id','date_received'],how='left')
dataset1.drop_duplicates(inplace=True)
print(dataset1.shape)

dataset1.user_merchant_buy_total = dataset1.user_merchant_buy_total.replace(np.nan,0)
dataset1.user_merchant_any = dataset1.user_merchant_any.replace(np.nan,0)
dataset1.user_merchant_received = dataset1.user_merchant_received.replace(np.nan,0)
dataset1['is_weekend'] = dataset1.day_of_week.apply(lambda x:1 if x in (6,7) else 0)
weekday_dummies = pd.get_dummies(dataset1.day_of_week)
weekday_dummies.columns = ['weekday'+str(i+1) for i in range(weekday_dummies.shape[1])]
dataset1 = pd.concat([dataset1,weekday_dummies],axis=1)
dataset1['label'] = dataset1.date.astype('str') + ':' +  dataset1.date_received.astype('str')
dataset1.label = dataset1.label.apply(get_label)
dataset1.drop(['merchant_id','day_of_week','date','date_received','coupon_id','coupon_count'],axis=1,inplace=True)
dataset1 = dataset1.replace('null',np.nan)
dataset1.to_csv('data/dataset1.csv',index=None)


