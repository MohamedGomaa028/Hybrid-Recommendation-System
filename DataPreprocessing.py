import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

Rating=pd.read_csv("E:/MoviesLensDataset/ml-1m/ratings.dat",sep="::",engine='python',names=["UserId","MovieIDs","Rating","Timestamp"],
                   dtype={"UserId":"int32","MovieIDs":"int32","Rating":"int32","Timestamp":"int32"})

Users=pd.read_csv("E:/MoviesLensDataset/ml-1m/users.dat",sep="::",engine='python',names=['UserId','Gender','Age','Occupation','Zip-code'])

Movie=pd.read_csv("E:/MoviesLensDataset/ml-1m/movies.dat",sep="::",engine='python',names=["MovieIDs","Ttile","Genres"],
                  encoding="latin-1" )

df_RU=pd.merge(Rating,Users,on='UserId')
df=pd.merge(df_RU,Movie,on='MovieIDs')
#print(df.head())
#print(df.shape)

x=df.isna().sum().sort_values(ascending=False)
#print(x)
#print(df.duplicated().sum())

X_Recommendation_train,X_Recommendation_test=train_test_split(df,test_size=0.2,random_state=1)

def prepare_Data(X):
    genre=X["Genres"].str.get_dummies("|")
    gender=pd.get_dummies(X["Gender"],prefix="Gender",dtype=int)
    #print(f'Genre {genre.head()}\n Gender {gender.head()}')
    UserRating=pd.concat([X.drop(['Genres','Gender'],axis=1),genre,gender],axis=1)
    #print(UserRating.head())
    columnsName=[]
    for i in genre.columns:
      columnsName.append(i)
      #print(columnsName)
    for i in columnsName:
       UserRating[i]= UserRating[i] * UserRating["Rating"]
       #print(UserRating.head())
    GenderColumnsName=[]
    for gender in gender.columns:
        GenderColumnsName.append(gender)
    #user=UserRating[['UserId',"Occupation","Rating"]+GenderColumnsName+columnsName]
    #print(user.head())
    Userrange=UserRating.groupby("UserId")[genre.columns].mean().reset_index()
    #print(Userrange.head())
    userinfo=UserRating[["UserId",GenderColumnsName[0],GenderColumnsName[1],"Occupation","Age"]].drop_duplicates()
    Userrates=userinfo.merge(Userrange,on='UserId')
    #print(Userrates.head())

    return Userrates


Userrates=prepare_Data(df)
X_Kmean_train,X_Kmean_test=train_test_split(Userrates,test_size=0.2,random_state=1)
##########################Training data Normalization##############################

UserDatascale_Train=X_Kmean_train[["Occupation","Age"]]
X_Kmean_train=X_Kmean_train.drop(["UserId","Occupation","Age"],axis=1)
X_Kmean_train=X_Kmean_train.to_numpy()
scaler=StandardScaler()
xscaled_train=scaler.fit_transform(UserDatascale_Train)
X=np.hstack((X_Kmean_train,xscaled_train))

##########################Test data Normalization##############################
UserDatascale_test=X_Kmean_test[["Occupation","Age"]]
X_Kmean_test=X_Kmean_test.drop(["UserId","Occupation","Age"],axis=1)
X_Kmean_test=X_Kmean_test.to_numpy()
scaler=StandardScaler()
xscaled_test=scaler.fit_transform(UserDatascale_test)
X_Kmean_test=np.hstack((X_Kmean_test,xscaled_test))
