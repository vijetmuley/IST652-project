import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import decomposition
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn import linear_model
#-------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------------

def pca(df):

    rem=["Date","Events"]
    col_names=df.columns.values.tolist()
    data=pd.DataFrame()
    for col in col_names:
        if col in rem:
            continue
        else:
            data[col]=df[col]

    print(data.info())
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print(data.isnull().sum())
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    data=data.replace("-",0)

    #There's 'T' in precipitation column, data source says T represents precipitation close to 0.0 For PCA, I will be replacing it with 0
    data=data.replace("T",0)

    data=data.astype(np.float16)
    print(data.info())
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print(data.head(20))
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

    x=data.values
    print(np.count_nonzero(np.isnan(x)))
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    #Scaling was causing problem. I guess it had floating point overflow and filled non scalable numbers with NaN which caused error in fitting.
    #x=scale(x)
    cov_mat=PCA(n_components=19)
    cov_mat.fit(x)
    var_ratio=cov_mat.explained_variance_ratio_
    var=np.cumsum(np.round(cov_mat.explained_variance_ratio_, decimals=3)*100)
    print(var)

    #Now PCA:
    fig=plt.figure()
    plt.plot(var)
    plt.ylabel('% Variance Explained')
    plt.xlabel('# of Features')
    plt.title('PCA Analysis')
    plt.style.context('seaborn-whitegrid')
    plt.show()

    #PCA shows that rough;y 7-8 components are enough to explain all the variance between the 19 components of original dataset
    #The inference I can draw from this is that; when counted, the average columns for all weather attributes and the precipitation
    #columns add upto 7 components. It also makes sense to actually use the avergae values. Max and min are too specific to explain the conditions
    #spread across all day. Also, using average won't cause error, for max and min in a day won't go too far from the average, since outlier temperatures
    #and drastic variances almost never happen. So, it'd make sense to use the average columns.

#-------------------------------------------------------------------------------------------------------------------------------------

def outlier_test(inp):
    threshold=3
    avg_inp=np.mean(inp)
    stdev=np.std(inp)
    zscores=[]
    for i in inp:
        z=(i-avg_inp)/stdev
        zscores.append(z)
    return np.where(np.abs(zscores) > threshold)

#-------------------------------------------------------------------------------------------------------------------------------------

def viz(df,date):


    df["date"]=date["date"]
    fig=plt.figure()
    sns.set(style="white")
    plot1=sns.scatterplot(x="date",y="VisibilityAvgMiles",hue="Events",data=df)
    plt.show(plot1)

    '''
    fig=plt.figure()
    sns.set(style="darkgrid")
    plot2=sns.countplot(y="Events",data=df)
    plt.show(plot2)
    '''

#-------------------------------------------------------------------------------------------------------------------------------------

def clean_data(df):

    #copying only the columns we need:
    col_names=df.columns.values.tolist()
    date=pd.DataFrame()
    date["date"]=df["Date"]
    col_keep=["TempAvgF","DewPointAvgF","HumidityAvgPercent","SeaLevelPressureAvgInches","VisibilityAvgMiles","WindAvgMPH","PrecipitationSumInches","Events"]
    clean_df=pd.DataFrame()
    for col in col_names:
        if col in col_keep:
            clean_df[col]=df[col]
        else:
            continue
    #let's start with reformating the 'Events' column:

    #viz(clean_df,date)

    i=0
    for ele in clean_df["Events"]:
        if ele==" ":
            clean_df.loc[i,"Events"]="Clear weather"

        elif ele=="Rain" or ele=="Rain , Snow":
            clean_df.loc[i,"Events"]="Rain"

        elif ele=="Rain , Thunderstorm":
            clean_df.loc[i,"Events"]="Rain+Thunderstorm"

        else:
            clean_df.loc[i,"Events"]="Fog+Rain+Thunderstorm"
        i=i+1

    print(clean_df.head(n=20))
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

    #next, we gotta conver the columns stored as 'object' and not numeric. But before that, we have to check for invalid entries:
    #There are entries in precipitation, 'T', which signify trace as per the source. Very close to 0, so I will rpelace T with 0

    #All 124 non numeric entries for precipitation were T, which are handled
    #clean_df=clean_df.replace("T",0)

    #This is a new method I learned, I wasn't knowing it before I read about it:

    col_keep=col_keep[:-1]
    to_rem_rows={}
    for col in col_keep:
        non_num=clean_df[pd.to_numeric(clean_df[col],errors="coerce").isnull()].loc[:,col].value_counts()
        if col=="PrecipitationSumInches":
            continue
        else:
            to_rem_rows[col]=list(clean_df[pd.to_numeric(clean_df[col],errors="coerce").isnull()].loc[:,col].index)
        print(non_num)

    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print(to_rem_rows)
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    final_row_rem=[]
    for row_rem in to_rem_rows:
        for ele in to_rem_rows[row_rem]:
            if ele not in final_row_rem:
                final_row_rem.append(ele)
            else:
                continue

    #Gives out 12 rows, planning to drop them. In this case, no data is sadly not any type of data. It's just readings that were probably lost.
    #The 'T' in Precipitation, as mentioned earlier, is very little precipitation amount. Although I'd be putting 0 in its place, that 0 and an actual 0 precipitation
    #have different meanings. So, I am thinking of making a dummy column,if Precipitation had T, this dumy would be 1.

    print(final_row_rem)
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

    clean_df["T_dummy"]=0
    for i in range(len(clean_df)):
        if clean_df.loc[i,"PrecipitationSumInches"]=="T":
            clean_df.loc[i,"T_dummy"]=1
        else:
            continue

    clean_df["PrecipitationSumInches"]=clean_df["PrecipitationSumInches"].replace("T",0)
    clean_df=clean_df.drop(final_row_rem)
    date=date.drop(final_row_rem)
    #Checking whether non numeric rows have been dropped or not:
    for col in col_keep:
        non_num=clean_df[pd.to_numeric(clean_df[col],errors="coerce").isnull()].loc[:,col].value_counts()
        print(non_num)
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    #Yup, successfully removed the rows having non numeric data (or at least the one that cannot be coerced)
    for col in col_keep:
        clean_df[[col]]=clean_df[[col]].apply(pd.to_numeric)

    out_idx={}
    for col in col_keep:
        print("Running for column",col,"Data type:",clean_df[col].dtype)
        if(clean_df[col].dtype=="int64" or clean_df[col].dtype=="float64"):
            out=list(outlier_test(clean_df[col]))
            if len(out[0])>0:
                print("Outliers found in column",col,"Number of outliers:",len(out[0]))
                out_idx[col]=out
        else:
            print("No outliers in column",col)

    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print(clean_df.head(n=20))
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

    #viz(clean_df,date)

    return clean_df

#-------------------------------------------------------------------------------------------------------------------------------------

def nb(data):

    x=pd.DataFrame()
    y=pd.DataFrame()
    col_names=data.columns.values.tolist()
    for col in col_names:
        if col=="Events":
            y["Events"]=data["Events"]
        else:
            x[col]=data.loc[:,col]
    #print(x.head(n=20))
    #print(y.head(n=20))
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=30)
    print(x_train.head())
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print(x_test.head())
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print(y_train.head())
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print(y_test.head())
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

    #We will try differentiate between the accuracy achieved using both scalers. Standard scaler first:

    scaler=StandardScaler()
    x_train_std=scaler.fit_transform(x_train)
    x_test_std=scaler.fit_transform(x_test)

    print(x_train_std[:5])
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print(x_test_std[:5])
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

    gnb=GaussianNB()
    gnb.fit(x_train_std,y_train)
    pred=gnb.predict(x_test_std)

    print("For Gaussian NB with standard scaler:")
    print(confusion_matrix(y_test,pred))
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print(classification_report(y_test,pred))
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    acc="Accuracy for Naive Bayes model using Standard Scaler is "+str((accuracy_score(y_test,pred))*100)+"%"
    print(acc)
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

    #Now for minmaxscaler
    mscaler=MinMaxScaler()
    x_train_m=mscaler.fit_transform(x_train)
    x_test_m=mscaler.fit_transform(x_test)

    print(x_train_m[:5])
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print(x_test_m[:5])
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

    gnb1=GaussianNB()
    gnb1.fit(x_train_m,y_train)
    pred1=gnb1.predict(x_test_m)

    print("For Gaussian NB with MinMaxScaler:")
    print(confusion_matrix(y_test,pred1))
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print(classification_report(y_test,pred1))
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    acc1="Accuracy for Naive Bayes model using MinMaxScaler is "+str((accuracy_score(y_test,pred1))*100)+"%"
    print(acc1)
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

#-------------------------------------------------------------------------------------------------------------------------------------

def log_reg(df):

    reg_df=pd.DataFrame()
    target=pd.DataFrame()
    col_names=df.columns.values.tolist()
    for col in col_names:
        if col=="Events":
            target[col]=df.loc[:,col]
        else:
            reg_df[col]=df.loc[:,col]

    x_train,x_test,y_train,y_test=train_test_split(reg_df,target,test_size=0.33,random_state=12)

    mod=linear_model.LogisticRegression(solver="newton-cg",multi_class="multinomial").fit(x_train,y_train)
    pred=mod.predict(x_test)
    acc="Accuracy for Multinomial Logistic Regression model is "+str(accuracy_score(y_test,pred)*100)+"%"
    print(acc)
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

#-------------------------------------------------------------------------------------------------------------------------------------

def main():

    df=pd.read_csv("austin_weather.csv")
    #pca(df)
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print(df.info())
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    #print(df.describe())
    print(df.head(n=20))
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    for attr in df["Events"].unique():
        print("Unique value: ",attr)
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    print(df["Events"].value_counts())
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    #So I can refine the count, taking this above thing as a dictionary then using split on the keys and checking for individual element to add to the new count.
    #But how do I bin it? Or do I leave the bins the way they are?

    data=clean_data(df)
    print(data.info())
    print("------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    nb(data)
    log_reg(data)

#-------------------------------------------------------------------------------------------------------------------------------------

if __name__=="__main__":

    main()
