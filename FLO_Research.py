import pandas as pd
import numpy as np
import seaborn as sns
import os
import warnings
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import datetime as dt

warnings.simplefilter("ignore", category = ConvergenceWarning)
warnings.simplefilter(action = 'ignore', category = FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


################################################
# 1. Exploratory Data Analysis
################################################

def check_df(dataframe, head = 5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


def columns_info(dataframe):
    columns, dtypes, unique, nunique, nulls = [], [], [], [], []

    for cols in dataframe.columns:
        columns.append(cols)
        dtypes.append(dataframe[cols].dtype)
        unique.append(dataframe[cols].unique())
        nunique.append(dataframe[cols].nunique())
        nulls.append(dataframe[cols].isnull().sum())

    return pd.DataFrame({"Columns": columns,
                         "Data_Type": dtypes,
                         "Unique_Values": unique,
                         "Number_of_Unique": nunique,
                         "Missing_Values": nulls})


########################## Loading The Data  ###########################
base_data = pd.read_csv("datasets/flo_data_20k.csv",
                        parse_dates = ["first_order_date", "last_order_date",
                                       "last_order_date_online", "last_order_date_offline"])
df = base_data.copy()

########################## Exploring Data  ###########################
check_df(df)

columns_info(df)


################################################
# 2. Data Preprocessing & Feature Engineering
################################################
def category_transform(dataframe, cat_repo_col):
    cats = ",".join(cat_repo_col.unique()).replace("[", "").replace("]", "").split(",")
    cat_cols = list(set([i.strip() for i in cats if len(i) > 1]))
    for col in cat_cols:
        dataframe[col] = cat_repo_col.apply(lambda x: 1 if x.__contains__(str(col)) else 0)


# Enlarging variables name
df.columns = [col.upper() for col in df.columns]

df.index = df["MASTER_ID"]
df.drop("MASTER_ID", axis = 1, inplace = True)

df["TENURE"] = (df["LAST_ORDER_DATE"].max() - df["FIRST_ORDER_DATE"]).dt.days + 1
df["RECENCY"] = (df["LAST_ORDER_DATE"].max() - df["LAST_ORDER_DATE"]).dt.days + 1
df["ORDER_NUM_TOTAL"] = df["ORDER_NUM_TOTAL_EVER_ONLINE"] + df["ORDER_NUM_TOTAL_EVER_OFFLINE"]
df["FREQUENCY"] = df["ORDER_NUM_TOTAL"] / df["TENURE"]
df["MONETARY"] = df["CUSTOMER_VALUE_TOTAL_EVER_OFFLINE"] + df["CUSTOMER_VALUE_TOTAL_EVER_ONLINE"]
df["AVG_ORDER_SIZE"] = df["MONETARY"] / df["ORDER_NUM_TOTAL"]
df["LAST_30"] = np.where(df["LAST_ORDER_DATE"] > df["LAST_ORDER_DATE"].max() - dt.timedelta(30), 1, 0)
df["LAST_60"] = np.where(df["LAST_ORDER_DATE"] > df["LAST_ORDER_DATE"].max() - dt.timedelta(60), 1, 0)
df["LAST_90"] = np.where(df["LAST_ORDER_DATE"] > df["LAST_ORDER_DATE"].max() - dt.timedelta(90), 1, 0)
category_transform(df, df["INTERESTED_IN_CATEGORIES_12"])

########################## Standardization  ###########################
df.info()
target_cols = [col for col in df.columns if df[col].dtype in ["float64", "int64", "int32"]]

new_df = df[target_cols]
new_df.head()

sc = MinMaxScaler((0, 1))
new_kmeans = sc.fit_transform(new_df)


##########################  Function  ###########################
def data_processing(dataframe):
    input_data = dataframe.copy()
    # Enlarging variables name
    input_data.columns = [col.upper() for col in input_data.columns]

    input_data["TENURE"] = (input_data["LAST_ORDER_DATE"].max() - input_data["FIRST_ORDER_DATE"]).dt.days + 1
    input_data["RECENCY"] = (input_data["LAST_ORDER_DATE"].max() - input_data["LAST_ORDER_DATE"]).dt.days + 1
    input_data["ORDER_NUM_TOTAL"] = input_data["ORDER_NUM_TOTAL_EVER_ONLINE"] + input_data["ORDER_NUM_TOTAL_EVER_OFFLINE"]
    input_data["FREQUENCY"] = input_data["ORDER_NUM_TOTAL"] / input_data["TENURE"]
    input_data["MONETARY"] = input_data["CUSTOMER_VALUE_TOTAL_EVER_OFFLINE"] + input_data["CUSTOMER_VALUE_TOTAL_EVER_ONLINE"]
    input_data["AVG_ORDER_SIZE"] = input_data["MONETARY"] / input_data["ORDER_NUM_TOTAL"]
    input_data["LAST_30"] = np.where(input_data["LAST_ORDER_DATE"] > input_data["LAST_ORDER_DATE"].max() - dt.timedelta(30), 1, 0)
    input_data["LAST_60"] = np.where(input_data["LAST_ORDER_DATE"] > input_data["LAST_ORDER_DATE"].max() - dt.timedelta(60), 1, 0)
    input_data["LAST_90"] = np.where(input_data["LAST_ORDER_DATE"] > input_data["LAST_ORDER_DATE"].max() - dt.timedelta(90), 1, 0)
    category_transform(input_data, input_data["INTERESTED_IN_CATEGORIES_12"])

    target_cols = [col for col in input_data.columns if input_data[col].dtype in ["float64", "int64", "int32"]]

    input_data = input_data[target_cols]

    sc = MinMaxScaler((0, 1))
    input_data = pd.DataFrame(sc.fit_transform(input_data), columns = input_data.columns)
    return input_data


input_data = data_processing(df)


######################################################
# 3. K-Means Model and Hyperparameter Optimization
######################################################
kmeans = KMeans(random_state = 42)
ssd = []
K = range(1, 30)

for k in K:
    kmeans = KMeans(n_clusters = k).fit(new_kmeans)
    ssd.append(kmeans.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()

kmeans = KMeans(random_state = 42)
elbow = KElbowVisualizer(kmeans, k = (2, 20))
elbow.fit(new_kmeans)
elbow.show()

kmeans = KMeans(n_clusters = elbow.elbow_value_, random_state = 42).fit(new_kmeans)

clusters_kmeans = kmeans.labels_

new_df["cluster"] = clusters_kmeans
new_df.head()

new_df["cluster"] = new_df.loc[:, "cluster"] + 1

##########################  K-Means Cluster Statistics ###########################
new_df.groupby("cluster").agg(["count", "mean", "median"])


##########################  Function  ###########################
def hyperparameter_opt(X, random_state= 42, k_range = (2, 20)):
    kmeans_model = KMeans(random_state = random_state)
    elbow = KElbowVisualizer(kmeans_model, k = k_range, timings = False)
    elbow.fit(X)
    kmeans_model = KMeans(n_clusters = elbow.elbow_value_, random_state = random_state).fit(X)
    return kmeans_model


kmeans_model = hyperparameter_opt(input_data)


######################################################
# 4. Hierarchic Model and Hyperparameter Optimization
######################################################
hc_average = linkage(input_data, "average")

plt.figure(figsize = (7, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           truncate_mode = "lastp",
           p = 20,
           show_contracted = True,
           leaf_font_size = 10)
plt.show()

cluster = AgglomerativeClustering(n_clusters = 10, linkage = "average")

clusters = cluster.fit_predict(new_kmeans)

new_df["hi_cluster_no"] = clusters

new_df["hi_cluster_no"] = new_df["hi_cluster_no"] + 1


##########################  Function  ###########################
def hierarchic_cluster(X,  n_clusters = 10, linkage = "average"):
    hierarchic_model = AgglomerativeClustering(n_clusters = n_clusters, linkage = linkage)
    hierarchic_model = hierarchic_model.fit(X)
    return hierarchic_model


hierarchic_model = hierarchic_cluster(input_data)
hierarchic_model.labels_


######################################################
# 5. Generate Final Dataframe
######################################################

def get_results(dataframe, kmeans_model, hierarchic_model):
    """"""
    kmeans_clusters = kmeans_model.labels_
    hierarchic_clusters = hierarchic_model.labels_
    dataframe["KMEANS_CLUSTER"] = kmeans_clusters
    dataframe["KMEANS_CLUSTER"] = dataframe.loc[:, "KMEANS_CLUSTER"] + 1
    dataframe["HIERARCHIC_CLUSTER"] = hierarchic_clusters
    dataframe["HIERARCHIC_CLUSTER"] = dataframe.loc[:, "HIERARCHIC_CLUSTER"] + 1
    kmeans_class_summary = dataframe.groupby("KMEANS_CLUSTER").agg(["count", "mean", "median"])
    hierarchic_class_summary = dataframe.groupby("HIERARCHIC_CLUSTER").agg(["count", "mean", "median"])
    return dataframe[["master_id", "KMEANS_CLUSTER", "HIERARCHIC_CLUSTER"]], kmeans_class_summary, hierarchic_class_summary


df, cs_1, cs_2 = get_results(df, kmeans_model, hierarchic_model)

