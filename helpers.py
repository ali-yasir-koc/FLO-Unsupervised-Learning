import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering
import datetime as dt

def category_transform(dataframe, cat_repo_col):
    cats = ",".join(cat_repo_col.unique()).replace("[", "").replace("]", "").split(",")
    cat_cols = list(set([i.strip() for i in cats if len(i) > 1]))
    for col in cat_cols:
        dataframe[col] = cat_repo_col.apply(lambda x: 1 if x.__contains__(str(col)) else 0)

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

def hyperparameter_opt(X, random_state= 42, k_range = (2, 20)):
    kmeans_model = KMeans(random_state = random_state)
    elbow = KElbowVisualizer(kmeans_model, k = k_range, timings = False)
    elbow.fit(X)
    kmeans_model = KMeans(n_clusters = elbow.elbow_value_, random_state = random_state).fit(X)
    return kmeans_model

def hierarchic_cluster(X,  n_clusters = 10, linkage = "average"):
    hierarchic_model = AgglomerativeClustering(n_clusters = n_clusters, linkage = linkage)
    hierarchic_model = hierarchic_model.fit(X)
    return hierarchic_model

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



