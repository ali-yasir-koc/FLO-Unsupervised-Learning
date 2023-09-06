import pandas as pd
import helpers


def main():
    df = pd.read_csv("datasets/flo_data_20k.csv",
                     parse_dates = ["first_order_date", "last_order_date",
                                    "last_order_date_online", "last_order_date_offline"])
    input_data = helpers.data_processing(df)
    kmeans_model = helpers.hyperparameter_opt(input_data)
    hierarchic_model = helpers.hierarchic_cluster(input_data)
    result_df, kmeans_summary, hierarchic_summary = helpers.get_results(df, kmeans_model, hierarchic_model)
    result_df.to_csv("miuul_homework/FLO/FLO_result_class.csv")
    kmeans_summary.to_csv("miuul_homework/FLO/FLO_kmeans_summary.csv")
    hierarchic_summary.to_csv("miuul_homework/FLO/FLO_hierarchic_summary.csv")


print("Process is started...")
main()

