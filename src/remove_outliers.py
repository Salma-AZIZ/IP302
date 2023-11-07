# Function to remove outliers using IQR for a specific list of features
def remove_outliers(df, feature_list):
    Q1 = df[feature_list].quantile(0.25)
    Q3 = df[feature_list].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df[feature_list] < (Q1 - 1.5 * IQR)) |(df[feature_list] > (Q3 + 1.5 * IQR))).any(axis=1)]
    return df