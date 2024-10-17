def remove_duplicates(df):
    return df.drop_duplicates()


def get_lowercase_cols(df):
    df.columns = df.columns.str.lower()
    return df



def get_column_charge(df):
    df["is_same_charge"] = df["Q1"] == df["Q2"]
    return df

def mark_outliers(df):
    import pandas as pd

    df["is_outlier"] = False

    def tukeys_test_outliers(data):
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        
        return outliers
    

    for column in df.columns:
        if column == "Event" and column != "is_same_charge" and column != "is_outlier":
            outliers = tukeys_test_outliers(df[column])
            df.loc[df[column].index.isin(outliers.index), "is_outlier"] = True

    return df


def main_cleaning(df):
    df = remove_duplicates(df)
    #get_lowercase_cols(df)
    get_column_charge(df)
    mark_outliers(df)
    return df