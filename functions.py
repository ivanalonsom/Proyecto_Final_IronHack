def remove_duplicates(df):
    return df.drop_duplicates()


def get_lowercase_cols(df):
    df.columns = df.columns.str.lower()
    return df



def get_column_charge(df):
    df["is_same_charge"] = df["Q1"] == df["Q2"]
    return df


def main_cleaning(df):
    df = remove_duplicates(df)
    #get_lowercase_cols(df)
    get_column_charge(df)
    return df