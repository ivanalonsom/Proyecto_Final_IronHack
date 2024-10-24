def remove_duplicates(df):
    """
    This function takes a pandas DataFrame as input and returns a new DataFrame with duplicate rows removed.

    Parameters
    ----------
    df : pandas DataFrame
        The DataFrame to be cleaned.

    Returns
    -------
    cleaned_df : pandas DataFrame
        The DataFrame after removing duplicates.
    """
    return df.drop_duplicates()


def get_lowercase_cols(df):
    """
    This function takes a pandas DataFrame as input and returns a new DataFrame with all column names in lower case.

    Parameters
    ----------
    df : pandas DataFrame
        The DataFrame to be cleaned.

    Returns
    -------
    cleaned_df : pandas DataFrame
        The DataFrame with all column names in lower case.
    """
    df.columns = df.columns.str.lower()
    return df



def get_column_charge(df):
    """
    This function takes a pandas DataFrame as input and adds a new column named "is_same_charge". This column will be a boolean indicating whether the charges of the two particles are the same or not.

    Parameters
    ----------
    df : pandas DataFrame
        The DataFrame to be cleaned.

    Returns
    -------
    cleaned_df : pandas DataFrame
        The DataFrame with a new column named "is_same_charge".
    """
    df["is_same_charge"] = df["Q1"] == df["Q2"]
    return df


def mark_outliers(df):
    """
    This function takes a pandas DataFrame as input and marks outliers in all columns except "Event", "is_same_charge", and "is_outlier" by setting the "is_outlier" column to True for outliers and False otherwise.

    Parameters
    ----------
    df : pandas DataFrame
        The DataFrame to be cleaned.

    Returns
    -------
    cleaned_df : pandas DataFrame
        The DataFrame with outliers marked.
    """
    import pandas as pd

    df["is_outlier"] = False

    def tukeys_test_outliers(data):
        """
        Identify outliers in a pandas Series using Tukey's method.

        This function calculates the first quartile (Q1), third quartile (Q3), 
        and the interquartile range (IQR) of the data. Outliers are defined as 
        data points that are below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR.

        Parameters
        ----------
        data : pandas Series
            The data for which outliers need to be identified.

        Returns
        -------
        pandas Series
            A Series containing the outliers in the data.
        """
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
    """
    This function takes a pandas DataFrame as input and performs the following operations on it:
        1. Removes duplicate rows from the DataFrame.
        2. Converts all column names to lower case.
        3. Adds a new column named "is_same_charge". This column will be a boolean indicating whether the charges of the two particles are the same or not.
        4. Marks outliers in all columns except "Event", "is_same_charge", and "is_outlier" by setting the "is_outlier" column to True for outliers and False otherwise.

    Parameters
    ----------
    df : pandas DataFrame
        The DataFrame to be cleaned.

    Returns
    -------
    cleaned_df : pandas DataFrame
        The DataFrame after performing the above operations.
    """
    df = remove_duplicates(df)
    #get_lowercase_cols(df)
    get_column_charge(df)
    mark_outliers(df)
    return df


def import_to_sql(df, name):

    """
    This function imports the DataFrame of video game deals to a SQL database.
    
    Parameters:
        df (DataFrame): The DataFrame containing the video game deal data.
        name (str): The name of the database to be created or used.

    """

    import pandas as pd
    from sqlalchemy import create_engine, text
    import pymysql
    import os
    from dotenv import load_dotenv

    bbdd_name = os.getenv("bbdd_name")
    passBD = os.getenv("passBD")

    # Tus parámetros de conexión
    bd = bbdd_name
    password = passBD 

    connection_string = 'mysql+pymysql://root:' + password + '@localhost/' + bd
    engine = create_engine(connection_string)

    # Enviar DataFrame a MySQL

    df.to_sql(name, con=engine, if_exists='append', index=False)


def make_query(query):
    """
    This function makes a query to a SQL database and returns the result.
    
    Parameters:
        query (str): The query to be executed.
    
    Returns:
        query_result (DataFrame): The result of the query.
    """
    import pandas as pd
    from sqlalchemy import create_engine
    import os
    from dotenv import load_dotenv

    # Cargar las variables de entorno
    load_dotenv()

    bbdd_name = os.getenv("bbdd_name")
    passBD = os.getenv("passBD")

    # Crear la cadena de conexión
    connection_string = f'mysql+pymysql://root:{passBD}@localhost/{bbdd_name}'
    engine = create_engine(connection_string)

    query_result = pd.read_sql(query, engine)

    return query_result
