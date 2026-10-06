"""
This module contains all the functions for handling the data like downloading the data, loading the data, saving the data, and sanitizing the data.
"""

import logging
import pandas as pd
import dill  # for saving pipeline

from Prediction_Model import config

def download_and_load_data(url : str = config.URL,file_name='') -> pd.DataFrame:
    """
    Downloads and loads the data from the specified url, saving it to a local csv file.
    
    Args:
        url (str): The url of the data file to download. Defaults to config.URL.
        file_name (str): The name of the local csv file to save the data to. Defaults to the name of the file from the url.
    
    Returns:
        pd.DataFrame: The loaded and sanitized data.
    """
    if not url:
        raise ValueError('url must be specified in config.py or passed as an argument')
    logging.info('Downloading data from %s', url)

    df = pd.read_csv(url).rename(lambda x: x.lower()
                                        .strip()
                                        .replace(' ', '_'),
                                        axis='columns')
    file_name = file_name or f"/{url.split('/')[-1]}" # fallback to name from url if not specified
    config.FILE_NAME = file_name
    save_data(df, file_name)
    return df

def load_data_and_sanitize(file_name : str = config.FILE_NAME) -> pd.DataFrame:
    """
    Loads data from a local csv file and sanitizes it by converting column names to lower snake case.

    Args:
        file_name (str): The name of the local csv file to load. Defaults to config.FILE_NAME.

    Returns:
        pd.DataFrame: The loaded and sanitized data.
    """
    if file_name.split('.')[1] != 'csv':
        raise ValueError('file_name must be a csv')
    logging.info('Ingesting data from %s', file_name)
    return pd.read_csv(f'{config.PARENT_ABS_PATH}/data/{file_name}').rename(lambda x: x.lower() # this module is imported in files with CWD as root thus '/data'
                                                                                        .strip()
                                                                                        .replace(' ', '_'),
                                                                            axis='columns')

def save_data(df : pd.DataFrame,file_name : str) -> None:
    """
    Saves a pandas DataFrame to a local csv file.
    
    Args:
        df (pd.DataFrame): The DataFrame to save.
        file_name (str): The name of the local csv file to save the DataFrame to.
    """
    logging.info('Saving data to %s', file_name)
    df.to_csv(f'{config.PARENT_ABS_PATH}/data/{file_name}', index=False) # this module is imported in files with CWD as root thus '/data'

def save_pipeline(pipeline, pipe_name: str) -> None:
    """
    Saves a pipeline to a pickle file.
    
    Args:
        pipeline (Pipeline): The pipeline to save.
    """
    logging.info('Saving pipeline to trained_models folder')
    with open(f'{config.PARENT_ABS_PATH}/Prediction_Model/trained_models/{pipe_name}.pkl', 'wb') as f:
        dill.dump(pipeline, f)
    print(f'Saved pipeline to trained_models/{pipe_name}.pkl')
    

def load_pipeline(pipe_name: str):
    """
    Loads a saved pipeline from a pickle file.

    Args:
        pipe_name (str): The name of the pipeline to load.

    Returns:
        Pipeline: The loaded pipeline.
    """
    with open(f'{config.PARENT_ABS_PATH}/Prediction_Model/trained_models/{pipe_name}.pkl', 'rb') as f:
        pipe = dill.load(f)
    return pipe