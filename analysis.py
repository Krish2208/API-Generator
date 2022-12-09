import spacy
import pandas as pd
from dateutil import parser
from tqdm import tqdm

# Loads Trained Spacy Model
nlp = spacy.load("model-best")


def get_dict(entities):
    '''
    Labels and Text that parser module detects.

    Parameters
    ----------
    entities: doc.ents
        All the entities detected by Spacy NER module in the log

    Returns
    -------
    data_dict: dict
        Returns the dictionary with all the custom generated
        labels with detected text
    '''

    data_dict = {
                    "IP_ADDRESS": None, "TIMESTAMP": None, "DATE": None,
                    "TIME": None, "EMAIL": None, "URL": None, "MAC_ADDRESS": None,
                    "QUERY": None, "ERROR": None, "MESSAGE": None,
                    "HTTP_REQUEST": None, "HTTP_RESPONSE": None
                }
    for entity in tqdm(entities, leave=False):
        if entity.label_ in data_dict.keys():
            if data_dict[entity.label_] is None:
                data_dict[entity.label_] = entity.text
    
    return data_dict


def clean_df(df):
    '''
    Cleans the Dataframe

    Parameters
    ----------
    df: Pandas Dataframe
        The Dataframe which we want to clean

    Returns
    -------
    df: Pandas Dataframe
        Returns the Dataframe after extracting Date,
        Time and removing NaN value columns
    '''

    for i in range(len(df)):
        try:
            x = parser.parse(df.TIMESTAMP[i])
            df.DATE[i] = x.strftime('%d/%m/%Y')
            df.TIME[i] = x.strftime('%H:%M:%S')
            df.TIMESTAMP[i] = x.strftime('%d/%m/%Y %H:%M:%S')
        except:
            continue
    return df.loc[:, df.isna().mean()==1.0]


def get_csv(file):
    '''
    Generates the Dataframe for the Analytics Module.

    Parameters
    ----------
    file: FileObject
        The Query Log File to be Analysed

    Returns
    -------
    df: Pandas Dataframe
        Returns the complete cleaned dataframe
        with log and labelled information
    '''

    raw_logs = file.read()
    logs = raw_logs.split("\n")
    all_entities = []
    for log in tqdm(logs[:100]):
        doc = nlp(log)
        entities = get_dict(doc.ents)
        for i in entities.values():
            if i is not None:
                entities["LOG"] = log
                all_entities.append(entities)
                break
    df = pd.DataFrame(all_entities)
    df = clean_df(df)
    return df
