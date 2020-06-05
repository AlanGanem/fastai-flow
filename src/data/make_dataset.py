# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
#from dotenv import find_dotenv, load_dotenv
import pandas as pd
import tqdm
import numpy as np

#@click.command()
#@click.argument('input_filepath', type=click.Path(exists=True))
#@click.argument('output_filepath', type=click.Path())
def main(
        input_filepath,
        output_filepath,
):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    data = pd.read_csv(input_filepath, sep = ',', encoding = 'utf-8')
    data = data.astype(str)

    data = data.drop_duplicates(subset=['Material', 'Nºdopedido', 'IVAMIRO'])
    data = data.astype(str)
    strip_columns = [
        'Material',
        'Filial',
        'IVAPC',
        'PEP',
        'Fornecedor',
        'Contrato',
        'UF',
        'TpImposto',
        'IVAMIRO',
        'Nºdopedido'
    ]
    for col in tqdm.tqdm(strip_columns):
        try:
            data[col] = data[col].str.strip(' ').str.lstrip('0')
        except:
            #change to warn in the future
            print('{} not in data.columns'.format(col))

    #data = data.replace({'': np.nan})
    #data = data.replace({'nan': np.nan})
    data['OrderType'] = data['Nºdopedido'].str[0:2]
    data = data.replace(to_replace = {'nan': ''})
    data.to_csv(output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    #project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())
    today = int(pd.to_datetime('today').timestamp())
    main(
        r'C:\Users\User Ambev\Desktop\Célula de analytics\Projetos\iva-apfj\data\external\history.csv',
        r'C:\Users\User Ambev\Desktop\Célula de analytics\Projetos\iva-apfj\data\external\hist_prep.csv'#.format(today)
    )
