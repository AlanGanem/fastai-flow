import json
import pandas as pd
from pipeline import ClassificationPipeline
from src.data.io.saver import export_dict_as_folder


def train():

	model_setup = {
	    'date_col':'DatadoDocumento',
	    'model_id':'teste1',
	    'dependent_vars': 'IVAMIRO',
	    'cat_features': ['IVAPC','PEP','Filial','Material','UF','TpImposto','UFUnd'],
	    'num_features': [],
	    'train_frac_split':0.7,
	    'pd_encoding':'utf-8',
	    'pd_sep':',',
	    'date_col':None,
	    'fastai_cycles':12,
	    'cat_emb_szs': {
	        'IVAPC':10,
	        #'PEP': 2,
	        'Filial':10,
	        'Material':40,
	        'UF': 10,
	        #'TpImposto':,
	        'UFUnd':10
	    },
	    'fastai_bs':40
	}


	model = ClassificationPipeline(**model_setup)

	DATA_PATH = r'mock_data/history_01_2019_06_2020_W1.csv'

	export_dict_as_folder(
		model.fit(DATA_PATH, generate_validation_dict = True),
		saving_path = 'mock_data', 
		folder_name = 'complete_dataset'
		)

	model.save(r'models/.', 'teste_iva.pkl')
	return


if __name__ == '__main__':
	train()