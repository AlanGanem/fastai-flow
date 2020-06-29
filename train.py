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
	    'train_frac_split':0.9,
	    'pd_encoding':'utf-8',
	    'pd_sep':',',
	    'date_col':None,
	    'fastai_cycles':12,
	    'cat_emb_szs': {
	        'IVAPC':8,
	        'Fornecedor': 2,
	        'Filial':2,
	        'Material':10,
	        'UF': 5,
	        'TpImposto': 2,
	        'UFUnd':5,
	        'OrgC': 2,
	        'GCm': 2
	    },
	    'fastai_bs':512
	}




	model = ClassificationPipeline(**model_setup)

	DATA_PATH = r'mock_data/complete_dataset.csv'

	# model.fit(PATH, generate_report=False)

	export_dict_as_folder(
		model.fit(DATA_PATH, generate_validation_dict = True),
		saving_path = 'mock_data', 
		folder_name = 'complete_dataset'
		)

	# model.save(r'models/.', 'completo_menor.pkl')
	return


if __name__ == '__main__':
	train()