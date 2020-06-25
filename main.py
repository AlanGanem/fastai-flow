import pandas as pd
from pipeline import ClassificationPipeline


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
	    'fastai_bs':512
	}


	model = ClassificationPipeline(**model_setup)

	PATH = 'mock_data/clusters_0106.csv'

	model.fit(PATH)

	model.save(r'models/.', 'teste_iva.pkl')

if __name__ == '__main__':
	train()