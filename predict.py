from pipeline import ClassificationPipeline


def predict(data_path):

	load_model = ClassificationPipeline.load(r'models/teste_iva.pkl')
	predicted_df = load_model.predict(data_path)

	predicted_df.to_csv(path)

	return

if __name__ == '__main__':
	predict(data_path=r'mock_data/recent_history_1806.csv')