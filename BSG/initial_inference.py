from snowflake_helper.snowflake_helper import SnowflakeHelper
from BSG.BSG_embedding_generator import embedding_generator
from BSG.closest_embedding_converter import compute_distance
import numpy as np
snowflake_manager = SnowflakeHelper()

QUERY = 'select * from PLAYGROUNDS.ADRIANCAMPOY.CLEAN_VISIBLE_PRODUCT_INFO'
data = snowflake_manager.load_data_from_query(QUERY)
destination_embeddings_bucket = 'gs://bsg-image-recommendation/ALBEF_embeddings.csv'

model_path = './models_downloaded/refcoco.pth'
bert_config_path = './configs/config_bert.json'
use_cuda = False

text_column_name = 'title'
embeddings_column_name = 'embedding'
product_column_name = 'product_no'
distance = 'cosine'
n_outputs = 300
upload_file_closest_articles = 'gs://bsg-image-recommendation/ALBEF/ALBEF_closest_articles_per_product_test.csv'
upload_file_embeddings = 'gs://bsg-image-recommendation/ALBEF/ALBEF_embeddings_of_products_test.csv'

# Getting products without description or image
data.fillna(value=np.nan)
data_without_description_or_image = data.loc[(data[text_column_name].isnull()) | (data['images_link'].isnull())]
data_without_description_or_image.to_csv('data_without_description_or_image.csv')
data = data[~data['bex_art_no'].isin(data_without_description_or_image['bex_art_no'].to_list())]

embeddings = embedding_generator(data=data, model_path=model_path, text_column_name=text_column_name, bert_config_path=bert_config_path, use_cuda=use_cuda)
embeddings.to_csv(upload_file_embeddings, index=None)

distances = compute_distance(input_data=embeddings,
                             embedding_column_name=embeddings_column_name,
                             product_no_column_name=product_column_name,
                             metric=distance,
                             n_outputs=n_outputs)

distances.to_csv(upload_file_closest_articles, index=None)
