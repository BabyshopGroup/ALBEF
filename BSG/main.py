from snowflake_helper.snowflake_helper import SnowflakeHelper
from BSG.BSG_embedding_generator import embedding_generator
from BSG.closest_embedding_converter import compute_distance

snowflake_manager = SnowflakeHelper()

QUERY = 'select * from PLAYGROUNDS.ADRIANCAMPOY.CLEAN_VISIBLE_PRODUCT_INFO'
data = snowflake_manager.load_data_from_query(QUERY)
destination_embeddings_bucket = 'gs://bsg-image-recommendation/ALBEF_embeddings.csv'

model_path = '../BSG/models_downloaded/refcoco.pth'
bert_config_path = '../BSG/configs/config_bert.json'
use_cuda = False

identifier_column = 'embedding'
distance = 'cosine'
n_outputs = 5
upload_file = 'gs://bsg-image-recommendation/ALBEF_distances_per_product.csv'

embeddings = embedding_generator(data=data, model_path=model_path, bert_config_path=bert_config_path, use_cuda=use_cuda)

# embeddings.to_csv(destination_embeddings_bucket)
distances = compute_distance(input_data=embeddings,
                             identifier_column=identifier_column,
                             distance=distance,
                             n_outputs=n_outputs,
                             upload_file=upload_file)
