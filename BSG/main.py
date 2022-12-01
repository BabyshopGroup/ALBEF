from snowflake_helper.snowflake_helper import SnowflakeHelper
from BSG.BSG_embedding_generator import embedding_generator
from BSG.closest_embedding_converter import compute_distance
from utils import get_new_existing_products, clean_article_information
import pandas as pd

snowflake_manager = SnowflakeHelper()

QUERY_PRODUCTS_IN_CATALOGUE = 'select * from PLAYGROUNDS.ADRIANCAMPOY.VISIBLE_PRODUCTS'
products_in_catalogue = snowflake_manager.load_data_from_query(QUERY_PRODUCTS_IN_CATALOGUE)
destination_embeddings_bucket = 'gs://bsg-image-recommendation/ALBEF_embeddings.csv'

model_path = './models_downloaded/refcoco.pth'
bert_config_path = './configs/config_bert.json'
use_cuda = False

text_column_name = 'description'  # We could use either "title" or "description" depending on the text we would like to use.
embeddings_column_name = 'embedding'
product_column_name = 'product_no'
distance = 'cosine'
n_outputs = 300
upload_file_closest_articles = 'gs://bsg-image-recommendation/ALBEF/ALBEF_closest_articles_per_product.pickle'
upload_file_embeddings = 'gs://bsg-image-recommendation/ALBEF/ALBEF_embeddings_of_products.pickle'

# Getting articles that already have embeddings
articles_with_embeddings = pd.read_pickle(upload_file_embeddings)

catalogue_products = products_in_catalogue['bex_art_no'].to_list()
products_with_embeddings = articles_with_embeddings['product_no'].to_list()
new_products = get_new_existing_products(catalogue_products=catalogue_products,
                                         products_with_embeddings=products_with_embeddings)

# Getting new products with new catalogues
QUERY_FOR_EMBEDDINGS = f'select * from PLAYGROUNDS.ADRIANCAMPOY.CLEAN_VISIBLE_PRODUCT_INFO ' \
                       f'where BEX_ART_NO in {tuple(new_products)}'
new_products_information = snowflake_manager.load_data_from_query(QUERY_FOR_EMBEDDINGS)

# Getting products without description or image #TODO: Find better way to deal with images without description or images.
cleaned_new_products_information = clean_article_information(new_products_information, text_column_name)

if cleaned_new_products_information.shape[0] > 0:
    new_article_embeddings = embedding_generator(data=cleaned_new_products_information, model_path=model_path,
                                     text_column_name=text_column_name, bert_config_path=bert_config_path,
                                     use_cuda=use_cuda)

    embeddings = pd.concat([articles_with_embeddings, new_article_embeddings], ignore_index=True)
    embeddings.to_pickle(upload_file_embeddings)

    # Compute distances
    distances = compute_distance(input_data=embeddings,
                                 embedding_column_name=embeddings_column_name,
                                 product_no_column_name=product_column_name,
                                 metric=distance,
                                 n_outputs=n_outputs)

    distances.to_pickle(upload_file_closest_articles)


else:
    print("No new products found")
