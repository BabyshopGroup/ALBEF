from BSG.closest_embedding_converter import compute_distance
from snowflake_helper.snowflake_helper import SnowflakeHelper
from google.cloud import firestore
from BSG.BSG_embedding_generator import embedding_generator
import pandas as pd
from tqdm import tqdm
from utils import get_new_existing_products, clean_article_information

use_cuda = False
model_path = './models_downloaded/refcoco.pth'
bert_config_path = './configs/config_bert.json'
snowflake_manager = SnowflakeHelper()
QUERY_PRODUCTS_IN_CATALOGUE = 'select * from PLAYGROUNDS.ADRIANCAMPOY.VISIBLE_PRODUCTS'
products_in_catalogue = snowflake_manager.load_data_from_query(QUERY_PRODUCTS_IN_CATALOGUE)
text_column_name = 'description'  # We could use either "title" or "description" depending on the text we would like to use.
embeddings_column_name = 'ALBEFembedding'
product_column_name = 'product_no'
distance = 'euclidean'
n_outputs = 301

db = firestore.Client(project='bsg-personalization')
# Getting articles that already have embeddings
collection = db.collection(u'embeddings')
docs = collection.stream()
articles_with_embeddings_ids = []
articles_with_embeddings_info = []
for doc in docs:
    articles_with_embeddings_ids.extend([doc.id])
    info = doc.to_dict()
    info.update({"product_no": doc.id})
    articles_with_embeddings_info.extend([info])

catalogue_products = products_in_catalogue['bex_art_no'].to_list()
new_products = get_new_existing_products(catalogue_products=catalogue_products,
                                         products_with_embeddings=articles_with_embeddings_ids)

# Getting new products with new catalogues
QUERY_FOR_EMBEDDINGS = f'select * from PLAYGROUNDS.ADRIANCAMPOY.CLEAN_VISIBLE_PRODUCT_INFO ' \
                       f'where BEX_ART_NO in {tuple(new_products)}'
new_products_information = snowflake_manager.load_data_from_query(QUERY_FOR_EMBEDDINGS)
cleaned_new_products_information = clean_article_information(new_products_information, text_column_name)

if cleaned_new_products_information.shape[0] > 0:
    articles_with_embeddings_info = pd.DataFrame(articles_with_embeddings_info)
    articles_with_embeddings_info = articles_with_embeddings_info.loc[:, ['product_no', 'ALBEFembedding']].copy()
    new_article_embeddings = embedding_generator(data=cleaned_new_products_information, model_path=model_path,
                                                 text_column_name=text_column_name, bert_config_path=bert_config_path,
                                                 use_cuda=use_cuda)
    new_article_embeddings.rename(columns={"embedding": "ALBEFembedding"}, inplace=True)
    embeddings = pd.concat([articles_with_embeddings_info, new_article_embeddings], ignore_index=True)
    distances = compute_distance(input_data=embeddings,
                                 embedding_column_name=embeddings_column_name,
                                 product_no_column_name=product_column_name,
                                 metric=distance,
                                 n_outputs=n_outputs)

    batch = db.batch()
    #TODO: How to do this update in firestore allowing just not changing certain fields in a flexible way (for instance if we want to add different types of embeddings)
    for index, row in tqdm(distances.iterrows(), total=distances.shape[0]):
        article = str(row['product_no'])
        embedding = str(embeddings.loc[embeddings['product_no']==article]['ALBEFembedding'].values[0])
        closest_items = str(row['closest_items'])
        distances = str(row['distances'])
        doc_ref_collection = db.collection(u'embeddings').document(article)
        info = {
            u'ALBEFembedding': embedding,
            u'closestItems': closest_items,
            u'euclideanDistancesToClosestItems': distances
        }
        batch.set(doc_ref_collection, info)
        # batch.delete(doc_ref_collection)
        if index % 200 == 0:  # Updating in batches of 200 to avoid limitations on uploading size
            batch.commit()
            batch = db.batch()

    batch.commit()
else:
    print("No new products found")