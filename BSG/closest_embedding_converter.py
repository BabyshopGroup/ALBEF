import pandas as pd
import pynndescent
import numpy as np
from tqdm import tqdm


def compute_distance(input_data,
                     embedding_column_name: str,
                     product_no_column_name: str,
                     metric: str = 'cosine',
                     n_outputs: int = 300) -> pd.DataFrame:
    if type(input_data) == str:
        embeddings = pd.read_csv(input_data, index_col=None)
    else:
        embeddings = input_data

    data = np.stack(embeddings[embedding_column_name].to_numpy())
    index = pynndescent.NNDescent(data, metric=metric)
    index.prepare()

    product_numbers = embeddings[product_no_column_name].to_list()
    output_df = pd.DataFrame(columns=['id', 'closest_items', 'distances'])
    for x in tqdm(range(len(embeddings))):
        neighbors, distances = index.query(embeddings.loc[x, embedding_column_name].reshape(1, -1), k=n_outputs)
        neighbors = neighbors[0].tolist()
        neighbors = neighbors[1:]  # We remove the first element since it is the article we are querying itself.
        # Not necessary if the "index" has not been trained with the vector with which we are querying.
        distances = distances[0].tolist()
        distances = distances[1:]
        closest_products = [product_numbers[x] for x in
                            neighbors]  # TODO: figure out a way to do this step more efficient
        product = embeddings.loc[x, product_no_column_name]
        new_output = pd.DataFrame([{'id': product, 'closest_items': closest_products, 'distances': distances}])
        output_df = pd.concat([output_df, new_output], ignore_index=True)

    output_df['closest_items'] = output_df['closest_items'].astype(str)
    return output_df
