import pandas as pd
import pynndescent
from tqdm import tqdm
import ast
import numpy as np


def check_and_convert_data_type(data):
    data_list = []
    for element in data:
        if type(element) == str:
            data_list.extend([ast.literal_eval(element)])
        else:
            data_list.extend([element])
    return data_list


def compute_distance(input_data,
                     embedding_column_name: str,
                     product_no_column_name: str,
                     metric: str = 'cosine',
                     n_outputs: int = 300) -> pd.DataFrame:
    if type(input_data) == str:
        embeddings = pd.read_csv(input_data, index_col=None)
    else:
        embeddings = input_data
    data = embeddings[embedding_column_name].to_numpy()
    data = check_and_convert_data_type(data=data)
    data = np.stack(data)
    index = pynndescent.NNDescent(data, metric=metric)
    index.prepare()

    product_numbers = embeddings[product_no_column_name].to_list()
    output_df = pd.DataFrame(columns=['product_no', 'closest_items', 'distances'])
    for x in tqdm(range(len(embeddings))):
        neighbors, distances = index.query(data[x, :].reshape(1, -1), k=n_outputs)
        neighbors = neighbors[0].tolist()
        neighbors = neighbors[1:]  # We remove the first element since it is the article we are querying itself.
        # Not necessary if the "index" has not been trained with the vector with which we are querying.
        distances = distances[0].tolist()
        distances = distances[1:]
        closest_products = [product_numbers[x] for x in
                            neighbors]
        product = embeddings.loc[x, product_no_column_name]
        new_output = pd.DataFrame([{'product_no': product, 'closest_items': closest_products, 'distances': distances}])
        output_df = pd.concat([output_df, new_output], ignore_index=True)#todo: how to avoid this concat and do extend or stwack instead

    output_df['closest_items'] = output_df['closest_items'].astype(str)
    return output_df
