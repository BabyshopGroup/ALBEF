import pandas as pd
import numpy as np


def get_new_existing_products(catalogue_products: list, products_with_embeddings: list) -> list:
    catalogue_products = set(catalogue_products)
    products_with_embeddings = set(products_with_embeddings)
    new_products = catalogue_products.difference(products_with_embeddings)
    return list(new_products)


def clean_article_information(data: pd.DataFrame, text_column_name: str) -> pd.DataFrame:
    data.fillna(value=np.nan)
    data_without_description_or_image = data.loc[(data[text_column_name].isnull()) | (data['images_link'].isnull())]
    data_without_description_or_image.to_csv('data_without_description_or_image.csv')
    new_products_information = data[
        ~data['bex_art_no'].isin(data_without_description_or_image['bex_art_no'].to_list())]
    return new_products_information
