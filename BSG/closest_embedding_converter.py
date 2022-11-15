import pandas as pd
import hnswlib
import numpy as np


def compute_distance(input_data='',
                     identifier_column='',
                     distance='cosine',
                     n_outputs=300,
                     upload_file=''):
    if type(input_data)==str:
        embeddings = pd.read_csv(input_data, index_col=None)
    else:
        embeddings = input_data

    embeddings['embedding_ID'] = np.arange(len(embeddings))

    k = np.array(list(embeddings['embedding_ID'].keys()))

    v = embeddings[identifier_column].values

    sidx = k.argsort()

    k = k[sidx]
    v = v[sidx]
    embedding_dimension = v[0].shape[-1]
    num_elements = embeddings.shape[0]

    p = hnswlib.Index(space=distance, dim=embedding_dimension)

    p.init_index(max_elements=num_elements, ef_construction=200, M=16)

    p.add_items(np.stack(embeddings['embedding'].to_numpy()),
                np.array(embeddings['embedding_ID']))

    output_df = pd.DataFrame(columns=['id', 'output'])

    for x in range(len(embeddings)):
        labels, distances = p.knn_query(
            np.array(embeddings.iloc[x]['embedding_ID']).reshape(-1),
            k=n_outputs)

        article = embeddings.iloc[x][identifier_column]

        labels = labels.ravel()
        mask = k==labels
        out = np.where(mask, v, 0)
        #idx = np.searchsorted(k, labels.ravel()).reshape(labels.shape)
        #idx[idx == len(k)] = 0
        #mask = k[idx] == labels
        #out = np.where(mask, v[idx], 0)

        output_df = output_df.append([{'id': article, 'output': out}], ignore_index=True)

    output_df['output'] = output_df['output'].astype(str)

    output_df.to_csv(upload_file, index=None)
