import pandas as pd
from BSG_embedding_model import BSGEmbeddingModel
from BSG_utils import pre_caption
from models.tokenization_bert import BertTokenizer
import torch
from torchvision import transforms
from PIL import Image
from BSG_utils import get_image, clean_description
from tqdm import tqdm

torch.device("mps")


def embedding_generator(data: pd.DataFrame, model_path: str, bert_config_path: str, use_cuda: bool):

    # if torch.has_mps:
    #     device = torch.device('mps')
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    transform = transforms.Compose([
        transforms.Resize((384, 384), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model = BSGEmbeddingModel(text_encoder='bert-base-uncased', config_bert=bert_config_path)

    checkpoint = torch.load(model_path, map_location='cpu')
    msg = model.load_state_dict(checkpoint, strict=False)
    model.eval()

    block_num = 8

    model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.save_attention = True

    if use_cuda:
        model.cuda()
    # elif torch.has_mps: # For Mac M1 GPU support
    #     model.to(device=device)

    embedding_data = []
    data = data.head(10)
    for index, row in tqdm(data.iterrows(), total=data.shape[0]):
        product_no = row['bex_art_no']
        image_path = row['images_link']
        image_pil = get_image(image_path)
        image = transform(image_pil).unsqueeze(0)

        caption = row['description']
        caption = clean_description(caption)
        text = pre_caption(caption)
        text_input = tokenizer(text, return_tensors="pt")

        if use_cuda:
            image = image.cuda()
            text_input = text_input.to(image.device)
        # elif torch.has_mps:
        #     image = image.to(device)
        #     text_input = text_input.to(device)

        embedding = model(image, text_input)
        dictionary_data = {'product_no': product_no, 'embedding': embedding.cpu().detach().numpy().reshape(-1)}
        embedding_data.extend([dictionary_data])
    embeddings = pd.DataFrame(embedding_data)
    return embeddings