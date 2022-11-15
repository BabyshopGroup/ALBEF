import re
from bs4 import BeautifulSoup
import numpy as np
from skimage import io, transform as skimage_transform
from scipy.ndimage import filters
from matplotlib import pyplot as plt
import requests
import shutil
from PIL import Image


# TEXT PROCESSING
def pre_caption(caption: str, max_words: int=30) -> str:
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])
    return caption


def text_cleaning(text: str) -> str:
    text = text.replace("–", " ")
    text = text.replace("°", " ")
    text = text.replace("\n", "")
    text = text.replace("®", " ")
    text = text.replace("  ", "")
    return text


def clean_description(description: str) -> str:
    soup = BeautifulSoup(description, features="html.parser")
    text = soup.get_text()
    text = text_cleaning(text)
    return text


# IMAGE PROCESSING
def getAttMap(img, attMap, blur=True, overlap=True):
    attMap -= attMap.min()
    if attMap.max() > 0:
        attMap /= attMap.max()
    attMap = skimage_transform.resize(attMap, (img.shape[:2]), order=3, mode='constant')
    if blur:
        attMap = filters.gaussian_filter(attMap, 0.02 * max(img.shape[:2]))
        attMap -= attMap.min()
        attMap /= attMap.max()
    cmap = plt.get_cmap('jet')
    attMapV = cmap(attMap)
    attMapV = np.delete(attMapV, 3, 2)
    if overlap:
        attMap = 1 * (1 - attMap ** 0.7).reshape(attMap.shape + (1,)) * img + (attMap ** 0.7).reshape(
            attMap.shape + (1,)) * attMapV
    return attMap


def get_image(image_url: str):
    r = requests.get(image_url, stream=True)
    if r.status_code == 200:
        with open("../../img.png", 'wb') as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)
    im = Image.open("../../img.png").convert('RGB')

    return im
