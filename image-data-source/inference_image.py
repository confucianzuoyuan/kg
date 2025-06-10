from transformers import AutoModel, AutoTokenizer
from pprint import pprint

from pathlib import Path
import time


def ocr_easyocr(img):
    import easyocr
    reader = easyocr.Reader(['ch_sim'], gpu=True)
    result = reader.readtext(img)
    result_string = ""
    for t in result:
        try:
            result_string += t[1]
        except:
            pass
    return result_string


tokenizer = AutoTokenizer.from_pretrained(
    "../uie-finetune/checkpoint/model_best", trust_remote_code=True)
model = AutoModel.from_pretrained(
    "../uie-finetune/checkpoint/model_best", trust_remote_code=True)

img_path = "images/product-1/12.jpg"
ocr_text = ocr_easyocr(img_path)

schema = ['商品属性']
model.set_schema(schema)
pprint(model.predict(tokenizer, ocr_text))
