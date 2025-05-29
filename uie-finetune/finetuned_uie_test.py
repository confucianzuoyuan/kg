from transformers import AutoModel, AutoTokenizer
from pprint import pprint

tokenizer = AutoTokenizer.from_pretrained("./checkpoint/model_best", trust_remote_code=True)
model = AutoModel.from_pretrained("./checkpoint/model_best", trust_remote_code=True)

schema = ['颜色', '商品名称', {'商品名称': '颜色'}]
model.set_schema(schema)
pprint(model.predict(tokenizer, "小米12S Ultra 骁龙8+旗舰处理器 徕卡光学镜头 2K超视感屏 120Hz高刷 67W快充 12GB+512GB 玫瑰金 5G手机"))