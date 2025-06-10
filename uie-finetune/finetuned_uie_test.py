from transformers import AutoModel, AutoTokenizer
from pprint import pprint

tokenizer = AutoTokenizer.from_pretrained("./checkpoint/model_best", trust_remote_code=True)
model = AutoModel.from_pretrained("./checkpoint/model_best", trust_remote_code=True)

schema = ['颜色', '商品名称', {'商品名称': '颜色'}]
model.set_schema(schema)
pprint(model.predict(tokenizer, "小米12S Ultra 骁龙8+旗舰处理器 徕卡光学镜头 2K超视感屏 120Hz高刷 67W快充 12GB+512GB 玫瑰金 5G手机"))

schema = ['商品属性']
model.set_schema(schema)
pprint(model.predict(tokenizer, "强散热  低温高效御风而战所向披靡热空气需要有去处;因此我们设计了侧面4个排气0加0面辅助散热以尽可能提供最佳气流使其有足够的机会将热空气排出系统:我们甚至将散热器移动到更靠近通风口的位置;让热空气以直线方式从机器中排出避免任何再循环8五热管双风扇四出风口"))