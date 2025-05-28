from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("xusenlin/uie-base", trust_remote_code=True)
model = AutoModel.from_pretrained("xusenlin/uie-base", trust_remote_code=True)

schema = ["时间", "选手", "赛事名称"]  # Define the schema for entity extraction
print(model.predict(tokenizer, "2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！", schema=schema))

schema = {'竞赛名称': ['主办方', '承办方', '已举办次数']}  # Define the schema for relation extraction
model.set_schema(schema)
print(model.predict(tokenizer, "2022语言与智能技术竞赛由中国中文信息学会和中国计算机学会联合主办，百度公司、中国中文信息学会评测工作委员会和中国计算机学会自然语言处理专委会承办，已连续举办4届，成为全球最热门的中文NLP赛事之一。"))

schema = {'地震触发词': ['地震强度', '时间', '震中位置', '震源深度']}  # Define the schema for event extraction
model.set_schema(schema)
print(model.predict(tokenizer, "中国地震台网正式测定：5月16日06时08分在云南临沧市凤庆县(北纬24.34度，东经99.98度)发生3.5级地震，震源深度10千米。"))

schema = {'评价维度': ['观点词', '情感倾向[正向，负向]']}  # Define the schema for opinion extraction
model.set_schema(schema)
print(model.predict(tokenizer, "店面干净，很清静，服务员服务热情，性价比很高，发现收银台有排队"))

schema = "情感倾向[正向，负向]"  # Define the schema for opinion extraction
model.set_schema(schema)
print(model.predict(tokenizer, "这个产品用起来真的很流畅，我非常喜欢"))

schema = ['法院', {'原告': '委托代理人'}, {'被告': '委托代理人'}]  # Define the schema for opinion extraction
model.set_schema(schema)
print(model.predict(tokenizer, "北京市海淀区人民法院\n民事判决书\n(199x)建初字第xxx号\n原告：张三。\n委托代理人李四，北京市 A律师事务所律师。\n被告：B公司，法定代表人王五，开发公司总经理。\n委托代理人赵六，北京市 C律师事务所律师。"))

schema = ['颜色', '商品名称', {'商品名称': '颜色'}]
model.set_schema(schema)
print(model.predict(tokenizer, "小米12S Ultra 骁龙8+旗舰处理器 徕卡光学镜头 2K超视感屏 120Hz高刷 67W快充 12GB+512GB 冷杉绿 5G手机"))