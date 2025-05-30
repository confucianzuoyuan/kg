from flask import Flask, request, jsonify, render_template

from transformers import AutoModel, AutoTokenizer
from pprint import pprint

tokenizer = AutoTokenizer.from_pretrained("../uie-finetune/checkpoint/model_best", trust_remote_code=True)
model = AutoModel.from_pretrained("../uie-finetune/checkpoint/model_best", trust_remote_code=True)

schema = ['颜色', '商品名称', {'商品名称': '颜色'}]
model.set_schema(schema)
pprint(model.predict(tokenizer, "小米12S Ultra 骁龙8+旗舰处理器 徕卡光学镜头 2K超视感屏 120Hz高刷 67W快充 12GB+512GB 玫瑰金 5G手机"))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/echo', methods=['POST'])
def echo_string():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing "text" field in JSON body'}), 400
    
    input_text = data['text']
    result = model.predict(tokenizer, input_text)
    res = {}
    res['颜色'] = [color['text'] for color in result[0]['颜色']]
    res['商品名称'] = [r['text'] for r in result[0]['商品名称'] if 'text' in r]
    res['商品的颜色'] = [result[0]['商品名称'][0]['text'] + '的颜色是：' + result[0]['商品名称'][0]['relations']['颜色'][0]['text']]
    return jsonify({'received': res})

if __name__ == '__main__':
    app.run(debug=True)
