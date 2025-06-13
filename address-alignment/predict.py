from model import BertForSeqTagging
import torch
import torch.nn.functional as F
from transformers import BertTokenizer

labels = [
    'O', 'B-assist', 'I-assist', 'S-assist', 'E-assist', 'B-cellno', 'I-cellno', 'E-cellno', 'B-city', 'I-city', 'E-city', 'B-community', 'I-community', 'E-community', 'B-devzone', 'I-devzone', 'E-devzone', 'B-district', 'I-district', 'E-district', 'B-floorno', 'I-floorno', 'E-floorno', 'B-houseno', 'I-houseno', 'E-houseno', 'B-poi', 'I-poi', 'S-poi', 'E-poi', 'B-prov', 'I-prov', 'E-prov', 'B-road', 'I-road', 'E-road', 'B-roadno', 'I-roadno', 'E-roadno', 'B-subpoi', 'I-subpoi', 'E-subpoi', 'B-town', 'I-town', 'E-town', 'B-intersection', 'I-intersection', 'S-intersection', 'E-intersection', 'B-distance', 'I-distance', 'E-distance', 'B-village_group', 'I-village_group', 'E-village_group'
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cls_token = '[CLS]'
eos_token = '[SEP]'
unk_token = '[UNK]'
pad_token = '[PAD]'
mask_token = '[MASK]'

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')


# 加载训练好的模型
model = BertForSeqTagging()
model.to(device)
model.load_state_dict(torch.load('bert.pkl', map_location=torch.device('cpu')))

def print_address_info(address, model):
    input_token = [cls_token] + list(address) + [eos_token]
    input_ids = tokenizer.convert_tokens_to_ids(input_token)
    attention_mask = [1] * (len(address) + 2)
    ids = torch.LongTensor([input_ids])
    atten_mask = torch.LongTensor([attention_mask])
    # x = model(ids, atten_mask)
    ids = ids.to(device)
    atten_mask= atten_mask.to(device)
    logits = model(ids, atten_mask)
    logits = F.softmax(logits, dim=-1)
    logits = logits.data.cpu()
    rr = torch.argmax(logits, dim=1)
    # print(rr)
    import collections
    r = collections.defaultdict(list)
    for i, x in enumerate(rr.numpy().tolist()[1:-1]):
        print(address[i], labels[x])
        r[labels[x][2:]].append(address[i])

    return r

r = print_address_info('广东省汕头市龙湖区黄山路30号荣兴大厦', model)
print(r)