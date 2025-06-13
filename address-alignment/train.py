import time
import torch
import torch.nn.functional as F
from tqdm import tqdm

from transformers import BertTokenizer, get_linear_schedule_with_warmup

from model import BertForSeqTagging

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_train_epochs = 10
warmup_proportion = 0.05

gradient_accumulation_steps = 1
train_batch_size = 32
valid_batch_size = 32
test_batch_size = 32

data_workers = 2
save_checkpoint = False

learning_rate = 5e-5
weight_decay = 0.01
max_grad_norm = 1.0

cls_token = '[CLS]'
eos_token = '[SEP]'
unk_token = '[UNK]'
pad_token = '[PAD]'
mask_token = '[MASK]'

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 标签

labels = [
    'O', 'B-assist', 'I-assist', 'S-assist', 'E-assist', 'B-cellno', 'I-cellno', 'E-cellno', 'B-city', 'I-city', 'E-city', 'B-community', 'I-community', 'S-community', 'E-community', 'B-devzone', 'I-devzone', 'E-devzone', 'B-district', 'I-district', 'S-district', 'E-district', 'B-floorno', 'I-floorno', 'E-floorno', 'B-houseno', 'I-houseno', 'E-houseno', 'B-poi', 'I-poi', 'S-poi', 'E-poi', 'B-prov', 'I-prov', 'E-prov', 'B-road', 'I-road', 'E-road', 'B-roadno', 'I-roadno', 'E-roadno', 'B-subpoi', 'I-subpoi', 'E-subpoi', 'B-town', 'I-town', 'E-town', 'B-intersection', 'I-intersection', 'S-intersection', 'E-intersection', 'B-distance', 'I-distance', 'E-distance', 'B-village_group', 'I-village_group', 'E-village_group'
]
print(len(labels))

label2id = {}
for i, l in enumerate(labels):
    label2id[l] = i
num_labels = len(labels)
print(num_labels)


# 载入数据
f_train = open('./data/train.txt')

f_dev = open('./data/dev.txt')


def get_data_list(f):
    data_list = []
    content = f.read()
    blocks = content.split("\n\n")
    for b in blocks:
        words = []
        labels = []
        lines = b.split('\n')
        for line in lines:
            try:
                word, label = line.split(' ')
                words.append(word)
                labels.append(label2id[label])
            except:
                print(line)
                pass
        if len(words) != len(labels):
            print(words, labels)
        data_list.append([words, labels])
    return data_list


train_list = get_data_list(f_train)
dev_list = get_data_list(f_dev)
print(len(train_list), len(dev_list))
max_token_len = 0
for ls in [train_list, dev_list]:
    for l in ls:
        max_token_len = max(max_token_len, len(l[0]))
print('max_token_len', max_token_len)


class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        sentence = example[0]
        label = example[1]

        sentence_len = len(sentence)
        pad_len = max_token_len - sentence_len

        total_len = sentence_len + 2

        input_token = [cls_token] + sentence + \
            [eos_token] + [pad_token] * pad_len

        input_ids = tokenizer.convert_tokens_to_ids(input_token)
        attention_mask = [1] + [1] * sentence_len + [1] + [0] * pad_len
        label = [-100] + label + [-100] + [-100] * pad_len

        assert max_token_len + \
            2 == len(input_ids) == len(attention_mask) == len(input_token)

        return input_ids, attention_mask, total_len, label, index


def the_collate_fn(batch):
    total_lens = [b[2] for b in batch]
    total_len = max(total_lens)
    input_ids = torch.LongTensor([b[0] for b in batch])
    attention_mask = torch.LongTensor([b[1] for b in batch])
    label = torch.LongTensor([b[3] for b in batch])
    input_ids = input_ids[:, :total_len]
    attention_mask = attention_mask[:, :total_len]
    label = label[:, :total_len]

    indexs = [b[4] for b in batch]

    return input_ids, attention_mask, label, indexs


train_dataset = MyDataSet(train_list)
train_data_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=train_batch_size,
    shuffle=True,
    num_workers=data_workers,
    collate_fn=the_collate_fn,
)


def the_collate_fn(batch):
    total_lens = [b[2] for b in batch]
    total_len = max(total_lens)
    input_ids = torch.LongTensor([b[0] for b in batch])
    attention_mask = torch.LongTensor([b[1] for b in batch])
    label = torch.LongTensor([b[3] for b in batch])
    input_ids = input_ids[:, :total_len]
    attention_mask = attention_mask[:, :total_len]
    label = label[:, :total_len]

    indexs = [b[4] for b in batch]

    return input_ids, attention_mask, label, indexs


train_dataset = MyDataSet(train_list)
train_data_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=train_batch_size,
    shuffle=True,
    num_workers=data_workers,
    collate_fn=the_collate_fn,
)


def log(msg):
    print(msg)


model = BertForSeqTagging()
model.to(device)
t_total = len(train_data_loader) // gradient_accumulation_steps * \
    max_train_epochs + 1

num_warmup_steps = int(warmup_proportion * t_total)
log('warmup steps : %d' % num_warmup_steps)

# no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
no_decay = ['bias', 'LayerNorm.weight']
param_optimizer = list(model.named_parameters())
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(
        nd in n for nd in no_decay)], 'weight_decay': weight_decay},
    {'params': [p for n, p in param_optimizer if any(
        nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                              lr=learning_rate)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total)

# 训练
for epoch in range(max_train_epochs):
    # train
    epoch_loss = None
    epoch_step = 0
    start_time = time.time()
    model.train()
    for step, batch in enumerate(tqdm(train_data_loader)):
        input_ids, attention_mask, label = (b.to(device) for b in batch[:-1])
        loss = model(input_ids, attention_mask, label)
        loss.backward()
        #         torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if epoch_loss is None:
            epoch_loss = loss.item()
        else:
            epoch_loss = 0.98 * epoch_loss + 0.02 * loss.item()
        epoch_step += 1

    used_time = (time.time() - start_time) / 60
    log('Epoch = %d Epoch Mean Loss %.4f Time %.2f min' %
        (epoch, epoch_loss, used_time))
    torch.save(model.state_dict(), 'bert.pkl')
