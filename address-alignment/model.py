from transformers import BertModel, BertForTokenClassification
import torch

bert = BertModel.from_pretrained('bert-base-chinese')

num_labels = 57

class BertForSeqTagging(BertForTokenClassification):
    def __init__(self):
        super().__init__(bert.config)
        self.num_labels = num_labels
        self.bert = bert
        self.dropout = torch.nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(
            self.bert.config.hidden_size, num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        batch_size, max_len, feature_dim = sequence_output.shape
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.view(-1, self.num_labels)[active_loss]

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            return loss
        else:
            return active_logits