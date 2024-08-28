import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader

class SeqDataset(Dataset):
    def __init__(self, posts, labels, tokenizer, max_len, time=None, id=None):
        self.tokenizer = tokenizer
        self.post_texts = posts
        self.targets = labels
        self.max_len = max_len
        self.time = time
        self.id = id

    def __len__(self):
        return len(self.post_texts)

    def __getitem__(self, index):
        zero_mask = self.post_texts[index]==0
        post_text = ['' if p==0  else " ".join(str(p).split()) for p in self.post_texts[index]]

        inputs = self.tokenizer.batch_encode_plus(
            post_text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        if (self.time==None):
            return (torch.tensor(ids, dtype=torch.long),
                            torch.tensor(mask, dtype=torch.long),
                            torch.tensor(token_type_ids, dtype=torch.long),
                            torch.tensor(zero_mask, dtype=torch.bool),
                            self.id[index],
                            self.targets[index]
                            )
        else:
            return (torch.tensor(ids, dtype=torch.long),
                    torch.tensor(mask, dtype=torch.long),
                    torch.tensor(token_type_ids, dtype=torch.long),
                    torch.tensor(zero_mask, dtype=torch.bool),
                    self.time[index],
                    self.id[index],
                    self.targets[index]
                    )


def get_bert_embeddings(text, labels, model_name, device, MAX_LEN=512, TRAIN_BATCH_SIZE=8):
    #Load the tokenizer and the pre-trained model
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name,
                                    output_hidden_states = True, # Whether the model returns all hidden-states.
                                    )
    model.to(device)

    params = {'batch_size': TRAIN_BATCH_SIZE,'shuffle': False,'num_workers': 32}
    training_set = CustomDataset(text, labels, tokenizer, MAX_LEN)
    training_loader = DataLoader(training_set, **params)

    model.eval()

    reps = torch.Tensor()
    masks = torch.Tensor()

    with torch.no_grad():
        for _, data in enumerate(training_loader, 0):
            ids = data[0].to(device, dtype = torch.long)
            mask = data[1].to(device, dtype = torch.long)
            token_type_ids = data[2].to(device, dtype = torch.long)
            representation = model(ids, mask, token_type_ids)
            representation = representation['last_hidden_state'].cpu().detach()

            reps = torch.cat((reps, representation), 0)
            masks = torch.cat((masks, data[1]), 0)

    return reps, masks