# fine_tune_model.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import json

label_mapping = {
    "Низкий": 0,
    "Ниже Среднего": 1,
    "Средний": 2,
    "Выше Среднего": 3,
    "Высокий": 4
}

def prepare_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    texts = []
    labels = []
    for item in data:
        texts.append(item['text'])
        risk_level = item['errors'][0]['risk_level']
        labels.append(label_mapping[risk_level])
    return texts, labels

def fine_tune_model(train_file, model_name="bert-base-uncased"):
    texts, labels = prepare_data(train_file)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_mapping))

    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
    dataset = [{'input_ids': encodings['input_ids'][i], 'attention_mask': encodings['attention_mask'][i], 'labels': labels[i]} for i in range(len(labels))]

    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = CustomDataset(encodings, labels)

    training_args = TrainingArguments(
        output_dir='./models/results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    model.save_pretrained('./models/fine-tuned-risk-model')
    tokenizer.save_pretrained('./models/fine-tuned-risk-model')

# Запуск дообучения модели
train_file = 'data/Contract_check.json'
fine_tune_model(train_file)
