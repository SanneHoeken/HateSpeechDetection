import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

def data_to_dataloader(filepath, tokenizer, label_encoder, batch_size, lowercase, is_test=False):

    df = pd.read_csv(filepath)
    texts = df['text'].to_list()
    if lowercase:
        texts = [text.lower() for text in texts]
    labels = list(df['label'])
    labels = torch.tensor([label_encoder[l] for l in labels])

    encodings = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    input_ids = encodings['input_ids']
    attention_masks = encodings['attention_mask']
    data = TensorDataset(input_ids, attention_masks, labels)

    if is_test:
        dataloader = DataLoader(data, batch_size=batch_size)
    else:
        sampler = RandomSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return dataloader


def train_model(model, optimizer, scheduler, train_data, test_data, epochs, device):
    
    lowest_loss = None
    best_model = None

    for epoch in range(epochs):
        print(f"Epoch {epoch+1} out of {epochs}...")
        print('Train...')
        model, avg_train_loss = train(model, optimizer, scheduler, train_data, device)
        print(f"Average training loss: {avg_train_loss}")
        print('Evaluate...')
        _, _, accuracy = evaluate(model, test_data, device)
        print(f"Accuracy: {accuracy}")
        
        if lowest_loss == None or avg_train_loss < lowest_loss:
            lowest_loss = avg_train_loss
            best_model = model
        
    return best_model


def train(model, optimizer, scheduler, train_data, device):
    model.train()
    total_loss = 0

    for batch in tqdm(train_data):
        batch = tuple(t.to(device) for t in batch)
        model.zero_grad()
        outputs = model(batch[0], attention_mask=batch[1], labels=batch[2]) #token_type_ids=None, 
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_data)
    return model, avg_train_loss
            
        
def evaluate(model, test_data, device):
    model.eval()
    preds = []
    labels = []
    for batch in tqdm(test_data):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            outputs = model(batch[0], attention_mask=batch[1]) #token_type_ids=None, 
        logits = outputs[0].detach().cpu().numpy()
        gold = batch[2].to('cpu').numpy()
        preds.extend(np.argmax(logits, axis=1).tolist())
        labels.extend(gold.tolist())
    accuracy = accuracy_score(labels, preds)

    return preds, labels, accuracy
    

def main(model_dir, output_dir, train_filepath, test_filepath, label_encoder, 
         batch_size, epochs, learning_rate, lowercase):
    
    #device = torch.device("mps")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    test_dataloader = data_to_dataloader(test_filepath, tokenizer, label_encoder, batch_size, lowercase, is_test=True)
    num_labels = len(label_encoder)
    
    train_dataloader = data_to_dataloader(train_filepath, tokenizer, label_encoder, batch_size, lowercase)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=num_labels)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    best_model = train_model(model, optimizer, scheduler, train_dataloader, test_dataloader, epochs, device)
    best_model.save_pretrained(output_dir)
    

if __name__ == '__main__':

    train_filepath = '' # path to csv with 'text' and 'label' columns
    dev_filepath = '' # path to csv with 'text' and 'label' columns
    model_dir = 'distilbert-base-german-cased' # pre-trained model, that works with Hugging Face Transformers AutoModels
                # ^^^ this is an example
    output_dir = '' # directory for fine-tuned model
    label_encoder = {"hate": 1, "neutral": 0} # labels in the test data, mapped to numerical encodings
                        # ^^^ this is an example
                        
    # HYPERPARAMETERS
    batch_size = 8
    epochs = 3
    learning_rate = 5e-5
    lowercase = False
    
    main(model_dir, output_dir, train_filepath, dev_filepath, label_encoder, 
         batch_size, epochs, learning_rate, lowercase)