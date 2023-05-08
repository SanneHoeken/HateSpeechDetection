import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
from train_transformer import data_to_dataloader, evaluate

def main(model_dir, output_file, test_filepath, label_encoder, batch_size, lowercase):   

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    test_dataloader = data_to_dataloader(test_filepath, tokenizer, label_encoder, batch_size, lowercase, is_test=True)
    num_labels = len(label_encoder)

    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=num_labels)
    print('Evaluate...')
    preds, labels, _ = evaluate(model, test_dataloader, device)
    label_decoder = {encoding: label for label, encoding in label_encoder.items()}
    preds = [label_decoder[p] for p in preds]
    labels = [label_decoder[l] for l in labels]
    print(classification_report(labels, preds))
    #print(confusion_matrix(labels, preds))
    
    test_df = pd.read_csv(test_filepath)
    test_df['prediction'] = pd.Series(preds)
    test_df.to_csv(output_file, index=False)


if __name__ == '__main__':

    test_filepath = '' #csv with 'text' and 'label' columns
    output_filepath = '' #csv
    tokenizer_dir = 'distilbert-base-german-cased' # pre-trained model for tokenization, that works with Hugging Face Transformers AutoModels
                    # ^^^ this is an example
    model_dir = '' # directory of fine-tuned model
    label_encoder = {"hate": 1, "neutral": 0} # labels in the test data, and their encodings used in training
                    # ^^^ this is an example
    batch_size = 8
    lowercase = False

    main(model_dir, output_filepath, test_filepath, label_encoder, batch_size, lowercase)