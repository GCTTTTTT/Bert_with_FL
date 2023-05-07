import pandas as pd
import torch 
from transformers import BertTokenizer
import numpy as np
from transformers import BertModel
from tqdm import tqdm 

sample = 'Hey my name is BERT'

class Dataset(torch.utils.data.Dataset): 
    def __init__(self,df): 
        #extract our labels from the df 
        self.labels = [labels[label] for label in df["category"]]
        self.labels = torch.Tensor(self.labels).long() # todo
        #tokenize our texts to the format that BERT expects to get as input 
        self.texts = [tokenizer(text, padding='max_length', max_length=512, truncation=True,return_tensors="pt") for text in df["text"]] 
    def classes(self):
        return self.labels
    
    def __len__(self): 
        return len(self.labels)
    
    #fetch a batch of labels
    def get_batch_labels(self,indx): 
        return np.array(self.labels[indx])
    # fetch a batch of texts 
    def get_batch_texts(self,indx): 
        return self.texts[indx]

    #get an item with the texts and the label
    def __getitem__(self,indx): 
        batch_texts = self.get_batch_texts(indx)
        batch_y = self.get_batch_labels(indx)
        
        
        return batch_texts, batch_y
    
class BertClassifier(torch.nn.Module): 
    def __init__(self,dropout=0.5): 
        super(BertClassifier,self).__init__()
        
        self.bert=BertModel.from_pretrained("bert-base-cased")
        # self.bert=BertModel.from_pretrained("./bert_model/bert-base-cased")
        self.dropout = torch.nn.Dropout(dropout)
        # bert output a vector of size 768
        self.lin = torch.nn.Linear(768,5)
        self.relu = torch.nn.ReLU()
    def forward(self,input_id,mask): 
        # as output, the bert model give us first the embedding vector of all the tokens of the sequence 
        # second we get the embedding vector of the CLS token.
        # fot a classification task it's enough to use this embedding for our classifier
        _,pooled_output = self.bert(input_ids= input_id,attention_mask = mask,return_dict = False)
        dropout_output = self.dropout(pooled_output)
        linear_output  = self.lin(dropout_output)
        final_layer = self.relu(linear_output)
        
        return final_layer
    
# we are creating a standard pytorch training loop 

def train(model, train_data, val_data, learning_rate, epochs=5):
    #creating a custom Dataset objects using the training and validation data
    train, val = Dataset(train_data), Dataset(val_data)
    #creating dataloaders
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)

    if use_cuda:
            model = model.cuda()
            criterion = criterion.cuda()

    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in tqdm(train_dataloader):
                #print(f"the train input : {train_input}")
                #print(f"train label : {train_label}")

                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)
    #             print(input_id.shape)

                # get the predictions 
                output = model(input_id, mask)

                #the output is a vector of 5 values (categs)
    #             print(output)
    #             print("the output shape is" ,  output.shape)
    #             print(train_label)
                
                batch_loss = criterion(output, train_label)
                total_loss_train += batch_loss.item()
                
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc
                # updating the Gradient Descent and Backpropagation operation
                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            # now we evaluate on the validation data
            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = model(input_id, mask)

                    batch_loss = criterion(output, val_label)
                    total_loss_val += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
            
            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')

if __name__ == '__main__':
    source_url = "./bbc-text.csv"
    df = pd.read_csv(source_url)

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    # tokenizer = BertTokenizer.from_pretrained("./bert_model/bert-base-cased/vocab.txt")
    bert_input  = tokenizer(sample,padding="max_length",max_length=15,truncation=True,return_tensors="pt")

    print(bert_input["input_ids"])
    print(tokenizer.decode(bert_input["input_ids"][0] ))

    labels = {
    'business':0,
    'entertainment':1,
    'sport':2,
    'tech':3,
    'politics':4
    }

    df_train, df_valid,df_test = np.split(df.sample(frac=1,random_state=42),[int(.8*len(df)), int(.9*len(df))])

    EPOCHS = 5
    model = BertClassifier()
    learning_rate = 1e-6
    train(model, df_train, df_valid, learning_rate, EPOCHS)