#packages
import os
import glob
import pandas as pd
import joblib
import pandas as pd
import numpy as np
import re
import spacy
my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@'
import string
#finding the terms in the response
from nltk.stem import PorterStemmer
import string
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
word_rooter = nltk.stem.snowball.PorterStemmer(ignore_stopwords=False).stem

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

max_len = 430

from transformers import AutoModelForSequenceClassification

from transformers import BertTokenizer
import torch
import re
import torch
import torch.nn as nn
from transformers import DebertaTokenizer, DebertaModel
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
import matplotlib as mpl
import matplotlib.pyplot as plt
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification
import pandas as pd
import psycopg2
import sys
import csv
import os
import glob
from io import StringIO
from sqlalchemy import create_engine

import pandas as pd
import psycopg2


import googleapiclient.discovery
from google.cloud import compute_v1
import time


def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text.lower()

def postgresql_to_dataframe(conn, select_query, column_names):
    """
    Transform a SELECT query result into a pandas DataFrame.

    :param conn: A PostgreSQL database connection object.
    :param select_query: The SQL SELECT query to execute.
    :param column_names: A list of column names for the resulting DataFrame.
    :return: A pandas DataFrame containing the query results.
    """
    cursor = conn.cursor()
    try:
        # Execute the SELECT query using the provided connection
        cursor.execute(select_query)
    except (Exception, psycopg2.DatabaseError) as error:
        # Handle any query execution errors, print the error message, and close the cursor
        print("Error: %s" % error)
        cursor.close()
        return 1
    # Fetch all the query results into a list of tuples
    tupples = cursor.fetchall()
    cursor.close()
    # Create a pandas DataFrame from the list of tuples with the specified column names
    df = pd.DataFrame(tupples, columns=column_names)

    return df




def connect(params_dic):
    """
    Connect to the PostgreSQL database server using the provided connection parameters.
        :param params_dic: A dictionary containing database connection parameters.
        :return: A database connection object.
    """
    conn = None
    try:
        # Attempt to connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params_dic)
    except (Exception, psycopg2.DatabaseError) as error:
        # Handle any connection errors, print the error message, and exit
        print(error)
        sys.exit(1)
    print("Connection successful")

    # Return the database connection object
    return conn



def extract(query, param_dic):
    # Passing the connection details for the PostgreSQL server
    # Connect to the database using the provided connection parameters
    conn = connect(param_dic)
    # Execute the provided SQL query and retrieve the results as a DataFrame
    df = pd.read_sql_query(query, conn)
    # Print a message to indicate that the data extraction is complete
    print("Data extract complete...")
    # Return the DataFrame containing the extracted data
    return df


def psql_insert_copy(table, conn, keys, data_iter):
    # Get a DBAPI connection that can provide a cursor
    dbapi_conn = conn.connection
    # Use a cursor for executing SQL commands
    with dbapi_conn.cursor() as cur:
        # Create a StringIO object to store the data in a CSV format
        s_buf = StringIO()
        # Create a CSV writer to write the data to the StringIO buffer
        writer = csv.writer(s_buf)
        # Write the data_iter (the data rows) to the StringIO buffer
        writer.writerows(data_iter)
        # Reset the buffer position to the beginning
        s_buf.seek(0)
        # Generate the list of column names as a comma-separated string
        columns = ', '.join('"{}"'.format(k) for k in keys)
        # Determine the full table name, including schema if it exists
        if table.schema:
            table_name = '{}.{}'.format(table.schema, table.name)
        else:
            table_name = table.name
        # Construct the SQL query for data insertion using the PostgreSQL COPY command
        sql = 'COPY {} ({}) FROM STDIN WITH CSV'.format(
            table_name, columns)
        # Execute the COPY command with the data from the StringIO buffer
        cur.copy_expert(sql=sql, file=s_buf)



def close_connection(param_dic):
    # Establish a connection to the PostgreSQL database using the provided parameters.
    conn = psycopg2.connect(database=param_dic['database'],
                            user=param_dic['user'],
                            password=param_dic['password'],
                            host=param_dic['host'],
                            port="5432")

    # Print the 'closed' attribute of the connection (0 means open, 1 means closed).
    print(conn.closed)
    # Close the database connection to release resources.
    conn.close()


def upload(df, param_dic):
    # Print a message to indicate the upload process has started.
    print('Uploading to PostgreSQL...')
    # Create a database engine using the provided connection parameters.
    engine = create_engine('postgresql://' + param_dic['user'] + ':' + param_dic['password'] + '@' + param_dic['host'] + ':5432/' + param_dic['database'])
    # Upload the DataFrame 'df' to the PostgreSQL database using the provided engine.
    ''' - 'a_test' is the table name in the database.
        - 'sandbox' is the schema where the table will be created.
        - if_exists='replace' will replace the table if it already exists.
        - index=False indicates not to include the DataFrame index in the database.'''
    df.to_sql('pjs_model', engine, method=psql_insert_copy, schema='sandbox', if_exists='append', index=False)
    # Close the database connection using a separate function.
    close_connection(param_dic)
    # Print a message to indicate that the database connection has been closed.
    print('Connection closed.')



def run_query(query, param_dic):
    # Establish a connection to the PostgreSQL database using the provided parameters.
    conn = psycopg2.connect(database=param_dic['database'],
                            user=param_dic['user'],
                            password=param_dic['password'],
                            host=param_dic['host'],
                            port="5432")

    # Create a cursor object to interact with the database.
    cursor = conn.cursor()
    # Execute the SQL query provided as a parameter.
    cursor.execute(query)
    # Commit the transaction to save the changes to the database.
    cursor.execute("COMMIT")
    # Close the database connection to release resources.
    conn.close()

def concat_csv_files(directory, extension='csv'):
    """
    Concatenate multiple CSV files in a directory into a single DataFrame.

    Args:
        directory (str): Path to the directory containing CSV files.
        extension (str): File extension of the CSV files (default is 'csv').

    Returns:
        pd.DataFrame: Concatenated DataFrame.

    # Specify the directory and extension
        directory_path = "G:/.shortcut-targets-by-id/1uyENisO-y_QgU7JTbN4TWyLtow8ngXtZ/PJS/Coach"
        file_extension = 'csv'
    # Call the function to concatenate CSV files
        df = concat_csv_files(directory_path, file_extension)
    """
    os.chdir(directory)  # Change the working directory to the specified directory
    all_filenames = [i for i in glob.glob(f'*.{extension}')]  # List of all CSV files

    # Concatenate all CSV files into a single DataFrame
    df = pd.concat([pd.read_csv(f, encoding='unicode_escape') for f in all_filenames], ignore_index=True)

    return df


def initialize_model(epochs=2):
    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    # Instantiate Bert Classifier
    deberta_classifier = DebertaClassifier()

    # Tell PyTorch to run the model on CPU
    deberta_classifier.to(device)

    # Create the optimizer
    optimizer = AdamW(deberta_classifier.parameters(),
                      lr=5e-5,    # Default learning rate
                      eps=1e-8    # Default epsilon value
                      )
    return deberta_classifier, optimizer

def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def preprocessing_for_deberta(data):
    """Perform required preprocessing steps for pretrained DeBERTa.
    @param    data (list): List of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []
    
    tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base', do_lower_case=True)
    MAX_LEN = 450
    # For every sentence...
    for sent in data:
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text=sent,                      # Tokenize sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN,             # Max length to truncate/pad
            pad_to_max_length=True,         # Pad sentence to max length
            return_attention_mask=True      # Return attention mask
        )

        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks

# Create the BertClassfier class
class DebertaClassifier(nn.Module):
    """DeBERTa Model for Classification Tasks.
    """
    def __init__(self, freeze_deberta=False):
        """
        @param    deberta: a DebertaModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_deberta (bool): Set `False` to fine-tune the DeBERTa model
        """
        super(DebertaClassifier, self).__init__()
        # Specify hidden size of DeBERTa, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 30, 2

        # Instantiate DeBERTa model
        self.deberta = DebertaModel.from_pretrained('microsoft/deberta-base')

        # Instantiate a one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(H, D_out)
        )

        # Freeze the DeBERTa model
        if freeze_deberta:
            for param in self.deberta.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Feed input to DeBERTa and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that holds attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to DeBERTa
        outputs = self.deberta(input_ids=input_ids,
                               attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for the classification task
        last_hidden_state_cls = outputs.last_hidden_state[:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits
    
    
def deberta_predict(model, test_dataloader):
    """Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    all_logits = []

    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)

    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits).cpu().numpy()

    return probs

from google.cloud import storage

def download_model_weight(bucket_name, source_blob_name, destination_file_path):
    """Downloads a model weight file from Google Cloud Storage."""
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    # Download the model weight file
    blob.download_to_filename(destination_file_path)

if __name__ == "__main__":
    bucket_name = "sentiment_response"
    source_blob_name = r"deberta/deberta_classifier.pt"  # Path to the model weight file in the bucket
    destination_file_path = "model_weights.pt"  # Local path where you want to save the model weight file

    download_model_weight(bucket_name, source_blob_name, destination_file_path)

def overal_sentiment(df):

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  deberta_classifier, optimizer = initialize_model(epochs=2)

  path = r'model_weights.pt'
  deberta_classifier.load_state_dict(torch.load(path))

  test_inputs, test_masks = preprocessing_for_deberta(df['response'].to_numpy())

  # Create the DataLoader for our test set
  test_dataset = TensorDataset(test_inputs, test_masks)
  test_sampler = SequentialSampler(test_dataset)
  test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=32)

  # Compute predicted probabilities on the test set
  probs = deberta_predict(deberta_classifier, test_dataloader)
  classification = np.argmax(probs, axis = 1)

  return classification

def emotion_detector(df):
    # Load the pre-trained tokenizer
    tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")

    # Create a new column for truncated/split text
    max_length = 500
    df['truncated_text'] = df['response'].apply(lambda text: text[:max_length])

    # Prepare the dataset
    prob_df = df[['truncated_text']].rename(columns={'truncated_text': 'response'})
    ds = Dataset.from_pandas(prob_df)

    # Initialize the pipeline
    pipe = pipeline('text-classification', model="j-hartmann/emotion-english-distilroberta-base", device=device)
    
    # Tokenize and predict for each example
    results = pipe(KeyDataset(ds, "response"))

    emotion_label = []
    for idx, extracted_entities in enumerate(results):
        emotion_label.append(extracted_entities['label'])

    return emotion_label

def ner(df):
  prob_df = df['response'].to_frame()
  ds = Dataset.from_pandas(prob_df)

  model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
  tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")

  pipe = pipeline(task="ner", model=model, tokenizer=tokenizer, device = device)
  results = pipe(KeyDataset(ds, "response"))

  original = []
  temp_ent = []
  entities = []

  for idx, extracted_entities in enumerate(results):
      temp_ent = []
      # print('idx: ', idx)
      # print("Original text:\n{}".format(ds[idx]["response"]))
      # print("Extracted entities:")
      #original.append(dataset[idx]["text"])
      for entity in extracted_entities:
          temp_ent.append(entity)
          # print(entity)
      entities.append(temp_ent)

  return entities

def extraction(df):
  df['entities'] = ner(df)
  df2 = df.explode('entities')

  df1 = df2[df2['entities'].isna() == True]
  df2 = df2[df2['entities'].isna() == False]

  #entity
  df2['entity_type'] = df2['entities'].apply(lambda x:  x['entity'])
  #start
  df2['entity_start'] = df2['entities'].apply(lambda x:  x['start'])
  #end
  df2['entity_end'] = df2['entities'].apply(lambda x:  x['end'])
  #word
  df2['entity_word'] = df2['entities'].apply(lambda x:  x['word'])

  df['Name_mask'] = np.where( df['ticket_number'].isin( df2[df2['entity_type'] == 'B-PER']['ticket_number'] ), True, False )

  return df


# cleaning master function
def clean_response(response, bigrams=False):
    response = str(response).lower() # lower case
    response = re.sub('['+my_punctuation + ']+', ' ', response) # strip punctuation
    response = re.sub('\s+', ' ', response) #remove double spacing
    response = re.sub('([0-9]+)', '', response) # remove numbers
    response_token_list = [word for word in response.split(' ')]
                           # if word not in my_stopwords] # remove stopwords

    response_token_list = [word_rooter(word) if '#' not in word else word
                        for word in response_token_list] # apply word rooter

    lem_response = ' '.join(response_token_list)
    return response, lem_response


#lemmatizing taxonomy
def lem_taxon(term):
    term = str(term).lower()
    term = word_rooter(term)
    return term
def match_both(response, non_lem, tax_list, tier2_list, tier1_list):
    tier_match = []
    tier_replace = ['replace', 'this', 'part']
    words = response.split(' ')
    non_lem_words = non_lem.split(' ')
    for i, x in enumerate(tax_list):
        for n,y in enumerate(words):
          if x == y:
            tier_match.append([non_lem_words[n], tier2_list[i], tier1_list[i]])

    tier_match = [list(x) for x in set(tuple(x) for x in tier_match)]
    return tier_match

def apply_taxonomy(df):
  df['tax_match'] = df.apply(lambda x: match_both(x['lem_response'], x['non_lem_response'], tax['Term'].to_list(), tax['Tier 2'].to_list(), tax['Tier 1'].to_list() ) , axis = 1)
  df = df.explode('tax_match')

  df['tax_match'] = df['tax_match'].fillna('No Taxonomy')
  import math
  #unpacking tax_match arrays into individual columns
  tax_match = df['tax_match'].to_list()

  term = []
  tier2 = []
  tier1 = []
  clause = []

  for i, c in enumerate(tax_match):
    if c != 'No Taxonomy':
      term.append(c[0])
      tier2.append(c[1])
      tier1.append(c[2])
    else:
      term.append(np.nan)
      tier2.append(np.nan)
      tier1.append(np.nan)

  df['Term'], df['Tier_2'], df['Tier_1']= term, tier2, tier1

  return df

def tokenize_function(example, max_length=500):
    checkpoint = "nxnag/absa-transport"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
    
    return tokenizer(
        example["response"],
        example["Term"],
        truncation=True,
        max_length=max_length,
        padding="max_length",  # You may also want to pad to the max_length
        return_tensors="pt"    # Ensure PyTorch tensors are returned
    )

def absa_classification(df):
  #packaging data appropriately

  

  absa = df.dropna(subset = 'Term')
  absa['Term'] = absa['Term'].apply(lambda x: text_preprocessing(x))
  predict_ds = Dataset.from_pandas(absa[['response','Term']])
  
  checkpoint = "nxnag/absa-transport"
  model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3)
  tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast = False)
  #tokenizing dataset
  tokenized_datasets = predict_ds.map(tokenize_function, batched=True)
  data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

  #removing unnecessary columns
  tokenized_datasets = tokenized_datasets.remove_columns(["response", "Term", "__index_level_0__"])
  tokenized_datasets.set_format("torch")
  tokenized_datasets.column_names

  predict_dataloader = DataLoader(
      tokenized_datasets, batch_size=8, collate_fn=data_collator
  )

  for batch in predict_dataloader:
      break
  {k: v.shape for k, v in batch.items()}



  #batch predictions
  model.to(device)
  all_logits = []
  predictions = []
  for batch in predict_dataloader:
      batch = {k: v.to(device) for k, v in batch.items()}
      with torch.no_grad():
          outputs = model(**batch)

      logits = outputs.logits
      all_logits.append(logits)

      prediction = torch.argmax(logits, dim=-1).cpu().numpy()
      predictions.extend(prediction)

  absa['term_sentiment'] = predictions
  df = df.merge( absa[['ticket_number','Term','term_sentiment']], how = 'left', left_on = ['ticket_number','Term'], right_on = ['ticket_number','Term'] )
  return df

def keep_alphanumeric_and_punctuation(text):
        allowed_characters = string.ascii_letters + string.digits + string.punctuation.replace('+', '') + string.whitespace
        cleaned_text = ''.join(char for char in text if char in allowed_characters)
        return cleaned_text

def pulling_data():
    #loading taxonomy
    taxonomy_path = r'gs://absa-classification/taxonomy.csv' 
    tax = pd.read_csv(taxonomy_path)
    
    query = ''' select pm.ticket_number from sandbox.pjs_model pm '''
    param_dic = {
            "host"      : "",
            "database"  : "",
            "user"      : "",
            "password"  : ""
        }
    prev = extract(query, param_dic)
    
    
    
    #clean reason response
    my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@'


    tax['Term'] = tax['Term'].apply(lambda x: lem_taxon(x))

    df = pd.read_csv(r'gs://sentiment_response/pjs_upload.csv')
    
    
    df = df.rename(columns = {'Please tell us why you gave that score':'response',
                          'How likely is it that you would recommend National Express to friends, family members or colleagues?':'nps',
                          'How satisfied were you with each of the following aspects? | The overall experience':'csat',
                          'ticket':'ticket_number'})
    
    df = df.dropna(subset = 'response')

    df['nps'] = df['nps'].replace('10 (Extremely likely)',10)
    df['nps'] = df['nps'].replace('0 (Not at all likely)',0)
    df['nps'] = df['nps'].astype(int)

    #target variable will nps split into demoters, passives and promoters
    df['label'] = np.where(df['nps'] == 3,2,
                      np.where(df['nps'] == 1,0,1))

    df['response'] = df['response'].apply(lambda x: text_preprocessing(x))
    
    df = df[df['ticket_number'].isin(prev['ticket_number']) == False]
    
    
    df['response'] = df['response'].apply(keep_alphanumeric_and_punctuation)
    df['response'] = df['response'].str.replace('+','')
    
    df['utf8'] = np.where(  df['response'].str.contains('Ã', case = False), False, True)
    df['utf8'] = np.where(  df['response'].str.contains('Ã', case = False), False, True)
    df = df[df['utf8'] == True]
    
    return df[['ticket_number','nps','response','csat']], tax

def model_applications(df,tax):
    prob_df = df['response'].to_frame()
    ds = Dataset.from_pandas(prob_df)
    
    df['Overall sentiment'] = overal_sentiment(df)
    df['emotion'] = emotion_detector(df)

    #Using regex to mask phone numbers and emails
    #UK phone number
    phone_pattern = '((\+44(\s\(0\)\s|\s0\s|\s)?)|0)7\d{3}(\s)?\d{6}'
    df['masked_response'] = df['response'].apply(lambda x: re.sub(phone_pattern, '[number_removed]',x ))

    #email address
    email_pattern = r'([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+'
    df['masked_response'] = df['response'].apply(lambda x: re.sub(email_pattern, '[email_removed]',x ))

    #topic modelling (couldn't get it to work in time)
    df['topic_model_label'] = 'not working'

    df = extraction(df)

    df['non_lem_response'],df['lem_response'] = zip(*df['response'].apply(lambda x: clean_response(x)))
    df = apply_taxonomy(df)


    df = absa_classification(df)
    
    return df

def clean_output(df):
    df['Overall sentiment'] = np.where(df['Overall sentiment'] == 0, 'Negative', 'Positive')
    df['term_sentiment'] = np.where(df['term_sentiment'] == 0, 'Negative',
                                    np.where(df['term_sentiment'] == 1, 'Neutral',
                                             np.where(df['term_sentiment'] == 2,  'Positive', np.nan)))
                                    
    return df[['ticket_number','nps','response','csat','Overall sentiment','emotion','topic_model_label','Name_mask','Term','Tier_2','Tier_1','term_sentiment']]
    
    

from datasets import Dataset
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline
)
from transformers.pipelines.pt_utils import KeyDataset
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#pulling data from the appropriate location
df, tax = pulling_data()


#applying all the models to the free text
output = model_applications(df,tax)
#cleaning it to be uploaded to the SCV
output = clean_output(output)


param_dic = {
        "host"      : "",
        "database"  : "",
        "user"      : "",
        "password"  : ""
    }

#uploading the clean output to the SCV using my current database login details
upload(output,param_dic)


# project_id = 'surveys-402414'
# zone = 'europe-west2-a'
# instance_name = 'pjs-process'

# compute = googleapiclient.discovery.build('compute', 'v1')
# result = compute.instances().stop(project=project_id, zone=zone, instance=instance_name).execute()
