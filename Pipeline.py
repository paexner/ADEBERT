import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from gensim.models import Word2Vec
from transformers import  TFAutoModel, AutoTokenizer, TFBertForSequenceClassification
import tensorflow as tf
from sklearn.model_selection import train_test_split
import re
from sentence_transformers import SentenceTransformer
import inflect
import time
from datetime import timedelta, datetime
import ConfigFile as cf
from ConfigFile import RANDOM_SEED
from sklearn.ensemble import IsolationForest
from sklearn import metrics
import random
from sklearn.metrics import classification_report, f1_score
from dateutil import parser
import Autoencoder as ae


class Encoder:
    def __init__(self, method, group_column, timestamp='None', columns2encode=['activity', 'timestamp'],
                 vector_size=100, window=5, min_count=1, workers=4, model_name='bert-base-uncased'):
        self.method = method
        self.group_column = group_column
        self.encoder = None
        self.timestamp = timestamp
        self.columns2encode = columns2encode

        #Parameters vor SVM
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers

        #Parameters for LLM
        if self.method == 'llm_chunked':
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = TFAutoModel.from_pretrained(model_name)
        if self.method == 'llm':
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = TFAutoModel.from_pretrained(model_name)
        elif self.method == 'llm_trained':
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = TFAutoModel.from_pretrained(model_name)
            self.sop_model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
        elif self.method == 'llm_trained_unchunked':
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = TFAutoModel.from_pretrained(model_name)
            self.sop_model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
        elif self.method == 'sentence_transformer':
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        elif self.method == 'sentence_transformer_chunked':
            self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def encode(self, data):
        data = data[[self.group_column] + self.columns2encode + [self.timestamp]]
        if self.method == 'one-hot':
            start_encoding = time.time()

            encodings = self.encode_one_hot(data)

            end_encoding = time.time()
            encoding_duration = end_encoding - start_encoding
            return encodings, (encoding_duration, 0, len(encodings))
        elif self.method == 'word2vec':
            start_process = time.time()

            encodings, encoding_duration = self.encode_word2vec_new(data)
            end_process = time.time()
            process_duration = end_process - start_process
            return encodings, (encoding_duration, process_duration, len(encodings))
        elif self.method == 'llm':
            start_encoding = time.time()

            encodings, encoding_duration = self.encode_llm(data)

            end_encoding = time.time()
            encoding_duration = end_encoding - start_encoding

            return encodings, (encoding_duration, 0, len(encodings))

        elif self.method == 'llm_chunked':
            start_encoding = time.time()

            encodings, encoding_duration = self.encode_chunked_llm(data)

            end_encoding = time.time()
            encoding_duration = end_encoding - start_encoding
            return encodings, (encoding_duration, 0, len(encodings))
        elif self.method == 'llm_trained':

            start_process = time.time()
            combined_sequences, grouped = self.prepare_sop_data(data)
            self.fine_tune_model(combined_sequences)
            end_process = time.time()
            process_duration = end_process - start_process

            start_encoding = time.time()
            encodings = self.generate_embeddings(grouped)
            end_encoding = time.time()
            encoding_duration = end_encoding - start_encoding

            return encodings, (encoding_duration, process_duration, len(encodings))
        elif self.method == 'sentence_transformer':
            start_encoding = time.time()

            encodings, encoding_duration = self.encode_sentence_transformer(data)

            end_encoding = time.time()
            encoding_duration = end_encoding - start_encoding
            return encodings, (encoding_duration, 0, len(encodings))


    def encode_one_hot(self, data):
        data_to_encode = data.copy()

        # Initialize OneHotEncoder
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
        columns_to_encode = [col for col in data_to_encode.columns if
                             col not in [self.group_column, self.timestamp]]
        encoder.fit(data[columns_to_encode])

        # List to store the transformed data for each group
        transformed_data_list = []

        # Process each group independently
        for group_id, group_data in data.groupby(self.group_column, sort=False):

            group_encoded = encoder.transform(group_data[columns_to_encode])
            flattened_encoded = group_encoded.toarray().flatten()
            column_names = [f"{feature}_{i + 1}" for i in range(group_encoded.shape[0]) for feature in
                            encoder.get_feature_names_out()]
            flattened_df = pd.DataFrame([flattened_encoded], columns=column_names)
            flattened_df.insert(0, self.group_column, group_id)

            transformed_data_list.append(flattened_df)


        encoded_df = pd.concat(transformed_data_list, ignore_index=True)
        encoded_df.fillna(0, inplace=True)
        return encoded_df

    def encode_word2vec_new(self, data):

        cases = data.groupby(self.group_column, sort=False).apply(lambda x: process_group(x[self.columns2encode]))

        listoflists = []
        event_size = 1
        for case in cases:
            events = []
            for event in case:
                event_array = event.split(' ')
                event_size = len(event_array)
                events.extend(event_array)
            listoflists.append(events)

        case_ids = cases.index

        # Train Word2Vec model
        word2vec = Word2Vec(sentences=listoflists, vector_size=self.vector_size,
                            window=self.window, min_count=self.min_count, workers=self.workers)

        start_encoding = time.time()

        embeddings = [process_events_V3(sentence, word2vec, event_size) for sentence in listoflists]

        end_encoding = time.time()
        encoding_duration = end_encoding - start_encoding

        df_embeddings = pd.DataFrame(embeddings)
        df_embeddings.insert(0, self.group_column, case_ids)
        df_embeddings.fillna(0, inplace=True)
        return df_embeddings, encoding_duration


    def encode_chunked_llm(self, data):

        grouped = data.groupby(self.group_column, sort=False).apply(lambda x: process_group_llm(x[self.columns2encode + ['timestamp']]))

        embeddings = []
        for case_id, event_text in grouped.items():
            # Split long texts into chunks of a maximum size
            max_length = 512 - 2  # Account for [CLS] and [SEP] tokens
            chunks = [event_text[i:i + max_length] for i in range(0, len(event_text), max_length)]

            case_embeddings = []
            for chunk in chunks:
                encoded_input = self.tokenizer(chunk, return_tensors='tf', truncation=False, padding='max_length', max_length=512)
                output = self.model(encoded_input)

                # Take the mean of the output embeddings (ignoring special tokens)
                chunk_embedding = tf.reduce_mean(output.last_hidden_state[0, 1:-1], axis=0)
                case_embeddings.append(chunk_embedding)

            # Concatenate embeddings from all chunks for a single case
            case_embedding = tf.concat(case_embeddings, axis=0)
            case_embedding = tf.reshape(case_embedding, [-1]).numpy()
            embeddings.append(case_embedding)

        # Create DataFrame from embeddings
        df_embeddings = pd.DataFrame(embeddings)
        df_embeddings.insert(0, self.group_column, grouped.index)
        df_embeddings.fillna(0, inplace=True)
        return df_embeddings, 0

    def encode_llm(self, data):

        grouped = data.groupby(self.group_column, sort=False).apply(lambda x: process_group_llm(x[self.columns2encode + ['timestamp']]))

        embeddings = []
        for case_id, event_text in grouped.items():
            encoded_input = self.tokenizer(event_text, return_tensors='tf', truncation=False, padding='max_length', max_length=512)

            output = self.model(encoded_input)


            embedding = tf.reduce_mean(output.last_hidden_state[0, 1:-1], axis=0)
            case_embedding = tf.reshape(embedding, [-1]).numpy()
            embeddings.append(case_embedding)

        # Create DataFrame from embeddings
        df_embeddings = pd.DataFrame(embeddings)
        df_embeddings.insert(0, self.group_column, grouped.index)
        df_embeddings.fillna(0, inplace=True)
        return df_embeddings, 0

    def encode_sentence_transformer(self, data):

        grouped = data.groupby(self.group_column, sort=False).apply(lambda x: process_group_llm(x[self.columns2encode + ['timestamp']]))

        embeddings = []
        for case_id, event_text in grouped.items():
            embedding = self.model.encode(event_text)
            embeddings.append(embedding)

        df_embeddings = pd.DataFrame(embeddings)
        df_embeddings.insert(0, self.group_column, grouped.index)
        return df_embeddings, 0

    def prepare_sop_data(self, data):

        if not pd.api.types.is_datetime64_any_dtype(data[self.timestamp]):
            data[self.timestamp] = data[self.timestamp].apply(parse_time)
        data = data.sort_values(by=[self.group_column, self.timestamp])

        grouped = data.groupby(self.group_column, sort=False).apply(lambda x: process_group_llm(x[self.columns2encode + ['timestamp']]))

        original_sequences = grouped.copy().reset_index(name='event_text')
        print('Before',len(original_sequences))
        original_sequences = original_sequences.sample(frac=0.2, random_state=42)
        print('After',len(original_sequences))
        shuffled_sequences = original_sequences.copy()
        shuffled_sequences['event_text'] = shuffled_sequences['event_text'].apply(lambda x: ' ||| '.join(random.sample(x.split(' ||| '), len(x.split(' ||| ')))))

        original_sequences['label'] = 1
        shuffled_sequences['label'] = 0

        combined_sequences = pd.concat([original_sequences, shuffled_sequences])

        return combined_sequences, grouped


    def fine_tune_model(self, combined_sequences, epochs=3, batch_size=8, max_len=128):
        # Split data into train and test sets
        train_data, test_data = train_test_split(combined_sequences, test_size=0.2, random_state=RANDOM_SEED)

        # Tokenize data
        def tokenize_function(examples):
            return self.tokenizer(examples["event_text"].tolist(), padding="max_length", truncation=True, max_length=max_len)

        train_encodings = tokenize_function(train_data)
        test_encodings = tokenize_function(test_data)

        train_labels = tf.keras.utils.to_categorical(train_data['label'].tolist())
        test_labels = tf.keras.utils.to_categorical(test_data['label'].tolist())

        # Create tf.data.Dataset objects
        train_dataset = tf.data.Dataset.from_tensor_slices((
            dict(train_encodings),
            train_labels
        )).shuffle(len(train_encodings['input_ids'])).batch(batch_size)

        test_dataset = tf.data.Dataset.from_tensor_slices((
            dict(test_encodings),
            test_labels
        )).batch(batch_size)

        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        self.sop_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        # Fine-tune model
        self.sop_model.fit(train_dataset, validation_data=test_dataset, epochs=epochs)



    def generate_embeddings(self, grouped):
        embeddings = []
        for case_id, event_text in grouped.items():
            max_length = 512 - 2  # Account for [CLS] and [SEP] tokens
            chunks = [event_text[i:i + max_length] for i in range(0, len(event_text), max_length)]

            case_embeddings = []
            for chunk in chunks:
                encoded_input = self.tokenizer(chunk, return_tensors='tf', truncation=False, padding='max_length', max_length=512)
                output = self.model(encoded_input)

                chunk_embedding = tf.reduce_mean(output.last_hidden_state[0, 1:-1], axis=0)
                case_embeddings.append(chunk_embedding)

            case_embedding = tf.concat(case_embeddings, axis=0)
            case_embedding = tf.reshape(case_embedding, [-1]).numpy()
            embeddings.append(case_embedding)

        df_embeddings = pd.DataFrame(embeddings)
        df_embeddings.insert(0, self.group_column, grouped.index)
        df_embeddings.fillna(0, inplace=True)
        return df_embeddings



def process_events_V3(sentence, model, event_size):
    vector_size = model.vector_size  # Assuming each token's embedding has the same size
    event_vectors = []

    # Process each event in the sentence
    token_vectors = []
    i = 0
    for word in sentence:

        # Retrieve embeddings for each token
        if word in model.wv:
            token_vectors.append(model.wv[word])
        else:
            token_vectors.append(np.zeros(vector_size))  # Handle missing tokens
            print(f'{word} not in word2vec model')

        if i == event_size:
            # Calculate the mean of the token vectors to form one vector per event
            if token_vectors:
                mean_vector = np.mean(token_vectors, axis=0)
            else:
                mean_vector = np.zeros(vector_size)  # Handle case with no valid tokens
                token_vectors = []
            i = 0
            event_vectors.append(mean_vector)
        i += 1

    # Concatenate all event mean vectors horizontally to form one large vector for the entire sentence
    if event_vectors:
        concatenated_vector = np.hstack(event_vectors)
    else:
        concatenated_vector = np.zeros(vector_size * len(word))  # Handle empty sentence case

    return concatenated_vector.tolist()
#%%
def select_columns(columns, data):
    # Check if all column names exist in the DataFrame
    missing_cols = [col for col in columns if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
    return data[columns]
#%%

class AnomalyDetector:
    def __init__(self, model_type='autoencoder'):
        self.model_type = model_type
        self.model = None
        self.encoder = None

    def fit_train(self, train_set, group_column, label_column, epochs=200, batch_size=50, learning_rate= 1e-3, hidden_layers=2,
                  hidden_size_factor=0.5, eps=0.5, min_samples=5, method='noMethodGiven'):

        if self.model_type == 'autoencoder':
            if label_column:
                train = train_set.drop(columns=[group_column, label_column])
            else:
                train = train_set.drop(columns=[group_column])

            encoded_values = train.values  # Convert DataFrame to NumPy array
            input_dim = encoded_values.shape[1]  # Get the number of features

            # Split the data into training and validation sets (80-20 split)
            split_ratio = 0.8
            split_index = int(split_ratio * len(encoded_values))
            train_data = encoded_values[:split_index]
            val_data = encoded_values[split_index:]

            #self.model = ae.build_dynamic_autoencoder(input_dim, hidden_layers=hidden_layers)
            self.model = ae.build_denoising_autoencoder(input_dim)
            ae.train_denoising_autoencoder(self.model, train_data, val_data, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, noise_factor=0.5)

        if self.model_type == 'iforest':
            if label_column:
                train = train_set.drop(columns=[group_column, label_column])
                test = train_set.drop(columns=[group_column, label_column])
            else:
                train = train_set.drop(columns=[group_column])
                test = train_set.drop(columns=[group_column])
            self.model = IsolationForest(contamination='auto', random_state=42, n_jobs=10)
            test['method'] = method
            self.model.fit(train)
            return self.model

    def determine_threshold(self,loss, method='percentile', labels=None, buffer_fraction=0.1):
        best_alpha = 0
        if method == 'iqr':
            q25, q75 = np.percentile(loss, 25), np.percentile(loss, 75)
            iqr = q75 - q25
            threshold = q75 + 1.5 * iqr
        elif method == 'gaussian':
            mean = np.mean(loss)
            std = np.std(loss)
            threshold = mean + 3 * std
        elif method == 'cross_val' and labels is not None:
            thresholds = np.linspace(min(loss), max(loss), num=100)
            best_threshold = thresholds[0]
            best_f1 = 0
            for threshold in thresholds:
                predictions = (loss > threshold).astype(int)
                f1 = f1_score(labels, predictions)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            threshold = best_threshold
        elif method == 'cross_gauss' and labels is not None:
            alphas = np.linspace(0.1, 1.5, num=15)
            best_alpha = alphas[0]
            best_f1 = 0
            mean = np.mean(loss)
            std = np.std(loss)
            for alpha in alphas:
                threshold = mean + alpha * std
                predictions = (loss > threshold).astype(int)
                f1 = f1_score(labels, predictions)
                if f1 > best_f1:
                    best_f1 = f1
                    best_alpha = alpha
            threshold = mean + best_alpha * std
        else:  # default to alpha = 0.1
            mean = np.mean(loss)
            std = np.std(loss)
            threshold = mean + 0.1 * std

        # Symmetric buffer zone around the main threshold
        buffer_margin = buffer_fraction * std
        buffer_margin_lower = buffer_fraction * std

        threshold_lower = threshold - buffer_margin_lower
        threshold_upper = threshold + buffer_margin

        return threshold_lower, threshold, threshold_upper, best_alpha

    def evaluate(self, data_c, label_column, group_column, method='NoMethodGiven', threshold=None):
        if 'autoencoder' in self.model_type:
            if label_column:
                data_features = data_c.drop(columns=[group_column, label_column])
                labels = data_c[label_column]
            else:
                data_features = data_c.drop(columns=[group_column])
                labels = None
            print("Data prepared for prediction.")
            reconstructions = self.model.predict(data_features)
            print("Model prediction completed.")
            loss = np.mean(np.square(data_features - reconstructions), axis=1)
            print("Loss computed.")
            # Fine-tune threshold on sample of the data
            sample_size = int(len(loss) * 0.2)
            sampled_indices = np.random.choice(len(loss), size=sample_size, replace=False)
            print("Sampling completed.")
            finetune_loss = loss[sampled_indices]
            finetune_labels = labels.iloc[sampled_indices] if labels is not None else None
            if label_column:
                threshold_lower, threshold, threshold_upper, best_alpha = self.determine_threshold(loss=finetune_loss, labels=finetune_labels, method='cross_gauss',  buffer_fraction=0.5)
            else:
                threshold_lower, threshold, threshold_upper, best_alpha = self.determine_threshold(loss=loss, labels=labels, method='cross_gauss',  buffer_fraction=0.5)
            print(f'Threshold_lower: {threshold_lower} and Threshold_upper: {threshold_upper}')

            predictions = np.zeros(len(loss))
            predictions[loss > threshold] = 1  # Anomalies
            predictions[(loss > threshold_upper) & (loss <= threshold_lower)] = 0 # Buffer data
            print("Predictions generated.")

            loss_analysis = pd.DataFrame({
                group_column: data_c[group_column],
                'loss': loss,
                'label': predictions
            })
            print("Loss analysis DataFrame created.")
            if label_column:
                loss_analysis[label_column] = data_c[label_column]
                #For the report
                report = classification_report(data_c[label_column], predictions, output_dict=True)
                report = pd.DataFrame(report).transpose()
                report = report.unstack()
                column_headers = ['_'.join(tuple) for tuple in report.index]
                report = pd.DataFrame([report])
                report.columns = column_headers
                report['encoding'] = method
                report['best_alpha'] = best_alpha

                fpr, tpr, thresholds = metrics.roc_curve(data_c[label_column], predictions, pos_label=1)
                roc_auc = metrics.auc(fpr, tpr)
                report['AUC'] = roc_auc
                print("Report generated.")
            else:
                report = None

            predictions[(loss > threshold_lower) & (loss <= threshold_upper)] = 2  # Buffer data

            print("Buffer Zone updated.")

            return report, loss_analysis

        elif self.model_type == 'iforest':

            if label_column:
                data_features = data_c.drop(columns=[group_column, label_column])
            else:
                data_features = data_c.drop(columns=[group_column])
            raw_predictions = self.model.predict(data_features)
            predictions = (raw_predictions == -1).astype(int)
            anomaly_scores = -self.model.decision_function(data_features)

            loss_analysis = pd.DataFrame({
                group_column: data_c[group_column],
                'loss': anomaly_scores,
                'label': predictions
            })

            if label_column:
                loss_analysis[label_column] = data_c[label_column]

                report = classification_report(data_c[label_column], predictions, output_dict=True)
                report = pd.DataFrame(report).transpose()
                report = report.unstack()
                column_headers = ['_'.join(tuple) for tuple in report.index]
                report = pd.DataFrame([report])
                report.columns = column_headers
                report['encoding'] = method

                fpr, tpr, thresholds = metrics.roc_curve(data_c[label_column], predictions, pos_label=1)
                roc_auc = metrics.auc(fpr, tpr)
                report['AUC'] = roc_auc
            else:
                report = None

            return report, loss_analysis


def process_group(group):
    return [" ".join(map(str, row)) for row in group.values]

def process_group_llm(group):
    # Apply the transformation to each row and then join with ' ||| '
    return ' ||| '.join(
        group.apply(lambda event: ' | '.join(event.map(str)), axis=1)
    )

#%%
def select_first_n_groups(dataframe, group_column, n=5):
    # Group by the specified column
    grouped = dataframe.groupby(group_column, sort=False)

    # Initialize an empty list to collect the DataFrame slices
    selected_groups = []

    # Iterate over the groups and select the first n groups
    for name, group in list(grouped)[:n]:
        selected_groups.append(group)
    # Concatenate all selected groups into a single DataFrame
    result = pd.concat(selected_groups)
    return result
#%%
def get_configuration(file):
    """
    Determine the configuration based on the file name.
    """
    if 'bpic12' in file:
        return cf.get_args_BPIC12()
    elif 'bpic13' in file:
        return cf.get_args_BPIC13()
    elif 'hospital' in file:
        return cf.get_args_hospitalbilling()
    elif 'artificial' in file:
        return cf.get_args_artifical_DS()
    elif 'PermitLog' in file:
        return cf.get_args_PermitLog()
    elif 'Case_Study' in file:
        return cf.get_args_Case_Study()
    else:
        raise ValueError('No Arguments for Dataset were given ')
#%%
def nth_weekday(date):
    """ Returns a string indicating the nth weekday of the month for the given date. """
    # Weekday and week list to match date.weekday() output to weekday name
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    counts = ["first", "second", "third", "fourth", "fifth"]

    weekday_count = (date.day - 1) // 7
    weekday_name = weekdays[date.weekday()]

    return f"{counts[weekday_count]} {weekday_name}"




def format_timestamp_V2(date):
    """Formats a datetime object into a descriptive natural language string with numbers written out."""
    inflect_engine = inflect.engine()

    months = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]

    components = []
    # Check and format each component if present
    if date.year is not None:
        components.append(f"the year is {inflect_engine.number_to_words(date.year)}")
    if date.month is not None:
        components.append(f"the month is {months[int(date.month) - 1]}")
    if date.day is not None:
        components.append(f"the day is the {inflect_engine.ordinal(date.day)}")
    if date.weekday() is not None:
        components.append(f"which is the {nth_weekday(date)}")
    if date.hour is not None:
        components.append(f"at {inflect_engine.number_to_words(date.hour)} o'clock")
    if date.minute is not None:
        components.append(f"{inflect_engine.number_to_words(date.minute)} minutes")
    if date.second is not None:
        components.append(f"and {inflect_engine.number_to_words(date.second)} seconds")

    return ', '.join(components)

def parse_natural_language_timestamp(natural_lang_str):
    """Parses a descriptive natural language string with numbers written out into a datetime object."""
    inflect_engine = inflect.engine()
    months = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]

    # Function to convert written numbers to digits
    def text_to_number(text):
        return inflect_engine.number_to_words(text, wantlist=True, group=1, andword='').pop()

    # Initialize components with default values
    year, month, day, hour, minute, second = None, None, None, None, None, None

    # Regular expressions to extract components
    year_re = re.compile(r"the year is (\w+)")
    month_re = re.compile(r"the month is (\w+)")
    day_re = re.compile(r"the day is the (\w+)")
    hour_re = re.compile(r"at (\w+) o'clock")
    minute_re = re.compile(r"(\w+) minutes")
    second_re = re.compile(r"and (\w+) seconds")

    # Extracting components using regex
    year_match = year_re.search(natural_lang_str)
    if year_match:
        year = int(text_to_number(year_match.group(1)))

    month_match = month_re.search(natural_lang_str)
    if month_match:
        month = months.index(month_match.group(1)) + 1

    day_match = day_re.search(natural_lang_str)
    if day_match:
        day = int(text_to_number(day_match.group(1)))

    hour_match = hour_re.search(natural_lang_str)
    if hour_match:
        hour = int(text_to_number(hour_match.group(1)))

    minute_match = minute_re.search(natural_lang_str)
    if minute_match:
        minute = int(text_to_number(minute_match.group(1)))

    second_match = second_re.search(natural_lang_str)
    if second_match:
        second = int(text_to_number(second_match.group(1)))

    # Construct datetime object
    if year and month and day is not None:
        return datetime.datetime(year, month, day, hour or 0, minute or 0, second or 0)
    else:
        raise ValueError("Missing essential date components")
#%%
def parse_time(x):
    if pd.isna(x):
        return pd.NaT
    else:
        try:
            timestamp = parser.parse(x)
        except ValueError:
            try:
                # Attempt to parse as a time duration in minutes and seconds
                minutes, seconds = map(float, x.split(':'))
                return datetime(1970, 1, 1) + timedelta(minutes=minutes, seconds=seconds)
            except ValueError:
                raise ValueError("The provided time string is not in a recognized format.")
        except TypeError:
            try:
                return x
            except ValueError:
                raise ValueError("The provided time string is not in a recognized format.")
        return timestamp