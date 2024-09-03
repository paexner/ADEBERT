import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertTokenizer, GPT2Tokenizer, TFBertForSequenceClassification, TFGPT2ForSequenceClassification
import numpy as np
import pandas as pd
import Pipeline as pm
import random
import Event_Pair_Generator as epg
import tensorflow as tf
import sys
import warnings
from ConfigFile import RANDOM_SEED, NUM_EPOCHS, BATCH_SIZE
from tensorflow.keras import mixed_precision
from tensorflow.keras import backend as K
K.clear_session()
# Check if a GPU is available


gpus = tf.config.list_physical_devices('GPU')

if gpus:
    policy = mixed_precision.Policy('float32')
    mixed_precision.set_global_policy(policy)
    print("Mixed precision policy set to 'mixed_float16'")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found, mixed precision policy not set.")

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


# Convert data to tf.data.Dataset
def create_tf_dataset(encodings, labels):
    return tf.data.Dataset.from_tensor_slices((dict(encodings), labels))




# Function to load and prepare data
def load_and_prepare_data(file_path, options):
    # Load Data
    data = pd.read_csv(file_path)
    if options['isAnomaly']:
        label = data[[options['case_id'], options['isAnomaly']]].copy()
        label = label.groupby(options['case_id'])[options['isAnomaly']].max().reset_index()
    else:
        label=None

    return data, label

# Function to encode data
def encode_data(method, data, options, run_id, file):
    encoder = pm.Encoder(method=method, group_column=options['case_id'], timestamp=options['timestamp'], columns2encode=options['columns2encode'])
    columns2use = pm.select_columns(options['columns2use'], data)
    encoded, performance = encoder.encode(columns2use)
    dir_save = options['output_dir']
    encoded.to_csv(f'{dir_save}_{file}_{method}_{run_id}_encoded.csv', index=False)
    return encoded, performance

# Function to train and evaluate an anomaly detector
def train_and_evaluate_anomaly_detector(encoded, label, options, output_file_detect, method, anomalydetector_type='autoencoder', run_id='NoID', file_name='noFileName'):
    anomalydetector = pm.AnomalyDetector(model_type=anomalydetector_type)
    if options['isAnomaly']:
        encoded = pd.merge(encoded, label, on=options['case_id'], how='inner')

    #fit autoencoder
    anomalydetector.fit_train(encoded, group_column=options['case_id'], label_column=options['isAnomaly'], method=method)
    #predict with autoencoder
    report, loss_analysis = anomalydetector.evaluate(encoded, group_column=options['case_id'], label_column=options['isAnomaly'], method=method)

    #report performance if label exist
    if options['isAnomaly']:
        report['file'] = file_name

    loss_analysis.to_csv(f'encodings/thresholds/thresholds_{method}_{file_name}_{run_id}.csv', index=False)

    return report, loss_analysis

# Main function to run the anomaly detection pipeline
def run_anomaly_detection_pipeline(file_path, options, output_file_detect, autoencoder_type, run_id):
    data, label = load_and_prepare_data(file_path, options)

    if options['timestamp'] is not None:
        data[options['timestamp']] = data[options['timestamp']].apply(pm.parse_time)
    data = data.sort_values(by=[options['case_id'], options['timestamp']])

    best_f1 = -1
    best_df = pd.DataFrame(columns=[options['case_id'], 'label'])
    file_name = os.path.basename(file_path)

    for method in ['one-hot', 'word2vec', 'llm_chunked', 'llm_trained','sentence_transformer']:
        print(f"Encode the Event Log using: {method}")
        encoded, (encoding_duration, process_duration, length) = encode_data(method, data, options, run_id, file_name)

        report, loss_analysis = train_and_evaluate_anomaly_detector(encoded, label, options, output_file_detect, method, autoencoder_type, run_id, file_name)

        current_f1 = report['f1-score_weighted avg'].iloc[0]
        if best_f1 < current_f1:
            best_f1 = current_f1
            best_df = loss_analysis.drop(['loss'], axis=1)
        report['encoding_duration'] = encoding_duration
        report['process_duration'] = process_duration
        report['length'] = length

        if not os.path.exists(output_file_detect):
            with open(output_file_detect, 'w', newline='') as f:
                report.to_csv(f, header=True, index=False)
        else:
            with open(output_file_detect, 'a', newline='') as f:
                report.to_csv(f, header=False, index=False)

    return data.merge(best_df, on=options['case_id'], how='right')


def run_anomaly_detection_pipeline_unsupervised(file_path, options, output_file_detect, autoencoder_type, method, run_id):
    data, label = load_and_prepare_data(file_path, options)
    df = pd.DataFrame(columns=[options['case_id'], 'label'])

    if options['timestamp'] is not None:
        data[options['timestamp']] = data[options['timestamp']].apply(pm.parse_time)
    data.sort_values(by=[options['case_id'], options['timestamp']], inplace=True)

    file_name = os.path.basename(file_path)

    encoded, *_ = encode_data(method, data, options, run_id, file_name)
    _, loss_analysis = train_and_evaluate_anomaly_detector(encoded, label, options, output_file_detect, method, autoencoder_type, run_id, file_name)
    df['label'] = loss_analysis['label']
    df[options['case_id']] = loss_analysis[options['case_id']]
    return data.merge(df, on=options['case_id'], how='right')



# Function to prepare data for training sequence models
def prepare_sequence_model_data(data, options):

    normal_data = data[data['label'] == 0]
    grouped = normal_data.groupby(options['case_id']).apply(lambda x: x.sort_values(options['timestamp'])).reset_index(drop=True)
    #transforms timestamp into text
    #normal_data[options['timestamp']] = normal_data[options['timestamp']].apply(pm.parse_time)
    grouped_data = grouped.groupby(options['case_id'])[
        options['columns2encode'] + [options['timestamp']] + [options['case_id']]
        ]

    event_sequences = grouped_data.apply(lambda x: [tuple(row) for row in x.to_records(index=False)]).tolist()
    print("Creating Eventually-Follow Pairs")
    eventually_follow_pairs = epg.get_eventually_follow_pairs(event_sequences)
    print("Eventually-Follow Pairs created")
    print('Inject Anomalies')
    combined_dataset = epg.inject_anomalies(eventually_follow_pairs, 100, event_structure=options['columns2encode']+[options['timestamp']]+[options['case_id']], attributes=options['attributes'])
    print('Anomalies Injected')
    return epg.prepare_training_data(combined_dataset)

# Function to prepare data for training sequence models
def prepare_sequence_model_anomalies(data, options):
    df = data[data['label'] == 1]
    grouped = df.groupby(options['case_id']).apply(lambda x: x.sort_values(options['timestamp'])).reset_index(drop=True)
    grouped = grouped.groupby(options['case_id'])[
        options['columns2encode'] + [options['timestamp']] + [options['case_id']]
        ]
    event_sequences = grouped.apply(lambda x: [tuple(row) for row in x.to_records(index=False)]).tolist()

    eventually_follow_pairs = epg.get_eventually_follow_pairs(event_sequences)
    return epg.prepare_anomaly_data(eventually_follow_pairs)

# Function to tokenize data
def get_Tokenizer(model_type):
    if model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    else:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# Function to train a sequence classification model
def train_sequence_model(model_type, train_encodings, train_labels, val_encodings, val_labels, csv_path, tokenizer):

    train_dataset = create_tf_dataset(train_encodings, train_labels)
    val_dataset = create_tf_dataset(val_encodings, val_labels)

    # Check for NaN values in the data
    if np.any(np.isnan(train_encodings["input_ids"])):
        print("Found NaN in train_encodings['input_ids']")
    if np.any(np.isnan(train_labels)):
        print("Found NaN in train_labels")

    if model_type == 'bert':
        model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3, from_pt=True)
    else:
        model = TFGPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=3)
        model.config.pad_token_id = tokenizer.pad_token_id

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    num_training_examples = len(train_encodings["input_ids"])  # Assuming train_encodings is a dictionary


    steps_per_epoch = num_training_examples // BATCH_SIZE
    total_steps = steps_per_epoch * NUM_EPOCHS

    # Create the optimizer with warm-up steps and weight decay
    '''
    optimizer, _ = create_optimizer(
        init_lr=5e-5,
        num_warmup_steps=500,
        num_train_steps=total_steps,
        weight_decay_rate=0.01
    )
    '''
    #optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    model.fit(
        train_dataset.shuffle(1000).batch(BATCH_SIZE),
        validation_data=val_dataset.batch(BATCH_SIZE),
        epochs=NUM_EPOCHS
    )


    return model



# Main execution flow
if __name__ == "__main__":
    dir_datasets = 'Data/RunSets/Supervised'  # Base directory containing datasets
    anomaly_detector = sys.argv[1]
    run_id = sys.argv[2]
    random.seed(RANDOM_SEED)

    # Walk through all directories and files in the specified directory
    for dirpath, dirnames, files in os.walk(dir_datasets):
        for file in files:
            if 'Encodings' in dirpath:
                continue
            if file.endswith('.csv') and not file.startswith('.'):
                print(f"Model applied on: {file}")
                file_path = os.path.join(dirpath, file)
                #Load dataset specific information
                parser = pm.get_configuration(file)
                args, unknown = parser.parse_known_args()
                options = vars(args)

                # Encode Data, Run Anomaly Detector
                if options['isAnomaly']:
                    if 'forest' in anomaly_detector:
                        data = run_anomaly_detection_pipeline(file_path, options, f'reports/report_anomaly_detection_iforest_{run_id}.csv', 'iforest', run_id)
                    else:
                        data = run_anomaly_detection_pipeline(file_path, options, f'reports/report_anomaly_detection_{run_id}.csv', 'autoencoder', run_id)
                else:
                    if 'forest' in anomaly_detector:
                        data = run_anomaly_detection_pipeline_unsupervised(file_path, options, f'reports/report_anomaly_detection_iforest_{run_id}.csv', 'iforest',method='word2vec', run_id=run_id)
                    else:
                        data = run_anomaly_detection_pipeline_unsupervised(file_path, options, f'reports/report_anomaly_detection_{run_id}.csv', 'autoencoder', method='word2vec', run_id=run_id)
                print("All Encodings have been generated")


                # Create artificial labels and event pairs and create eventually follow pairs
                inputs, labels, case_ids = prepare_sequence_model_data(data, options)


                #Split the data into train and validation set
                train_texts, val_texts, train_labels, val_labels, train_case_ids, val_case_ids = train_test_split(inputs, labels, case_ids, test_size=0.2, random_state=RANDOM_SEED)

                # Convert labels to numpy arrays
                train_labels = np.array(train_labels)
                val_labels = np.array(val_labels)


                # Load the defined tokenizer
                tokenizer = get_Tokenizer('bert')

                #Create the encodings
                train_encodings = tokenizer(train_texts, truncation=True, padding=True)
                val_encodings = tokenizer(val_texts, truncation=True, padding=True)

                # Train the model
                print("Train LLM on the created Evenutally-Follow Pairs")
                model = train_sequence_model('bert', train_encodings=train_encodings,
                                                      train_labels=train_labels, val_encodings=val_encodings,
                                                      val_labels=val_labels, csv_path=f'encodings/Evaluation_Pairs_{file}',
                                                      tokenizer=tokenizer)
                print("Create Evenutally-Follow Pairs for Anomalous Data")
                inputs, anomaly_case_ids = prepare_sequence_model_anomalies(data, options)
                print("Evenutally-Follow Pairs created")
                anomaly_encodings = tokenizer(inputs, truncation=True, padding=True, return_tensors='tf')
                dataset = tf.data.Dataset.from_tensor_slices(dict(anomaly_encodings)).batch(batch_size=BATCH_SIZE)
                print("Apply trained LLM on Anomalous Eventually-Follow Pairs")
                predictions = []
                for batch in dataset:
                    outputs = model(batch)
                    logits = outputs.logits
                    preds = tf.argmax(logits, axis=-1)
                    predictions.extend(preds.numpy().tolist())

                df = pd.DataFrame({
                    'input': inputs,
                    'prediction': predictions,
                    'case_id': anomaly_case_ids
                })

                label_mapping = {'swap_anomaly': 2, 'attribute_anomaly': 1, 'normal': 0}
                # Inverse mapping
                inverse_label_mapping = {v: k for k, v in label_mapping.items()}

                df['label'] = df['prediction'].map(inverse_label_mapping)
                df.to_csv(f'encodings/Anomalies_Explained_{run_id}_{file}', index=False)


                for row in df.itertuples():

                    actA, actB, duration = row.input.split(';')

                    if row.label == 'normal':
                        text = f"The data is normal"
                    elif row.label == 'swap_anomaly':
                        text = f"CASE {row.case_id}: A swap anomaly occurred because {actA.split(':')[0]} happened before {actB.split(':')[0]}"
                        print(text)
                    elif row.label == 'attribute_anomaly':
                        text =  f"CASE {row.case_id}: Attribute anomaly - Time between {actA.split(':')[0]} and {actB.split(':')[0]} is with {duration} outside of the norm"
                        print(text)
                    else:
                        text = "Unknown label"
                        print(text)



                tf.keras.backend.clear_session()


    print('code finished')

