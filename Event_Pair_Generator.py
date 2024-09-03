import pandas as pd
import random
import Pipeline as pm
import inflect
import multiprocessing
import numpy as np
from datetime import datetime, timedelta

duration_dic = {}

def get_eventually_follow_pairs(event_traces):
    all_pairs = []
    for trace in event_traces:
        all_pairs.extend(recursive_method(trace))

    return all_pairs

def recursive_method(trace):
    eventually_follow_pairs = []
    seen_events = set()

    for i, event1 in enumerate(trace):
        if event1[0] in seen_events:
            break
        seen_events.add(event1[0])
        seen_events2 = set()
        for j in range(i + 1, len(trace)):
            event2 = trace[j]
            if event2[0] in seen_events2:
                break
            if event2[0] in seen_events:
                eventually_follow_pairs.extend(recursive_method(trace[i+1:j]))
                eventually_follow_pairs.extend(recursive_method(trace[j:]))
                return eventually_follow_pairs
            duration_key = (event1[0], event2[0])  # Explicitly define the pair order
            timestamp1 = event1[-2]
            timestamp2 = event2[-2]
            duration = (timestamp2 - timestamp1)
            if isinstance(duration, pd.Timedelta):
                duration = (timestamp2 - timestamp1).total_seconds()
            else:
                duration = (timestamp2 - timestamp1).astype('timedelta64[s]').astype(int)

            if duration_key in duration_dic:
                total_duration, count = duration_dic[duration_key]
                total_duration += duration
                count += 1
                duration_dic[duration_key] = (total_duration, count)
            else:
                duration_dic[duration_key] = (duration, 1)
            event_pair = (event1, event2)
            eventually_follow_pairs.append(event_pair)
            seen_events2.add(event2[0])

    return eventually_follow_pairs

def numpy_datetime_to_python_datetime(numpy_datetime):
    return datetime.utcfromtimestamp(numpy_datetime.astype('O') / 1e9)

def get_average_durations(duration_dic):
    average_duration_dic = {}
    for duration_key, (total_duration, count) in duration_dic.items():
        average_duration_dic[duration_key] = total_duration / count
    return average_duration_dic

def get_eventually_follow_pairs_multi(event_sequences, num_workers: int = None):
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    # Split the event sequences into chunks for each worker
    chunk_size = len(event_sequences) // num_workers
    chunks = [event_sequences[i:i + chunk_size] for i in range(0, len(event_sequences), chunk_size)]

    # Create a pool of worker processes
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Map the process_sequence_chunk function to each chunk
        results = pool.map(get_eventually_follow_pairs, chunks)

    # Combine the results from all processes
    all_pairs = []
    for result in results:
        all_pairs.extend(result)

    return all_pairs

# Function to generate swap anomalies
def create_swap_anomalies(event_pairs, num_anomalies, used_pairs, max_attempts = 100):
    swap_anomalies = []
    attempts = 0
    max_attempts = num_anomalies * 1.2
    while len(swap_anomalies) < num_anomalies and attempts < max_attempts:
        pair = random.choice(event_pairs)
        if pair in used_pairs:
            continue
        swapped_pair = (pair[1], pair[0])
        if swapped_pair not in event_pairs and swapped_pair not in used_pairs:
            swap_anomalies.append((swapped_pair, 'swap_anomaly'))
            used_pairs.add(pair)
            used_pairs.add(swapped_pair)
    print('finished_anomaly_generation')
    if len(swap_anomalies) < num_anomalies:
        print(f"Warning: Only {len(swap_anomalies)} out of {num_anomalies} swap anomalies could be generated.")

    return swap_anomalies

# Function to generate attribute anomalies
def create_attribute_anomalies(event_pairs, num_anomalies, used_pairs, attribute_indices):
    attribute_anomalies = []
    while len(attribute_anomalies) < num_anomalies:
        pair = random.choice(event_pairs)
        if pair in used_pairs:
            continue
        event1, event2 = pair

        # Create a mutation in one of the attributes of the second event
        mutated_event2 = list(event2)
        attr_to_change = random.choice(attribute_indices)
        mutated_event2[attr_to_change] = 'mutated_attribute'
        mutated_event2 = tuple(mutated_event2)

        anomaly_pair = (event1, mutated_event2)
        if anomaly_pair not in attribute_anomalies and anomaly_pair not in used_pairs:
            attribute_anomalies.append((anomaly_pair, 'attribute_anomaly'))
            used_pairs.add(pair)
            used_pairs.add(anomaly_pair)
    return attribute_anomalies

def create_duration_anomalies(event_pairs, num_anomalies, used_pairs, timestamp_index, min_large_deviation=timedelta(hours=24)):
    attribute_anomalies = []
    while len(attribute_anomalies) < num_anomalies:
        pair = random.choice(event_pairs)
        if pair in used_pairs:
            continue
        event1, event2 = pair

        # Create a mutation in the timestamp attribute of the second event
        mutated_event1 = list(event1)
        mutated_event2 = list(event2)

        # Parse the original timestamp of the first event
        timestamp1 = pm.parse_time(mutated_event1[timestamp_index])



        duration_key = (mutated_event1[0], mutated_event2[0])
        average_duration_dict = get_average_durations(duration_dic)

        avg_duration = average_duration_dict[duration_key]

        if random.choice([True, False]):
            # Large deviation: Add at least avg_duration times a random factor
            #new_timestamp = timestamp2 + min_large_deviation + timedelta(minutes=random.randint(1, 180))
            random_duration_seconds = 86400 + random.randint(1, 180) + int(avg_duration / 60)
            new_timestamp = timestamp1 + np.timedelta64(random_duration_seconds, 's')
        else:
            # Small deviation: Add a small random amount (up to a few seconds)
            #new_timestamp = timestamp1 + timedelta(seconds=random.randint(1, 180))
            random_duration_seconds = random.randint(1, 180)
            new_timestamp = timestamp1 + np.timedelta64(random_duration_seconds, 's')

        # Update the timestamp in the event
        mutated_event2[timestamp_index] = new_timestamp
        mutated_event2 = tuple(mutated_event2)

        anomaly_pair = (event1, mutated_event2)
        if anomaly_pair not in attribute_anomalies and anomaly_pair not in used_pairs:
            attribute_anomalies.append((anomaly_pair, 'attribute_anomaly'))
            used_pairs.add(pair)
            used_pairs.add(anomaly_pair)
    return attribute_anomalies

# Function to inject anomalies into event pairs
def inject_anomalies(event_pairs, anomaly_percentage, event_structure = ['event_name', 'name', 'day'], attributes = ['name', 'day']):
    attribute_indices = [event_structure.index(attr) for attr in attributes]
    num_pairs = len(event_pairs)
    total_anomalies = int(num_pairs * anomaly_percentage / 100)
    anomalies_per_method = total_anomalies // 2  # Split evenly between methods

    used_pairs = set()

    swap_anomalies = create_swap_anomalies(event_pairs, total_anomalies, used_pairs)
    #duration_anomalies = create_duration_anomalies(event_pairs=event_pairs, num_anomalies=anomalies_per_method, used_pairs=used_pairs, timestamp_index=-2)

    normal_pairs = [(pair, 'normal') for pair in event_pairs]

    combined_dataset = normal_pairs + swap_anomalies# + duration_anomalies

    num_normal_pairs = len(normal_pairs)
    num_anomalous_pairs = len(swap_anomalies)
    print(f"Number of normal event pairs: {num_normal_pairs}")
    print(f"Number of anomalous event pairs: {num_anomalous_pairs}")

    return combined_dataset

def map_labels(label):
    label_mapping = {'swap_anomaly': 2, 'attribute_anomaly': 1, 'normal': 0}
    return label_mapping.get(label, 0)

# Function to prepare training data
def prepare_training_data(combined_dataset):
    inputs = []
    labels = []
    case_ids = []
    for (event1, event2), label in combined_dataset:
        # Extract case_id from the events
        case_id = event1[-1]
        timestamp1 = event1[-2]
        timestamp2 = event2[-2]
        #print(timestamp1, timestamp2)
        duration = abs(pm.parse_time(timestamp2) - pm.parse_time(timestamp1))

        #print(case_id, timestamp1, timestamp2, duration, format_timedelta(duration))
        # Prepare input string without the case_id
        event1_str = " : ".join(map(str, event1[:-2]))
        event2_str = " : ".join(map(str, event2[:-2]))
        # Transform duration to a readable string
        duration_str = format_timedelta(duration)

        input_str = " ; ".join([event1_str, event2_str, duration_str])
        inputs.append(input_str)

        # Convert label to binary
        labels.append(map_labels(label))

        # Store case_id
        case_ids.append(case_id)

    return inputs, labels, case_ids

def format_timedelta(td):
    # Create an inflect engine instance
    p = inflect.engine()

    if isinstance(td, timedelta):
        # Handle timedelta logic
        total_seconds = td.total_seconds()
        total_seconds = int(total_seconds)
    else:
        total_seconds = td.astype('timedelta64[s]').astype(int)

    days = total_seconds // (24 * 3600)
    total_seconds %= 24 * 3600
    hours = total_seconds // 3600
    total_seconds %= 3600
    minutes = total_seconds // 60
    seconds = total_seconds % 60

    parts = []
    if days:
        day_word = p.number_to_words(days)
        parts.append(f"{day_word} day{'s' if days != 1 else ''}")
    if hours:
        hour_word = p.number_to_words(hours)
        parts.append(f"{hour_word} hour{'s' if hours != 1 else ''}")
    if minutes:
        minute_word = p.number_to_words(minutes)
        parts.append(f"{minute_word} minute{'s' if minutes != 1 else ''}")
    if seconds or not parts:  # Add seconds if no other parts are present
        second_word = p.number_to_words(seconds)
        parts.append(f"{second_word} second{'s' if seconds != 1 else ''}")

    return ', '.join(parts)

def prepare_anomaly_data(combined_dataset):
    inputs = []
    case_ids = []
    for (event1, event2) in combined_dataset:
        # Extract case_id from the events
        case_id = event1[-1]
        timestamp1 = event1[-2]
        timestamp2 = event2[-2]
        duration = abs(pm.parse_time(timestamp2) - pm.parse_time(timestamp1))


        # Prepare input string without the case_id
        event1_str = " : ".join(map(str, event1[:-2]))
        event2_str = " : ".join(map(str, event2[:-2]))
        duration_str = format_timedelta(duration)

        input_str = " ; ".join([event1_str, event2_str, duration_str])
        inputs.append(input_str)
        # Store case_id
        case_ids.append(case_id)
    return inputs, case_ids

