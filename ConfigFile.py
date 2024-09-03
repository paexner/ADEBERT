from argparse import ArgumentParser


BATCH_SIZE = 64
NUM_EPOCHS = 3
RANDOM_SEED = 123

def get_args_artifical_DS():

    parser = ArgumentParser()
    parser.add_argument('--name', default='Artificial', type=str,
                        help='The name of the dataset')
    parser.add_argument('--output_dir', default='./encodings/', type=str,
                        help='The output directory of parsing results')
    parser.add_argument('--seed', default=7, type=int, help='random seed (default: 7)')
    parser.add_argument('--columns2use', default=['case_id', 'name', 'user', 'day', 'timestamp'], type=list, help='Array of column names (default: case_id, timestamp, activity_name)')
    parser.add_argument('--columns2encode', default=['name', 'user', 'day'], type=list, help='Array of column names (default: case_id, timestamp, activity_name)')
    parser.add_argument('--attributes', default=['user', 'day'], type=list, help='Array of column names (default: case_id, timestamp, activity_name)')
    parser.add_argument('--case_id', default='case_id', type=str, help='Column name of the case grouping (default: case_id)')
    parser.add_argument('--timestamp', default='timestamp', type=str, help='Column name of the timestamp (default: timestamp)')
    parser.add_argument('--isAnomaly', default='isAnomaly', type=str, help='Column name of the label (default: isAnomaly)')
    return parser

def get_args_BPIC12():

    parser = ArgumentParser()
    parser.add_argument('--name', default='BPIC12', type=str,
                        help='The name of the dataset')
    parser.add_argument('--output_dir', default='./encodings/', type=str,
                        help='The output directory of parsing results')
    parser.add_argument('--seed', default=7, type=int, help='random seed (default: 7)')
    parser.add_argument('--columns2use', default=['case_id', 'name', 'org:resource', 'timestamp'], type=list, help='Array of column names (default: case_id, timestamp, activity_name)')
    parser.add_argument('--columns2encode', default=['name', 'org:resource'], type=list, help='Array of column names (default: case_id, timestamp, activity_name)')
    parser.add_argument('--attributes', default=['org:resource'], type=list, help='Array of column names (default: case_id, timestamp, activity_name)')
    parser.add_argument('--case_id', default='case_id', type=str, help='Column name of the case grouping (default: case_id)')
    parser.add_argument('--timestamp', default='timestamp', type=str, help='Column name of the timestamp (default: timestamp)')
    parser.add_argument('--isAnomaly', default='isAnomaly', type=str, help='Column name of the label (default: isAnomaly)')
    parser.add_argument('--download_datasets', default=True, type=bool, help='Download datasets (default: True)')
    return parser

def get_args_BPIC13():

    parser = ArgumentParser()
    parser.add_argument('--name', default='BPIC13', type=str,
                        help='The name of the dataset')
    parser.add_argument('--output_dir', default='./encodings/', type=str,
                        help='The output directory of parsing results')
    parser.add_argument('--seed', default=7, type=int, help='random seed (default: 7)')
    parser.add_argument('--columns2use', default=['case_id', 'name', 'org_group', 'org_resource', 'resource_country', 'timestamp'], type=list, help='Array of column names (default: case_id, timestamp, activity_name)')
    parser.add_argument('--columns2encode', default=['name', 'org:group', 'org:resource', 'resource country'], type=list, help='Array of column names (default: case_id, timestamp, activity_name)')
    parser.add_argument('--attributes', default=['org:group', 'org:resource', 'resource country'], type=list, help='Array of column names (default: case_id, timestamp, activity_name)')
    parser.add_argument('--case_id', default='case_id', type=str, help='Column name of the case grouping (default: case_id)')
    parser.add_argument('--timestamp', default='timestamp', type=str, help='Column name of the timestamp (default: timestamp)')
    parser.add_argument('--isAnomaly', default='isAnomaly', type=str, help='Column name of the label (default: isAnomaly)')
    parser.add_argument('--download_datasets', default=True, type=bool, help='Download datasets (default: True)')
    return parser

def get_args_hospitalbilling():

    parser = ArgumentParser()
    parser.add_argument('--name', default='hospitalbilling', type=str,
                        help='The name of the dataset')
    parser.add_argument('--output_dir', default='./encodings/', type=str,
                        help='The output directory of parsing results')
    parser.add_argument('--seed', default=7, type=int, help='random seed (default: 7)')
    parser.add_argument('--columns2use', default=['case_id', 'name', 'timestamp'], type=list, help='Array of column names (default: case_id, timestamp, activity_name)')
    parser.add_argument('--columns2encode', default=['name'], type=list, help='Array of column names (default: case_id, timestamp, activity_name)')
    parser.add_argument('--attributes', default=[], type=list, help='Array of column names (default: case_id, timestamp, activity_name)')
    parser.add_argument('--case_id', default='case_id', type=str, help='Column name of the case grouping (default: case_id)')
    parser.add_argument('--timestamp', default='timestamp', type=str, help='Column name of the timestamp (default: timestamp)')
    parser.add_argument('--isAnomaly', default='isAnomaly', type=str, help='Column name of the label (default: isAnomaly)')
    parser.add_argument('--download_datasets', default=True, type=bool, help='Download datasets (default: True)')
    return parser

def get_args_PermitLog():

    parser = ArgumentParser()
    parser.add_argument('--name', default='PermitLog', type=str,
                        help='The name of the dataset')
    parser.add_argument('--output_dir', default='./encodings/', type=str,
                        help='The output directory of parsing results')
    parser.add_argument('--seed', default=7, type=int, help='random seed (default: 7)')
    parser.add_argument('--columns2use', default=['concept:name', 'case:id', 'org:resource', 'org:role', 'time:timestamp',
                                                  'case:TotalDeclared', 'case:OverspentAmount', 'case:RequestedBudget'], type=list, help='Array of column names (default: case_id, timestamp, activity_name)')
    parser.add_argument('--columns2encode', default=['concept:name', 'org:resource', 'org:role',
                                                     'case:TotalDeclared', 'case:OverspentAmount', 'case:RequestedBudget'], type=list, help='Array of column names (default: case_id, timestamp, activity_name)')
    parser.add_argument('--attributes', default=['concept:name', 'org:resource', 'org:role',
                                                 'case:TotalDeclared', 'case:OverspentAmount', 'case:RequestedBudget'], type=list, help='Array of column names (default: case_id, timestamp, activity_name)')
    parser.add_argument('--case_id', default='case:id', type=str, help='Column name of the case grouping (default: case_id)')
    parser.add_argument('--timestamp', default='time:timestamp', type=str, help='Column name of the timestamp (default: timestamp)')
    parser.add_argument('--isAnomaly', default=None, type=str, help='Column name of the label (default: None)')
    return parser

def get_args_Case_Study():

    parser = ArgumentParser()
    parser.add_argument('--name', default='CaseStudy_Mortage', type=str,
                        help='The name of the dataset')
    parser.add_argument('--output_dir', default='./encodings/', type=str,
                        help='The output directory of parsing results')
    parser.add_argument('--seed', default=7, type=int, help='random seed (default: 7)')
    parser.add_argument('--columns2use', default=['CustomerId', 'subtopic', 'TimestampContact'], type=list, help='Array of column names (default: case_id, timestamp, activity_name)')
    parser.add_argument('--columns2encode', default=['subtopic'], type=list, help='Array of column names (default: case_id, timestamp, activity_name)')
    parser.add_argument('--attributes', default=[], type=list, help='Array of column names (default: case_id, timestamp, activity_name)')
    parser.add_argument('--case_id', default='CustomerId', type=str, help='Column name of the case grouping (default: case_id)')
    parser.add_argument('--timestamp', default='TimestampContact', type=str, help='Column name of the timestamp (default: timestamp)')
    parser.add_argument('--isAnomaly', default=None, type=str, help='Column name of the label (default: isAnomaly)')
    return parser