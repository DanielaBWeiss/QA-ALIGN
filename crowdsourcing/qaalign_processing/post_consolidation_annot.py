import pandas as pd
import argparse
from tabulate import tabulate

def convert_time(x):
    if x == '{}':
        return 0
    else:
        return float(x)

def remove_pre_agreed(row):
    aligns = []
    for item in row['all_agreed_aligns']:
        if item not in row['aligned_qas']:
            aligns.append(item)
    return aligns

def clean_columns(df):
    if 'AssignmentId' not in df.columns and 'worker_id' not in df.columns:
        df = df[["Input.key",'WorkerId', 'WorkTimeInSeconds',
           'Input.abs_sent_id_1', 'Input.text_1','Input.text_1_html','Input.text_2','Input.text_2_html',
           'Input.prev_text_1', 'Input.abs_sent_id_2','Input.prev_text_2',
           'Input.worker_id_1', 'Input.worker_id_2', 'Input.aligned_qas',
             'Input.disagreements_a', 'Input.disagreements_b',
             'Answer.all_agreed_aligns', 'Answer.all_dis_agreed_aligns',
             'Answer.feedback', 'Answer.hit-submit', 'Answer.time', 'Approve',
             'Reject']].copy()

        df.columns =  ['key','worker_id', 'work_time_in_seconds',
           'abs_sent_id_1',  'text_1','text_1_html','text_2','text_2_html',
           'prev_text_1', 'abs_sent_id_2', 'prev_text_2','worker_id_1','worker_id_2',
           'aligned_qas','dissagreements_a','disagreements_b','all_agreed_aligns','all_dis_agreed_aligns',
            'feedback','hit-submit','submit_time','Approve','Reject']
    elif "WorkerId" in df.columns:
        df = df[["Input.key",'WorkerId', 'AssignmentId','WorkTimeInSeconds',
           'Input.abs_sent_id_1', 'Input.text_1','Input.text_1_html','Input.text_2','Input.text_2_html',
           'Input.prev_text_1', 'Input.abs_sent_id_2','Input.prev_text_2',
           'Input.worker_id_1', 'Input.worker_id_2', 'Input.aligned_qas',
             'Input.disagreements_a', 'Input.disagreements_b',
             'Answer.all_agreed_aligns', 'Answer.all_dis_agreed_aligns',
             'Answer.feedback', 'Answer.hit-submit', 'Answer.time', 'Approve',
             'Reject']].copy()

        df.columns = ['key','worker_id', 'assignment_id','work_time_in_seconds',
           'abs_sent_id_1', 'text_1','text_1_html','text_2','text_2_html',
           'prev_text_1', 'abs_sent_id_2', 'prev_text_2','worker_id_1','worker_id_2',
           'aligned_qas','dissagreements_a','disagreements_b','all_agreed_aligns','all_dis_agreed_aligns',
            'feedback','hit-submit','submit_time','Approve','Reject']

    if 'assignment_id' not in df.columns:
        df['assignment_id'] = ['NA']*len(df)
    if 'answers' in df.columns or 'alignments' in df.columns:
        if 'alignments' not in df.columns:
            df.rename(columns={"answers": "alignments"}, inplace=True);
        df['work_time_in_min'] = df['work_time_in_seconds'] / 60
        df['submit_time'] = df['submit_time'].apply(lambda x: convert_time(x))
        df['hit_count'] = df.groupby(["worker_id"])['worker_id'].transform("count")
        return df
    df['all_agreed_aligns'] = df['all_agreed_aligns'].apply(lambda x: eval(x))
    df['aligned_qas'] = df['aligned_qas'].apply(lambda x: eval(x))
    df['added_aligns'] = df.apply(lambda x: remove_pre_agreed(x), axis=1)
    df['work_time_in_min'] = df['work_time_in_seconds']/60
    df['submit_time'] = df['submit_time'].apply(lambda x: convert_time(x))
    df['len_annotations'] = df.apply(lambda x: len(x['added_aligns']),axis=1)
    df['avg_num_aligns_added'] = df.groupby('worker_id')['len_annotations'].transform('mean')
    df['hit_count'] = df.groupby(["worker_id"])['worker_id'].transform("count")
    if "all_agreed_aligns" in df.columns:
        df.rename(columns={"all_agreed_aligns": "alignments"}, inplace=True);
    df['hit_time_mean'] = df.groupby(["worker_id"])['submit_time'].transform("mean")

    #df['num_bonus_cents'] = df['hit_count']*3 #3 cents per HIT, for a total of 30 cents per hit.

    return df


def print_stats(df):
    if 'added_aligns' not in df.columns:
        workers = df.groupby(['worker_id', 'hit_count']).size().reset_index()
        workers.drop([0], axis=1, inplace=True)

        print("Workers stats on batch:")
        print(tabulate(workers, headers=workers.columns, tablefmt='orgtbl'))
        print("sample hit type")
        print(tabulate(df.groupby('worker_id')[['worker_id', 'assignment_id']].apply(lambda x: x.sample(1)),
                       tablefmt='orgtbl'))
        return
    workers = df.groupby(['worker_id', 'hit_count','hit_time_mean', 'avg_num_aligns_added']).size().reset_index()
    workers.drop([0],axis=1, inplace=True)

    print("Workers stats on batch:")
    print(tabulate(workers, headers=workers.columns, tablefmt='orgtbl'))
    print("sample hit type")
    print(tabulate(df.groupby('worker_id')[['worker_id','assignment_id']].apply(lambda x: x.sample(1)) ,tablefmt='orgtbl'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='csv file containing qa alignment annotations')

    args = parser.parse_args()
    df = pd.read_csv(args.input)
    df = clean_columns(df)
    print_stats(df)

    df.to_csv(args.input, index=False)

