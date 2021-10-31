import pandas as pd
import argparse
from tabulate import tabulate

def convert_time(x):
    if x == '{}':
        return 0
    else:
        return float(x)

def clean_columns(df):
    if 'AssignmentId' not in df.columns and 'worker_id' not in df.columns:
        df = df[['WorkerId', 'WorkTimeInSeconds', 'Input.abs_scu_id',
                 'Input.abs_sent_id_1', 'Input.text_1_html','Input.text_2_html','Input.text_1', 'Input.prev_text_sent_id_1',
                 'Input.prev_text_1', 'Input.abs_sent_id_2', 'Input.text_2',
                 'Input.prev_text_sent_id_2', 'Input.prev_text_2', 'Input.year',
                 'Input.topic', 'Input.source', 'Input.source-doc_1',
                 'Input.source-doc_2', 'Input.source-data',
                 'Input.key', 'Answer.all_my_answers',
                 'Answer.feedback', 'Answer.hit-submit', 'Answer.time', 'Approve',
                 'Reject']].copy()

        df.columns = ['worker_id', 'work_time_in_seconds', 'abs_scu_id',
                      'abs_sent_id_1','text_1_html','text_2_html', 'text_1', 'prev_text_sent_id_1',
                      'prev_text_1', 'abs_sent_id_2', 'text_2',
                      'prev_text_sent_id_2', 'prev_text_2', 'year',
                      'topic', 'source', 'source-doc_1', 'source-doc_2', 'source-data',
                      'key', 'answers', 'feedback', 'hit-submit', 'submit_time', 'Approve', 'Reject']
    elif "WorkerId" in df.columns:
        df = df[['WorkerId', 'AssignmentId','WorkTimeInSeconds', 'Input.abs_scu_id',
           'Input.abs_sent_id_1', 'Input.text_1_html','Input.text_2_html','Input.text_1', 'Input.prev_text_sent_id_1',
           'Input.prev_text_1', 'Input.abs_sent_id_2', 'Input.text_2',
           'Input.prev_text_sent_id_2', 'Input.prev_text_2', 'Input.year',
           'Input.topic', 'Input.source', 'Input.source-doc_1',
           'Input.source-doc_2', 'Input.source-data',
           'Input.key','Answer.all_my_answers',
           'Answer.feedback', 'Answer.hit-submit', 'Answer.time', 'Approve',
           'Reject']].copy()

        df.columns = ['worker_id', 'assignment_id','work_time_in_seconds', 'abs_scu_id',
           'abs_sent_id_1', 'text_1_html','text_2_html', 'text_1', 'prev_text_sent_id_1',
           'prev_text_1', 'abs_sent_id_2', 'text_2',
           'prev_text_sent_id_2', 'prev_text_2', 'year',
           'topic', 'source', 'source-doc_1', 'source-doc_2', 'source-data',
           'key', 'answers', 'feedback', 'hit-submit', 'submit_time', 'Approve', 'Reject']

    if 'assignment_id' not in df.columns:
        df['assignment_id'] = ['NA']*len(df)
    df['work_time_in_min'] = df['work_time_in_seconds']/60
    df['answers'] = df['answers'].apply(lambda x: eval(x))
    df['submit_time'] = df['submit_time'].apply(lambda x: convert_time(x))
    df['len_annotations'] = df['answers'].apply(lambda x: len(x))
    df['avg_num_annotations'] = df.groupby('worker_id')['len_annotations'].transform('mean')
    df['hit_count'] = df.groupby(["worker_id"])['worker_id'].transform("count")
    df['hit_time_mean'] = df.groupby(["worker_id"])['submit_time'].transform("mean")

    df['num_bonus_cents'] = df['hit_count']*3 #3 cents per HIT, for a total of 30 cents per hit.

    return df

def print_stats(df):
    workers = df.groupby(['worker_id', 'hit_count','num_bonus_cents','hit_time_mean', 'avg_num_annotations']).size().reset_index()
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

