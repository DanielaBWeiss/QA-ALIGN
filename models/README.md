# Training

## Data Processing 
To process the alignments data (found under data/official_qa_alignments), run preprocess_train.py with the following commands:

```
python preprocess_train.py --input_dir
../data/official_qa_alignments/
--pred_token
--neg_sample
--out_dir
../data/official_qa_alignments/
--out_name
processed_wpred_wneg.csv
```

This file iterates over every unique *key* sentence pair, and create all possible QA-QA candidate alignments.
It negatively samples with a ratio k=10. 
An example row candidate in the output looks like this:
> Who did someone [P] fire [/P] ? [Q] The Philadelphia 76ers [P] fired [/P] [A] coach Maurice Cheeks [/A] on Saturday, one day after the team continued its slide with a season-worst offensive effort, dpa reported.
> Who was [P] fired [/P] ? [Q] If you don’t know by now: you disappoint in the NBA, you get canned. Today,
[A] Maurice Cheeks [/A] became the fifth coach [P] fired [/P] within the first quarter of the season.

The above example begins with the candidate question separated by special token [Q], and continues with the sentence context (target sentence + previous sentence if exists). Target question predicate is marked ([P]) both in the question and the sentence, along with the question answer ([A]).
More info can be found in the paper.

## Model training

The training model expects input as formatted in *preprocess_train.py*. Pretrained model used is CorefRoberta, with special tokens parameter passed ([P], [A], [Q]).
Trained model used in the paper can be found in Huggingface (will upload and attach a link soon).
Rest of the details can be found in the paper and the appendix.

```
python preprocess_train.py--input_dir ../data/official_qa_alignments/ --batch_size 4 --gpus "1" --exp_name coref-roberta --max_epochs 6 --distributed_backend ddp --file_ext wneg_wpred --special_token --model nielsr/coref-roberta-base
```


## Model eval and prediction


Run qa_alignment_predict.py on any *_processed file created by *preprocess_train.py* to produce prediction on candidate QA-QA pairs.
Example run:
```
python qa_alignment_predict.py --file_path ../data/predict/dev_thadani_pairs_wqas_processed.csv --model <qa-align model> --output <corefroberta-pred.csv> --batch_size 16 --gpu "2" --exp_name coref-roberta --threshold 0.44
```
Pass threshold for pred=1. We tried 0.44 for the fusion experiment.

Run qa_alignment_eval.py on qa_alignment_predict.py's output if you have gold. pass --find_thredhold if evaluating dev. Otherwise pass --threshold followed the threshold you found when evaluating dev (if this is test).


## Fusion Experiment

For new data, create sentence pairs + QASRL qas using the notebook **Prepare Data For QA-Align Prediction** notebook. Once prepared (an example file "dev_fusion_pairs_wqas.csv"), run preprocess_train.py, and then the _predict.py file to predict on your data.

To create the fusion training files for the BART model, run create_aligned_fusions.py.

```
python create_aligned_fusions.py
--pred_file
<test-fusion-corefroberta-pred.csv> //predicted file from qa_alignment_predict.py
--fusion_file
<fusion_test.csv>
--qasrl_file
<test_fusion_sents_qasrl_gen.csv> //created in the above mentioned notebook using QASRL predicted files
--data_split
test
--out_dir
<eval dir>
```
This file creates 20 shuffled versions of the fusion train/dev/test dataset, since BART is highly sensitive to the ordering of the fusion sentences in the input.

**TODO: input link to the transformers repo used to train the BART fusion models.**