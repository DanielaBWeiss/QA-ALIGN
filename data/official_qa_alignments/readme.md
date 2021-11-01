**Gold data**

f=Files contain the following important columns:
abs_sent_id_1 & abs_sent_id_2 (unique sentence id, unique across all data sources)
text_1,text_2,prev_text_1, and prev_text_2 - are the two candidate sentences for alignments. The previous sentences are for context (shown to workers and for the model).
qas_1,qas_2 - are the sets of QASRL QAs for each sentence. For test and dev they were created by workers, while in train, the QASRL parser generated them.
alignments - the aligned qas that workers have matched. This is a list of qa-alignments, where a single alignment looks like this:

```
 {'sent1': [{'qa_uuid': '33_1ecbplus~!~8~!~195~!~12~!~charged~!~4082',
    'verb': 'charged',
    'verb_idx': 12,
    'question': 'Who was charged?',
    'answer': 'the two youths',
    'answer_range': '9:11'}],
  'sent2': [{'qa_uuid': '33_8ecbplus~!~3~!~328~!~11~!~accused~!~4876',
    'verb': 'accused',
    'verb_idx': 11,
    'question': 'Who was accused of something?',
    'answer': 'two men',
    'answer_range': '9:10'}]}
```

Where the two keys for sent1&2 contain a list each of the aligned QAs from that sentence.
Note that this is a single alignment and may contain multiple QAs (4% of the data may contain many-many although most of the time its a 2-1}.

//Make sure to run eval() on the alignments column to use with pandas.