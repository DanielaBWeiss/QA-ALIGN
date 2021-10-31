import torch
from torch.utils.data import Dataset, DataLoader
import transformers

class QAAlignmentDataset(Dataset):
    TOK_ARGS = { #TODO: to read
        'add_special_tokens': True,
        #'additional_special_tokens':[BEG_SPEC_TOKEN,END_SPEC_TOKEN,BEG_PRED_TOKEN,END_PRED_TOKEN],
        'padding': 'max_length',
        'truncation': True,
        'max_length': 256,
        'return_tensors': 'pt'
    }

    def __init__(self, samples, tokenizer: transformers.PreTrainedTokenizer):
        '''
        Relevant columns: 'text_1', 'text_2', 'prev_text_1', 'prev_text_2',
        'all_agreed_aligns', 'all_dis_agreed_aligns'.
        :param data: Data contains an instance of sentence pairs per row, with all of its alignments + QAs.
        :param tokenizer:
        :param smart_negative_sampling: whether to "smartly" select negative examples for training
        '''
        self.tokenizer = tokenizer
        self.samples = samples


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item]

    def collate_samples(self, samples): #TODO: build input examples
        tok_args = dict(self.TOK_ARGS)
        texts_1 = []
        texts_2 = []
        labels = []
        keys = []
        qa1 = []
        qa2 = []
        ids_1 = []
        ids_2 = []
        for instance in samples:
            texts_1.append(instance['input_1'])
            texts_2.append(instance['input_2'])
            labels.append(instance['label'])
            keys.append(instance['key'])
            qa1.append(instance['qa_uuid_1'])
            qa2.append(instance['qa_uuid_2'])
            #ids_1.append(instance['abs_sent_id_1']) #for fusion we add this back
            #ids_2.append(instance['abs_sent_id_2'])

        batch = self.tokenizer(texts_1,texts_2, **tok_args)
        batch = dict(batch)
        # Just a python list to `torch.tensor`
        batch['label'] = torch.tensor(labels)
        batch['key'] = keys
        batch['qa1'] = qa1
        batch['qa2'] = qa2
        #batch['id_1'] = ids_1
        #batch['id_2'] = ids_2
        return batch

    def collate_samples_roberta(self, samples):  # TODO: build input examples
        tok_args = dict(self.TOK_ARGS)
        #tok_args['add_prefix_space'] = True
        texts = []
        texts_1 = []
        texts_2 = []
        labels = []
        keys = []
        qa1 = []
        qa2 = []
        #ids_1 = []
        #ids_2 = []
        for instance in samples:
            texts_1.append(instance['input_1'])
            texts_2.append(instance['input_2'])
            #texts.append("<s> "+instance['input_1'] + " </s></s> " +instance['input_2']+" </s>")
            labels.append(instance['label'])
            keys.append(instance['key'])
            qa1.append(instance['qa_uuid_1'])
            qa2.append(instance['qa_uuid_2'])
            #ids_1.append(instance['abs_sent_id_1'])
            #ids_2.append(instance['abs_sent_id_2'])

        batch = self.tokenizer(texts_1,texts_2, **tok_args)
        batch = dict(batch)
        # Just a python list to `torch.tensor`
        batch['label'] = torch.tensor(labels)
        batch['key'] = keys
        batch['qa1'] = qa1
        batch['qa2'] = qa2
        #batch['id_1'] = ids_1
        #batch['id_2'] = ids_2
        return batch

    def collate_samples_cdlm(self, samples):  # TODO: build input examples
        tok_args = dict(self.TOK_ARGS)
        #tok_args['add_prefix_space'] = True
        texts = []
        labels = []
        keys = []
        qa1 = []
        qa2 = []
        ids_1 = []
        ids_2 = []
        texts_1 = []
        texts_2 = []
        for instance in samples:
            texts_1.append(instance['input_1'])
            texts_2.append(instance['input_2'])
            #texts.append("<g> <doc-s> <s> "+instance['input_1'] + " </s> </doc-s> <doc-s> <s> " +instance['input_2']+" </s> </doc-s>")
            labels.append(instance['label'])
            keys.append(instance['key'])
            qa1.append(instance['qa_uuid_1'])
            qa2.append(instance['qa_uuid_2'])
            #ids_1.append(instance['abs_sent_id_1'])
            #ids_2.append(instance['abs_sent_id_2'])

        batch = self.tokenizer(texts_1,texts_2, **tok_args)
        batch = dict(batch)
        # Just a python list to `torch.tensor`
        batch['label'] = torch.tensor(labels)
        batch['key'] = keys
        batch['qa1'] = qa1
        batch['qa2'] = qa2
        #batch['id_1'] = ids_1
        #batch['id_2'] = ids_2
        return batch