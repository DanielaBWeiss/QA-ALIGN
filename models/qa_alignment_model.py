import logging
import os
from collections import defaultdict
from typing import Dict, Any, List
import json

import pytorch_lightning as pl
import torch
from bipartite_matching import max_bipartite_match
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only
from torch import nn
from torch.optim import AdamW
from transformers import BatchEncoding, get_linear_schedule_with_warmup

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.DEBUG)

NO_DECAY = ['bias', 'LayerNorm.weight']

def is_in_no_decay(param_name: str):
    for nd in NO_DECAY:
        if nd in param_name:
            return True
    return False

class QAAlignmentModule(pl.LightningModule):
    MODEL_FIELDS = {'input_ids', 'attention_mask', 'token_type_ids',
                    'start_positions', 'end_positions',
                    'predicate_idx'}

    def __init__(self, model: nn.Module,
                 n_grad_update_steps=0,
                 weight_decay=0.0, learning_rate=0.0, adam_eps=0.0, threshold=0.5):
        # a pre-trained transformer model
        super().__init__()

        self.model = model #sequenceClassification model
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.adam_eps = adam_eps
        self.n_grad_update_steps = n_grad_update_steps
        self.num_labels = 2
        self.threshold = threshold
        self.text_logger: SimpleTextLogger = None  # will be initialized in setup()
        #self.log_file_name = "exp.log"

    def log_file(self, data):
        with open(self.trainer.logger.log_dir+"/"+self.log_file_name, 'a') as f:
            json.dump(data, f)

    def setup(self, stage:str):
        # Called at the beginning of fit and test.
        # This is a good hook when you need to build models dynamically or adjust something about them.
        #  This hook is called on every process when using DDP.
        #  Args:
        #   stage: either 'fit' or 'test'
        if self.trainer.global_rank == 0:
            print("Initializing ----")
            self.text_logger = SimpleTextLogger.from_logger(self.logger)

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, inputs, batch_idx):
        labels = inputs["label"]
        inputs.pop("label")
        inputs.pop("key")
        inputs.pop("qa1")
        inputs.pop("qa2")
        result = self.model(**inputs, labels=labels)
        ''' #For bert
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        result = self.model(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        '''
        loss = result.loss

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return {"loss": loss}

    def validation_step(self, inputs, batch_idx):
        labels = inputs["label"]
        inputs.pop("label")
        inputs.pop("key")
        inputs.pop("qa1")
        inputs.pop("qa2")
        result = self.model(**inputs, labels=labels)
        '''
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        n_samples = len(labels)
        result = self.model(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        '''
        loss = result.loss.item()
        logits = result.logits
        #logits = self.forward(**inputs)
        labels_hat = torch.argmax(logits, dim=1)

        val_f1 = self.calc_f1(labels_hat, labels)
        val_logits = [torch.sigmoid(logit)[1].item() for logit in logits]

        self.log("val_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        self.log("val_f1", val_f1, on_epoch=True, on_step=True, prog_bar=True)
        return {"val_f1": val_f1, "val_logits": val_logits, "val_loss": loss}


    def calc_f1(self, preds, labels):
        TP = 0
        total_true = torch.sum(labels).item()
        total_preds = torch.sum(preds).item()
        for pred,gold in zip(preds, labels):
            if pred == 1 and gold == 1:
                TP += 1

        prec = TP/total_preds if total_preds != 0 else 0
        recall = TP/total_true if total_true != 0 else 0
        if (prec + recall) == 0: f1 = 0
        else: f1 = 2 * (prec * recall) / (prec + recall)
        return f1

    def validation_epoch_end(self, validation_step_outputs):
        if self.trainer.running_sanity_check:
            return

        result = self.decode_logits(validation_step_outputs, self.val_inputs)
        self.text_logger.log_object({"val_f1": result['val_f1']}, self.trainer.global_step)
        self.text_logger.log_object({"val_prec": result['val_prec']}, self.trainer.global_step)
        self.text_logger.log_object({"val_recall": result['val_recall']}, self.trainer.global_step)

    def test_step(self, inputs, batch_idx):
        return self.validation_step(inputs, batch_idx)

    def test_epoch_end(self, test_step_outputs):
        result = self.decode_logits(test_step_outputs, self.val_inputs)

        self.text_logger.log_object({"After training validation": result['val_f1']},self.trainer.global_step)
        self.text_logger.log_object({"best_model_path":self.trainer.checkpoint_callback.best_model_path},self.trainer.global_step)

    def decode_logits(self, outputs, eval_set):
        all_outputs = []
        all_val_loss = []
        for batch in outputs:
            all_outputs.extend(batch['val_logits'])
            all_val_loss.append(batch['val_loss'])

        if len(all_outputs) != len(eval_set):#sanity check
            print("WARNING - len of all outputs != eval set")
            return {"val_f1": None}

        all_labels, preds2maximalprob = self.get_max_bipartite_matching(all_outputs, eval_set)
        f1, prec, recall, total_preds, total_true = self.calc_bipartite_f1(all_labels, preds2maximalprob)

        val_loss = sum(all_val_loss) / len(outputs)
        self.log("val_f1", f1, prog_bar=True, on_epoch=True)
        self.log("val_prec", prec, prog_bar=True)
        self.log("val_recall", recall, prog_bar=True)
        self.log("val_T_predicted", total_preds, prog_bar=True)
        self.log("val_true", total_true, prog_bar=True)
        return {"val_f1": f1,"val_prec": prec,"val_recall": recall, "val_loss":val_loss}

    def get_max_bipartite_matching(self, all_outputs, eval_set, predict=False,fusion=False):
        ids2outputs = defaultdict(list)
        all_labels = defaultdict(int)
        if predict:
            for i,pair in enumerate(zip(all_outputs, eval_set)):
                prob = pair[0]
                val_inst_key = pair[1]['key']
                pair_qa = (pair[1]['qa_uuid_1'], pair[1]['qa_uuid_2'])
                if fusion:
                    all_labels[pair_qa[0] + "|" + pair_qa[1]] = [val_inst_key, pair[1]['input_1'], pair[1]['input_2'],pair[1]['abs_sent_id_1'], pair[1]['abs_sent_id_2'], pair[1]['qa_uuid_1'], pair[1]['qa_uuid_2'], pair[1]['label']]
                else: all_labels[pair_qa[0] + "|" + pair_qa[1]] = [val_inst_key, pair[1]['input_1'], pair[1]['input_2'],
                                                             pair[1]['qa_uuid_1'], pair[1]['qa_uuid_2'],pair[1]['label']]
                ids2outputs[val_inst_key].append((pair_qa, prob))#key,and prob
        else:
            for i,pair in enumerate(zip(all_outputs, eval_set)):
                prob = pair[0]
                val_inst_key = pair[1]['key']
                pair_qa = (pair[1]['qa_uuid_1'], pair[1]['qa_uuid_2'])
                all_labels[pair_qa[0] + "|" + pair_qa[1]] = [pair[1]['label']]
                ids2outputs[val_inst_key].append((pair_qa, prob))#key,and prob

        preds2maximalprob = {}
        for key, qas in ids2outputs.items():#qas == (pair_qa, pair[0])
            bipartite_result = max_bipartite_match(qas)
            for pair, prob in bipartite_result.items():
                preds2maximalprob[pair] = prob
        return all_labels, preds2maximalprob

    def calc_bipartite_f1(self, all_labels, preds2maximalprob):

        TP = 0
        total_true = 0
        total_preds = 0
        for pair,inst in all_labels.items():
            label = inst[-1]
            total_true += label
            if pair in preds2maximalprob: #passed maximal bipartite matching
                if preds2maximalprob[pair] > self.threshold: #predicted to align
                    if label == 1:
                        TP += 1
                    total_preds += 1

        prec = TP / total_preds if total_preds != 0 else 0
        recall = TP / total_true if total_true != 0 else 0
        if (prec + recall) == 0:
            f1 = 0
        else:
            f1 = 2 * (prec * recall) / (prec + recall)

        return f1, prec, recall, total_preds, total_true

    def get_lr_scheduler(self, optimizer, n_total_steps: int):
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=n_total_steps)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

    def configure_optimizers(self):
        optimizer_grouped_parameters = [{
            'params': [p for n, p in self.model.named_parameters()
                       if not is_in_no_decay(n)],
            'weight_decay': self.weight_decay
        }, {
            'params': [p for n, p in self.model.named_parameters()
                       if is_in_no_decay(n)],
            'weight_decay': 0.0,
        }]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.learning_rate,
                          eps=self.adam_eps)

        if self.n_grad_update_steps == 1:
            return optimizer

        scheduler = self.get_lr_scheduler(optimizer, self.n_grad_update_steps)
        return [optimizer], [scheduler]

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # nothing to save here
        if self.trainer.running_sanity_check:
            return

        print("\tSaving model on step: ", checkpoint['global_step'])
        # We save a checkpoint of our own, because getting the model directly
        # from PL checkpoint and loading it into a bare bones HuggingFace model
        # is not trivial, especially when we don't want to re-use the PL wrapper code.
        name = f"saved_model_step_{checkpoint['global_step']}"
        save_dir = os.path.join(self.trainer.logger.log_dir, name)
        self.model.save_pretrained(save_dir)
        self.last_saved_model_path = save_dir

    def infer(self, batch_inputs: BatchEncoding, device) -> List[int]:

        req_inputs = self.to_required_inputs(batch_inputs)
        req_inputs = req_inputs.to(device)
        result = self.model(
            req_inputs["input_ids"],
            token_type_ids=req_inputs["token_type_ids"],
            attention_mask=req_inputs["attention_mask"]
        )
        logits = result['logits']
        val_logits = [torch.sigmoid(logit)[1].item() for logit in logits]
        return val_logits

    def infer_roberta(self, batch_inputs: BatchEncoding, device) -> List[int]:

        req_inputs = self.to_required_inputs(batch_inputs)
        req_inputs = req_inputs.to(device)
        result = self.model(
            req_inputs["input_ids"],
            attention_mask=req_inputs["attention_mask"]
        )
        logits = result['logits']
        val_logits = [torch.sigmoid(logit)[1].item() for logit in logits]
        return val_logits


    @classmethod
    def to_required_inputs(cls, batch: BatchEncoding):
        """
        Returns inputs with only the required fields that for the forward function of a module.
        The inputs may contain extra fields for decoding (subword_to_token) and will produce
        an error if passed to the model as: model.forward(**inputs)
        """
        raw_inputs = {key: batch[key] for key in cls.MODEL_FIELDS & batch.keys()}
        new_batch = BatchEncoding(data=raw_inputs)
        return new_batch

class SimpleTextLogger:

    def __init__(self, log_dir: str):
        print("--  Logger initialized --- ")
        self.file_path = os.path.join(log_dir, 'exp.log')
        self.log_name = os.path.basename(log_dir)
        file_handler = logging.FileHandler(self.file_path)
        file_handler.setLevel("INFO")
        fmt = logging.Formatter("%(message)s")
        file_handler.setFormatter(fmt)
        self.logger = logging.getLogger(self.log_name)
        self.logger.setLevel("INFO")
        self.logger.addHandler(file_handler)
        self.logger.propagate = False

    @rank_zero_only
    def log_object(self, data: Dict[str, Any], step):
        data['step'] = step
        self.logger.info(data)

    @classmethod
    def from_logger(cls, logger: LightningLoggerBase):
        return cls(logger.log_dir)
