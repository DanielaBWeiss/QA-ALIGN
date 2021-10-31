import types
from argparse import ArgumentParser
import os

import pandas as pd
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import warnings

from qa_alignment_dataset import QAAlignmentDataset
from qa_alignment_model import QAAlignmentModule

warnings.filterwarnings('ignore',
                        "The dataloader, train dataloader, does not have many workers which may be a bottleneck")
warnings.filterwarnings('ignore',
                        "The dataloader, val dataloader 0, does not have many workers which may be a bottleneck")

def total_train_steps(args, max_epochs, train_set: Dataset) -> int:
    """The number of total training steps that will be run. Used for lr scheduler purposes."""
    is_default = isinstance(args.gpus, types.FunctionType)
    gpus = 0 if is_default else args.gpus
    if isinstance(gpus, str) and ',' in gpus:
        gpus = len(gpus.split(","))

    num_devices = max(1, 0) #unfortunately because we need to do max bipartite matching, we can't use gpus
    effective_batch_size = args.batch_size * args.accumulate_grad_batches * num_devices
    return (len(train_set) / effective_batch_size) * max_epochs

def load_files(args):
    print("Loading files")
    if args.file_ext == "":
        file_ext = ""
    else: file_ext = "_"+args.file_ext
    train = pd.read_csv(args.input_dir+"/train_processed"+file_ext+".csv")
    print("#train samples: ",len(train))
    if args.n_train:
        print("Taking ",args.n_train, " samples from train")
        train = train[:args.n_train].to_dict(orient='records')
    else: train = train.to_dict(orient='records')

    val = pd.read_csv(args.input_dir+"/val_processed"+file_ext+".csv")
    print("#val samples: ", len(val))
    if args.n_val:
        print("Taking ", args.n_val, " samples from val")
        val = val[:args.n_val].to_dict(orient='records')
    else: val = val.to_dict(orient='records')

    if args.test:
        test = pd.read_csv(args.input_dir + "/test_processed"+file_ext+".csv")
        print("#test samples: ", len(test))
        test = test.to_dict(orient='records')
        return {"train":train, "val":val, "test":test}
    return {"train":train, "val":val}

def get_data_loaders(train, val, args, test=None): #TODO: change collate func if training bert again
    if "roberta" in args.exp_name:
        train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=train.collate_samples_roberta)
        val_loader = DataLoader(val, batch_size=args.batch_size*2, shuffle=False, collate_fn=val.collate_samples_roberta)
        if test:
            test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, collate_fn=test.collate_samples_roberta)
            return {"train":train_loader,"val":val_loader, "test":test_loader}
    elif "cdlm" in args.exp_name:
        train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=train.collate_samples_cdlm)
        val_loader = DataLoader(val, batch_size=args.batch_size * 2, shuffle=False,
                                collate_fn=val.collate_samples_cdlm)
        if test:
            test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False,
                                     collate_fn=test.collate_samples_cdlm)
            return {"train": train_loader, "val": val_loader, "test": test_loader}
    else:
        train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=train.collate_samples)
        val_loader = DataLoader(val, batch_size=args.batch_size * 2, shuffle=False,
                                collate_fn=val.collate_samples)
        if test:
            test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False,
                                     collate_fn=test.collate_samples)
            return {"train": train_loader, "val": val_loader, "test": test_loader}
    return {"train":train_loader,"val":val_loader}

def main(args):
    data = load_files(args)
    print("Exp name: ",args.exp_name)
    print("Num train: ", len(data['train']))
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    if args.special_token:
        print("Adding special token")
        tokenizer.add_special_tokens({"additional_special_tokens": ['[A]', '[/A]', '[P]', '[/P]', '[Q]']})
        model.resize_token_embeddings(len(tokenizer))
    train_set = QAAlignmentDataset(data['train'], tokenizer)
    val_set = QAAlignmentDataset(data['val'], tokenizer)
    print()
    if args.test:
        test_set = QAAlignmentDataset(data['test'], tokenizer)
    else:
        test_set = None

    max_epochs = args.max_epochs if args.max_epochs < 1000 else 5
    print(f"Max Epochs: {max_epochs}")
    data_loaders = get_data_loaders(train_set, val_set, args, test_set)
    print("Example training sample: ", data_loaders['train'].dataset.samples[0])
    n_grad_update_steps = total_train_steps(args, max_epochs, train_set)
    logger = pl_loggers.TensorBoardLogger(save_dir=args.save_dir, name=args.exp_name)

    #
    pl_model = QAAlignmentModule(model,
                                  n_grad_update_steps,
                                  args.weight_decay,
                                  args.learning_rate,
                                  args.adam_eps,0.5)
    pl_model.val_inputs = val_set
    pl_model.test_inputs = test_set
    model_cp = pl.callbacks.ModelCheckpoint(monitor='val_f1',
                                            #dirpath=args.save_dir,
                                            #filename='alig-cls-{epoch:02d}-{val_f1:.2f}',
                                            #save_top_k=3,
                                            mode='max')

    early_stop = pl.callbacks.EarlyStopping(
        monitor='val_f1',
        min_delta=0.00,
        patience=5,
        verbose=True,
        mode='max'
    )

    trainer = Trainer(gpus=str(args.gpus),
                         checkpoint_callback=model_cp,
                         callbacks=[early_stop],
                         distributed_backend=args.distributed_backend,
                         logger=logger,
                         limit_train_batches=args.limit_train_batches,
                         limit_val_batches=args.limit_val_batches,
                         val_check_interval=args.val_check_interval,
                         gradient_clip_val=1.0,
                         accumulate_grad_batches=args.accumulate_grad_batches,
                         max_epochs=max_epochs,
                         fast_dev_run=args.fast_dev_run,
                         weights_save_path=args.save_dir)

    trainer.logger.log_hyperparams(args)
    trainer.fit(pl_model, data_loaders['train'], data_loaders['val'])

    #tok_path = os.path.dirname(pl_model.last_saved_model_path)
    tokenizer.save_pretrained(pl_model.last_saved_model_path)

    print("final validation")
    print("last saved model path: ", pl_model.last_saved_model_path)

    trainer.test(test_dataloaders=data_loaders['val'], ckpt_path='best', verbose=True)  # later test

    pl_model.text_logger.log_object({"best_model_path": model_cp.best_model_path},trainer.global_step)
    pl_model.text_logger.log_object({"best_model_score": model_cp.best_model_score.item()},trainer.global_step)
    pl_model.text_logger.log_object({"last_saved_model_path": pl_model.last_saved_model_path},trainer.global_step)



if __name__ == "__main__":
    parser = ArgumentParser()
    # ap.add_argument("--train", required=True, action="append")
    # ap.add_argument("--dev", required=True, action="append")
    parser.add_argument('--input_dir', type=str, required=True, help='directory containing train,dev, and test files')
    #parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--test", default=None, action="store_true")
    parser.add_argument("--predict", default=None, action="store_true")
    parser.add_argument("--special_token", default=None, action="store_true")
    parser.add_argument("--save_dir", default="./qa-align-evaluations")
    parser.add_argument("--n_train", default=None, type=int)
    parser.add_argument("--n_val", default=None, type=int)
    parser.add_argument("--exp_name", default="wneg_sample")
    parser.add_argument("--file_ext", default="")
    parser.add_argument("--model", default="bert-base-uncased", type=str)
    parser.add_argument("--roberta", default="roberta", type=str, required=False)
    parser.add_argument("--batch_size", default=16, type=int)
    #parser.add_argument("--model_name", required=True)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_eps", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_length", default=256, type=int)

    ap = pl.Trainer.add_argparse_args(parser)
    main(ap.parse_args())