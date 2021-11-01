# QA-ALIGN: Representing Cross-Text Content Overlap by Aligning Question-Answer Propositions
To Appear in EMNLP 2021

* Paper can be found [here](https://arxiv.org/abs/2109.12655)
* Official QA-Alignments can be found [here](data/official_qa_alignments)
* Crowdsourcing related materials can be found [here](crowdsourcing)
* Files relatedd to training and experimenting with the QA-Align model can be found [here](models/)
* Fusion experiemnt related data can be found [here](data/fusion_data_and_experiment)
* Processed qa-alignments for training can be found [here](data/) in a zip called: processed_alignments_for_training


## Paper Abstract
Multi-text applications, such as multi-document summarization, are typically required to model redundancies across related texts. Current methods confronting consolidation struggle to fuse overlapping information. In order to explicitly represent content overlap, we propose to align predicate-argument relations across texts, providing a potential scaffold for information consolidation. We go beyond clustering coreferring mentions, and instead model overlap with respect to redundancy at a propositional level, rather than merely detecting shared referents. Our setting exploits QA-SRL, utilizing question-answer pairs to capture predicate-argument relations, facilitating laymen annotation of cross-text alignments. We employ crowd-workers for constructing a dataset of QA-based alignments, and present a baseline QA alignment model trained over our dataset. Analyses show that our new task is semantically challenging, capturing content overlap beyond lexical similarity and complements cross-document coreference with proposition-level links, offering potential use for downstream tasks.
