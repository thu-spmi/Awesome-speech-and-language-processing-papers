# Awesome speech and language processing papers

Speech and language processing (SLP) is a large field, covering a wide range of research topics. A classic book, "Speech and language processing - An Introduction to Natural Language Processing, Computational Linguistics, and Speech Recognition" is from [Dan Jurafsky](http://web.stanford.edu/people/jurafsky/) and [James H. Martin](http://www.cs.colorado.edu/~martin/), which you can find [here](https://web.stanford.edu/~jurafsky/slp3/). 

The SLP field is moving fast. Here we are maintaining awesome speech and language processing papers. We hope the list of papers could serve for readers who are interested in entering the field and doing further research. We also include some tutorials, which are online freely accessible.

IEEE Signal Processing Society (SPS) uses Editor’s Information Classification Schemes (EDICS's) for the SPS transactions and conferences to classify and assign papers for reviewing and publishing. We will follow [the EDICS from SPS SLTC](https://signalprocessingsociety.org/community-involvement/speech-and-language-processing/edics) (Speech and Language Processing Technical Committee), but with some modifications.

_Disclaimer: We may miss some relevant papers in the list. If you have any suggestions or would like to add some papers, please submit a pull request or email us. Your contribution is much appreciated!_

1. [Automatic Speech Recognition (ASR)](#automatic-speech-recognition-asr)
2. [Language Modeling](#language-modeling)
3. [Dialog Systems](#dialog-systems)
4. Much more to be added


## Automatic Speech Recognition (ASR)

It includes a mix of EDICS topics:
- [SPE-RECO] Acoustic Modeling for Automatic Speech Recognition
- [SPE-LVCR] Large Vocabulary Continuous Recognition/Search (End-to-end approaches, Other LVSCR approaches)
- [SPE-ROBU] Robust Speech Recognition
- [SPE-ADAP] Speech Adaptation/Normalization
- [SPE-MULT] Multilingual Recognition and Identification
- [SPE-GASR] General Topics in Speech Recognition



1. L. R. Rabiner, “A tutorial on hidden Markov models and selected applications in speech recognition”, Proceedings of the IEEE, 1989. [PDF](https://web.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf) (_HMM_)
2. M. Mohri, F. Pereira, and M. Riley, “Speech Recognition with Weighted Finite-State Transducers”, Handbook on Speech Processing and Speech, Springer, 2008. [PDF](https://cs.nyu.edu/~mohri/pub/hbka.pdf) (_WFST_)
3. Zhijian Ou, "State-of-the-Art of End-to-End Speech Recognition", Tutorial at The 6th Asian Conference on Pattern Recognition (ACPR 2021), Jeju Island, Korea, 2021.  [PDF](http://oa.ee.tsinghua.edu.cn/~ouzhijian/pdf/ACPR2021%20Tutorial%20State-of-the-Art%20of%20End-to-End%20Speech%20Recognition.pdf)
4. G. Dahl, et al., "Context-dependent pre-trained deep neural networks for large-vocabulary speech recognition", TASLP, 2012. (_DNN-HMM_)
5. A. Graves, S. Fernandez, F. Gomez, and J. Schmidhuber, “Connectionist temporal classiﬁcation: Labelling unsegmented sequence data with recurrent neural networks”, ICML, 2006. [PDF](https://www.cs.toronto.edu/~graves/icml_2006.pdf) (_CTC_)
6. D. Povey, et al., "Purely sequence-trained neural networks for ASR based on lattice-free MMI", INTERSPEECH 2016. (_Kaldi_)
7. S. Watanabe, T. Hori, S. Kim, J. R. Hershey, and T. Hayashi, "Hybrid CTC/attention architecture for end-to-end speech recognition", IEEE Journal of Selected Topics in Signal Processing, 2017. (_ESPnet_)
8. Hongyu Xiang, Zhijian Ou, "CRF-based Single-stage Acoustic Modeling with CTC Topology", ICASSP, 2019. [PDF](http://oa.ee.tsinghua.edu.cn/~ouzhijian/pdf/ctc-crf.pdf) (_CTC-CRF_)
9. D. Bahdanau, et al, “Neural machine translation by jointly learning to align and translate”, ICLR 2015. (_AED, Attention-based Encoder-Decoder_)
10. W. Chan, et al., "Listen, attend and spell: A neural network for large vocabulary conversational speech recognition", ICASSP 2016. (_LAS_)
11. A. Graves, “Sequence transduction with recurrent neural networks,” ICML 2012 Workshop on Representation Learning. (_RNN-T_)
12. Lafferty, et al., "Conditional random fields: Probabilistic models for segmenting and labeling sequence data”, ICML 2001. (_CRF_)
13. “Conformer: Convolution-augmented Transformer for Speech Recognition”, Interspeech 2020. (_Conformer_)
14. Chengrui Zhu, Keyu An, Huahuan Zheng, Zhijian Ou. “Multilingual and Crosslingual Speech Recognition using Phonological-Vector based Phone Embeddings”, ASRU 2021. (_JoinAP_)

## Language Modeling
- [HLT-LANG] Language Modeling
- [HLT-LACL] Language Acquisition and Learning
- [HLT-UNDE] Spoken Language Understanding and Computational Semantics
- [HLT-MLMD] Machine Learning Methods for Language

1. S. F. Chen and J. Goodman, “An empirical study of smoothing techniques for language modeling”, Computer Speech & Language, 1999. (_N-Gram_)

2. Zhijian Ou, "Energy-Based Models with Applications to Speech and Language Processing", ICASSP 2022 Tutorial. [PDF](http://oa.ee.tsinghua.edu.cn/~ouzhijian/ICASSP2022/ICASSP2022_Tutorial_EBM.pdf)

3. Bin Wang, Zhijian Ou, Zhiqiang Tan, "Learning Trans-dimensional Random Fields with Applications to Language Modeling", TPAMI, 2018. (_TRF_)

4. Radford, et al, "Improving language understanding by generative pre-training", 2018. (_GPT_)

5. Devlin, et al, "BERT: Pre-training of deep bidirectional transformers for language understanding", ACL 2019. (_BERT_)

6. Radford, et al, "Language models are unsupervised multitask learners", OpenAI Blog, 2019. (_GPT_-2)

7. Brown, et al, "Language Models are Few-Shot Learners", NeurIPS 2020. (_GPT_-3)

   


## Dialog Systems
[HLT-DIAL] Discourse and Dialog

1. Yichi Zhang, Zhijian Ou, and Zhou Yu, “Task-oriented dialog systems that consider multiple appropriate responses under the same context”, AAAI, 2020. (_DAMD_)
2. Ehsan Hosseini-Asl, Bryan McCann, Chien-Sheng Wu, Semih Yavuz, and Richard Socher, “A simple language model for task-oriented dialogue,” arXiv:2005.00796, 2020. (_SimpleTOD_)
3. Lewis, et al, "Retrieval-Augmented Generation for Knowledge-Intensive NLP tasks", NeurIPS 2020. (_RAG_)
4. Ouyang, et al, "Training language models to follow instructions with human feedback", arXiv:2203.02155, 2022. (_InstructGPT_)

## Speech Perception and Psychoacoustics
- [SPE-SPER] Speech Perception and Psychoacoustics

## Speech Analysis
- [SPE-ANLS] Speech Analysis

## Speech Coding
- [SPE-CODI] Speech Coding

## Speech Generation
- [SPE-SPRD] Speech Production
- [SPE-SYNT] Speech Synthesis and Generation

## Speech Enhancement and Separation
- [SPE-ENHA] Speech Enhancement and Separation

## Speaker Recognition
- [SPE-SPKR] Speaker Recognition and Characterization

## Parsing
- [HLT-STPA] Segmentation, Tagging, and Parsing

## Retrieval
- [HLT-SDTM] Spoken Document Retrieval and Text Mining

## Machine Translation
- [HLT-MTSW] Machine Translation for Spoken and Written Language

## Language Resources and Systems
- [HLT-LRES] Language Resources and Systems

## Multimodal
- [HLT-MMPL] Multimodal Processing of Language