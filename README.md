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



1. **A tutorial on hidden Markov models and selected applications in speech recognition**. Lawrence Rabiner. Proceedings of the IEEE, 1989. [PDF](https://web.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf) (_HMM_)
2. **Speech Recognition with Weighted Finite-State Transducers**. Mehryar Mohri, Fernando Pereira, Michael Riley. Handbook on Speech Processing and Speech, Springer, 2008. [PDF](https://cs.nyu.edu/~mohri/pub/hbka.pdf) (_WFST_)
3. **State-of-the-Art of End-to-End Speech Recognition**. Zhijian Ou. Tutorial at The 6th Asian Conference on Pattern Recognition (ACPR 2021), Jeju Island, Korea, 2021.  [PDF](http://oa.ee.tsinghua.edu.cn/~ouzhijian/pdf/ACPR2021%20Tutorial%20State-of-the-Art%20of%20End-to-End%20Speech%20Recognition.pdf)
4. **Context-dependent pre-trained deep neural networks for large-vocabulary speech recognition**. George E. Dahl, Dong Yu, Li Deng, Alex Acero. IEEE Transactions on Audio, Speech, and Language Processing, 2011. [PDF](https://www.cs.toronto.edu/%7Egdahl/papers/DRAFT_DBN4LVCSR-TransASLP.pdf) (_DNN-HMM_)
5. **Connectionist temporal classification: Labelling unsegmented sequence data with recurrent neural networks**. Alex Graves, Santiago Fernandez, Faustino Gomez, and Juergen Schmidhuber. ICML, 2006. [PDF](https://www.cs.toronto.edu/~graves/icml_2006.pdf) (_CTC_)
6. **Purely sequence-trained neural networks for ASR based on lattice-free MMI**. Daniel Povey, et al. INTERSPEECH, 2016. [PDF](https://www.danielpovey.com/files/2016_interspeech_mmi.pdf) (_Kaldi_)
7. **Hybrid CTC/attention architecture for end-to-end speech recognition**. Shinji Watanabe, Takaaki Hori, Suyoun Kim, John Hershey, Tomoki Hayashi. IEEE Journal of Selected Topics in Signal Processing, 2017. [PDF](https://www.merl.com/publications/docs/TR2017-190.pdf) (_ESPnet_)
8. **CRF-based Single-stage Acoustic Modeling with CTC Topology**. Hongyu Xiang, Zhijian Ou. ICASSP, 2019. [PDF](http://oa.ee.tsinghua.edu.cn/~ouzhijian/pdf/ctc-crf.pdf) (_CTC-CRF_)
9. **Neural machine translation by jointly learning to align and translate**. Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio. ICLR, 2015. [PDF](https://arxiv.org/pdf/1409.0473.pdf) (_AED, Attention-based Encoder-Decoder_)
10. **Listen, attend and spell: A neural network for large vocabulary conversational speech recognition**. William Chan, Navdeep Jaitly, Quoc V. Le, Oriol Vinyals. ICASSP, 2016. [PDF](https://arxiv.org/pdf/1409.0473.pdf) (_LAS_)
11. **Sequence transduction with recurrent neural networks**. Alex Graves. ICML 2012 Workshop on Representation Learning, 2012 [PDF](https://arxiv.org/pdf/1211.3711.pdf) (_RNN-T_)
12. **Conditional random fields: Probabilistic models for segmenting and labeling sequence data**. Lafferty John, Andrew McCallum, Fernando Pereira. ICML, 2001. [PDF](https://repository.upenn.edu/server/api/core/bitstreams/4905e2c0-e9d5-4961-804b-973de8bdfc7c/content) (_CRF_)
13. **Conformer: Convolution-augmented Transformer for Speech Recognition**. Anmol Gulati, et al. Interspeech, 2020. [PDF](https://arxiv.org/pdf/2005.08100.pdf) (_Conformer_)
14. **Multilingual and Crosslingual Speech Recognition using Phonological-Vector based Phone Embeddings**. Chengrui Zhu, Keyu An, Huahuan Zheng, Zhijian Ou. ASRU, 2021. [PDF](https://arxiv.org/pdf/2107.05038.pdf) (_JoinAP_)

## Language Modeling
- [HLT-LANG] Language Modeling
- [HLT-LACL] Language Acquisition and Learning
- [HLT-UNDE] Spoken Language Understanding and Computational Semantics
- [HLT-MLMD] Machine Learning Methods for Language

1. **An empirical study of smoothing techniques for language modeling**. Stanley F. Chen, Joshua Goodman. Computer Speech & Language, 1999. [PDF](http://oa.ee.tsinghua.edu.cn/~ouzhijian/pgm/pgm-pdf/Chen&Goodman1998_An_empirical_study_of_smoothing_techniques_for_language_modeling.pdf) (_N-Gram_)

2. **Energy-Based Models with Applications to Speech and Language Processing**. Zhijian Ou. ICASSP 2022 Tutorial. [PDF](http://oa.ee.tsinghua.edu.cn/~ouzhijian/ICASSP2022/ICASSP2022_Tutorial_EBM.pdf)

3. **Learning Trans-dimensional Random Fields with Applications to Language Modeling**. Bin Wang, Zhijian Ou, Zhiqiang Tan. TPAMI, 2018. [PDF](http://oa.ee.tsinghua.edu.cn/ouzhijian/pdf/TRF.pdf) (_TRF_)

4. **Improving language understanding by generative pre-training**. Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever. 2018.  [PDF](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) (_GPT_)

5. **BERT: Pre-training of deep bidirectional transformers for language understanding**. Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. ACL 2019. [PDF](https://arxiv.org/pdf/1810.04805.pdf)  (_BERT_)

6. **Language models are unsupervised multitask learners**. Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever. OpenAI Blog, 2019. [PDF](https://insightcivic.s3.us-east-1.amazonaws.com/language-models.pdf) (_GPT-2_)

7. **Language Models are Few-Shot Learners**. Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei. NeurIPS 2020. [PDF](https://proceedings.neurips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf) (_GPT-3_)



## Dialog Systems
- [HLT-DIAL] Discourse and Dialog

1.  **Task-Oriented Dialog Systems that Consider Multiple Appropriate Responses 
under the Same Context**. Yichi Zhang, Zhijian Ou, Zhou Yu. AAAI, 2020. [PDF](https://arxiv.org/pdf/1911.10484.pdf) (_DAMD_)
2. **A Simple Language Model for Task-Oriented Dialogue**. Ehsan Hosseini-Asl, Bryan McCann, Chien-Sheng Wu, Semih Yavuz, Richard Socher. NeurIPS, 2020. [PDF](https://arxiv.org/pdf/2005.00796.pdf) (_SimpleTOD_)
3. **Retrieval-Augmented Generation for Knowledge-Intensive NLP tasks**. Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela. NeurIPS, 2020. [PDF](https://arxiv.org/pdf/2005.11401.pdf) (_RAG_)
4. **Training language models to follow instructions with human feedback**. Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe. NeurIPS, 2022. [PDF](https://arxiv.org/pdf/2203.02155.pdf) (_InstructGPT_)

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

1. **Dense Passage Retrieval for Open-Domain Question Answering**. Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih. EMNLP, 2020. [PDF](https://arxiv.org/pdf/2004.04906.pdf) (_DPR_)
2. **Text and Code Embeddings by Contrastive Pre-Training**. Arvind Neelakantan, Tao Xu, Raul Puri, Alec Radford, Jesse Michael Han, Jerry Tworek, Qiming Yuan, Nikolas Tezak, Jong Wook Kim, Chris Hallacy, Johannes Heidecke, Pranav Shyam, Boris Power, Tyna Eloundou Nekoul, Girish Sastry, Gretchen Krueger, David Schnurr, Felipe Petroski Such, Kenny Hsu, Madeleine Thompson, Tabarak Khan, Toki Sherbakov, Joanne Jang, Peter Welinder, Lilian Weng. arXiv:2201.10005, 2022. [PDF](https://arxiv.org/pdf/2201.10005.pdf) (_openai cpt-text, cpt-code_)

## Machine Translation
- [HLT-MTSW] Machine Translation for Spoken and Written Language

## Language Resources and Systems
- [HLT-LRES] Language Resources and Systems

## Multimodal
- [HLT-MMPL] Multimodal Processing of Language
