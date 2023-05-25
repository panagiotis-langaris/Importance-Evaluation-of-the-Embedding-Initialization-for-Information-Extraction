# Importance Evaluation of the Embedding’s Initialization for Information Extraction

# Context
Information Extraction (IE) is an important task of Natural Language Processing. End-to-end
IE requires solving both the Named Entity Recognition (NER) and the Relation Extraction
(RE) tasks. Strict evaluation ([2]) of the RE task presupposes correct detection of the
boundaries and the entity type of each argument in the relation ([1]). Hence, the IE problem
can be solved as a joint task, solving the NER and RE tasks joint, or can be treated in a
pipeline setting, where first the named entities are detected and then the relations between
them are classified. The raw input of the model is unstructured text. In input initialization,
each word should be mapped to a vector, named embeddings. So, the first layer of the IE
models is the embedding’s layer. This initialization step can be crucial for the overall
performance of the task. Currently, a widely used strategy is to initialize the word
embeddings, using pre-trained language models, like BERT ([3]) and ELMO ([4]). However,
more advanced initialization strategies, based on representation learning, have been proposed
([1]). The IE task is solved in different domains, like biomedical and newsfeed. In this thesis,
we will work with biomedical text ([5]).

# Goal
Develop a neural network architecture for RE and NER. Experiment with different
embeddings initialization strategies and compare the performance of the different
variations. Gain insights into the properties that word embeddings should have when used
as input in information extraction models.

# Architecture of the proposed IE model
![ie_model (1)](https://github.com/panagiotis1994/thesis/assets/16323614/0502b297-06d5-440c-9fdb-2f880c6c437f)

# Results
![total_performance](https://github.com/panagiotis1994/thesis/assets/16323614/adfde5f5-eeba-4b4b-abcd-40513f161424)


# References
1. Theodoropoulos, C., Henderson, J., Coman, A. C., & Moens, M. F. (2021). Imposing
Relation Structure in Language-Model Embeddings Using Contrastive Learning. arXiv
preprint arXiv:2109.00840.
2. Taillé, B., Guigue, V., Scoutheeten, G., & Gallinari, P. (2020, November). Let’s Stop Error
Propagation in the End-to-End Relation Extraction Literature! In Proceedings of the 2020
Conference on Empirical Methods in Natural Language Processing (EMNLP) (pp. 3689-
3701).
3. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep
bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.
4. Peters, M. E., Neumann, M., Iyyer, M., Gardner, M., Clark, C., Lee, K., & Zettlemoyer, L.
(2018). Deep contextualized word representations. arXiv preprint arXiv:1802.05365.
5. Gurulingappa, H., Rajput, A. M., Roberts, A., Fluck, J., Hofmann-Apitius, M., & Toldo, L.
(2012). Development of a benchmark corpus to support the automatic extraction of drugrelated
adverse effects from medical case reports. Journal of biomedical informatics, 45(5),
885-892.
