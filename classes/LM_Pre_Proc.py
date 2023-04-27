class LM_Pre_Proc:
    def __init__(self, processed_data_path, lang_model):
        self.processed_data_path = processed_data_path
        self.lang_model = lang_model
        self.out_folder_path = lang_model + '/'

        # Find the path files for training
        self.processed_files_list = []
        for file_index in range(hyperparameters['dataset_length']):
            self.processed_files_list.append(str(file_index) + '.json')

        self.original_dataset = ADE_Dataset(filenames = self.processed_files_list,
                                            processed_data_path = processed_data_path,
                                            folder_path = 'original/')

    def get_bert_embeddings(self, tokens, tokenizer, lang_model):

        # Convert words to input tokens
        word_pieces = ['[CLS]']
        word_pieces_lengths = []
        for word in tokens:
            word_pieces_of_token = tokenizer.tokenize(word)
            word_pieces.extend(word_pieces_of_token)
            word_pieces_lengths.append(len(word_pieces_of_token))

        word_pieces.extend(['[SEP]'])

        # Convert tokens to input IDs and attention masks
        input_ids = tokenizer.convert_tokens_to_ids(word_pieces)
        attention_mask = [1] * len(input_ids)

        # Convert input IDs and attention masks to tensors
        input_ids = torch.tensor([input_ids])
        attention_mask = torch.tensor([attention_mask])

        # Get BERT embeddings for input IDs and attention masks
        with torch.no_grad():
            outputs = lang_model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_states = outputs.last_hidden_state

        # Compute average embeddings for each word
        # add reference to paper that says this is better!!!!!!!!!!!!!!!
        bert_avg_embeddings = torch.tensor([])
        i = 1
        j = 0
        for word in tokens:
            subword_embeddings = last_hidden_states[0][i:i+word_pieces_lengths[j]]
            avg_embedding = torch.mean(subword_embeddings, dim=0)
            avg_embedding = avg_embedding.unsqueeze_(0)
            bert_avg_embeddings = torch.cat((bert_avg_embeddings, avg_embedding), dim=0)
            i += word_pieces_lengths[j]
            j += 1

        return bert_avg_embeddings.tolist()
    
    def get_charbert_embeddings(self, tokens, tokenizer, char_bert_model):
        
        # Add [CLS] and [SEP]
        word_pieces = ['[CLS]', *tokens, '[SEP]']
        
        # Convert token sequence into character indices
        indexer = CharacterIndexer()
        batch = [word_pieces]  # This is a batch with a single token sequence x
        batch_ids = indexer.as_padded_tensor(batch)

        # Feed batch to CharacterBERT & get the embeddings
        embeddings_for_batch, _ = char_bert_model(batch_ids)
        tmp_embeddings = embeddings_for_batch[0]
        charbert_embeddings = tmp_embeddings[1:len(tmp_embeddings)-1]
        
        return charbert_embeddings.tolist()
    
    def get_glove_embeddings(self, tokens):
        # Load the GloVe pre-trained embeddings
        # Trained on Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors, 822 MB download): glove.6B.zip
        # Embedding vector's size is 100
        global_vectors = GloVe(name='6B', dim=100)

        # Take the pre-trained embeddings for the tokens
        glove_embeddings = global_vectors.get_vecs_by_tokens(tokens, lower_case_backup=True)

        return glove_embeddings.tolist()

    def get_elmo_embeddings(self, tokens, options_file, weights_file):
        # Create an instance of the default ELMo module
        elmo = Elmo(options_file = options_file,
                    weight_file = weights_file,
                    num_output_representations = 1,
                    dropout = 0)

        # Take the character ids for each word
        character_ids = batch_to_ids([tokens]) # A full sentence for ELMo is expected in this format: [["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog."]]

        embeddings = elmo(character_ids)

        return embeddings['elmo_representations'][0][0].tolist()

    def save_json(self, tokens, ne_tags, relation_pairs, embeddings, ner_tags_numeric, train_file, lang_model_name):

        dict_out = {'tokens': tokens,
                    'ne tags': ne_tags,
                    'relation pairs': relation_pairs,
                    'embeddings': embeddings,
                    'ner_tags_numeric': ner_tags_numeric
                    }

        f_name = lang_model_name + train_file

        lm_output_path = self.processed_data_path + self.out_folder_path
        if not os.path.isdir(lm_output_path):
            os.makedirs(lm_output_path)

        # Extract the last file
        with open(lm_output_path + f_name, 'w') as fp:
            json.dump(dict_out, fp)

        return

    def execute_preprocessing(self):

        counter = 0

        if self.lang_model == 'BERT':
            # Pre-trained uncased BERT base model and tokenizer
            bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            bert_lang_model = BertModel.from_pretrained('bert-base-uncased')
        elif self.lang_model == 'BioBERT':
            # Pre-trained BioBERT base model and tokenizer
            biobert_tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1", do_lower_case=True)
            biobert_lang_model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")
        elif self.lang_model == 'CharBERT':
            # Define the tokenizer
            charbert_tokenizer = BertTokenizer.from_pretrained('./character-bert-main/pretrained-models/bert-base-uncased/')
            # Load pre-trained CharacterBERT
            charbert_model = CharacterBertModel.from_pretrained('./character-bert-main/pretrained-models/general_character_bert/')
        elif self.lang_model == 'MedCharBERT':
            # Define the tokenizer
            medcharbert_tokenizer = BertTokenizer.from_pretrained('./character-bert-main/pretrained-models/bert-base-uncased/')
            # Load pre-trained CharacterBERT
            medcharbert_model = CharacterBertModel.from_pretrained('./character-bert-main/pretrained-models/medical_character_bert/')
        elif self.lang_model == 'ELMo_default':
            # Default ELMo model is 5.5B model, as recommended by AllenNLP
            default_options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'
            default_weights_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'
        elif self.lang_model == 'ELMo_pubmed':
            # Contributed ELMo Models / PubMed
            pubmed_options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pubmed/elmo_2x4096_512_2048cnn_2xhighway_options.json'
            pubmed_weights_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pubmed/elmo_2x4096_512_2048cnn_2xhighway_weights_PubMed_only.hdf5'           
            
        for i in range(len(self.original_dataset)):
            # Take the existing information from the generally pre-processed data set.
            tokens, ne_tags, relation_pairs, embeddings, ner_tags_numeric, filename = self.original_dataset[i]

            lang_model_name = self.lang_model + '_'
            # Take the embeddings using the selected language model
            if self.lang_model == 'BERT':
                embeddings = self.get_bert_embeddings(tokens, bert_tokenizer, bert_lang_model)
            elif self.lang_model == 'BioBERT':
                embeddings = self.get_bert_embeddings(tokens, biobert_tokenizer, biobert_lang_model)
            elif self.lang_model == 'CharBERT':
                embeddings = self.get_charbert_embeddings(tokens, charbert_tokenizer, charbert_model)
            elif self.lang_model == 'MedCharBERT':
                embeddings = self.get_charbert_embeddings(tokens, medcharbert_tokenizer, medcharbert_model)
            elif self.lang_model == 'GloVe':
                embeddings = self.get_glove_embeddings(tokens)
            elif self.lang_model == 'ELMo_default':
                embeddings = self.get_elmo_embeddings(tokens, default_options_file, default_weights_file)
            elif self.lang_model == 'ELMo_pubmed':
                embeddings = self.get_elmo_embeddings(tokens, pubmed_options_file, pubmed_weights_file)

            # Extracted the processed json file
            self.save_json(tokens,
                           ne_tags,
                           relation_pairs,
                           embeddings,
                           ner_tags_numeric,
                           filename,       
                           lang_model_name)

            counter += 1
            if counter % 100 == 0:
                print('{} sentences processed.'.format(counter))

        return