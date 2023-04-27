class Data_Proc:
    def __init__(self, initial_data_path, processed_data_path):
        self.initial_data_path = initial_data_path
        self.out_path = processed_data_path + 'original/'
        self.ade_full = self.read_data()

    def read_data(self):
        # Read the initial ADE data
        with open(self.initial_data_path) as json_file:
            ade_full = json.load(json_file)

        return ade_full

    def lower_casing(self, data_instance):
        lower_case_tokens = [t.lower() for t in data_instance['tokens']]
        return lower_case_tokens

    def get_ne_tags(self, data_instance):
        # Initialize a list with 'O' tags
        ne_tags = []
        ner_tags_numeric = []
        for i in range(len(data_instance['tokens'])):
            ne_tags.append('O')
            ner_tags_numeric.append(0)

        # Update the list based on the entities
        for en in data_instance['entities']:
            if en['type'] == 'Adverse-Effect':
                ne_tags[en['start']] = 'B-AE'
                ner_tags_numeric[en['start']] = 3
                for i in range(en['start'] + 1, en['end']):
                    ne_tags[i] = 'I-AE'
                    ner_tags_numeric[i] = 4
            elif en['type'] == 'Drug':
                ne_tags[en['start']] = 'B-DRUG'
                ner_tags_numeric[en['start']] = 1
                for i in range(en['start'] + 1, en['end']):
                    ne_tags[i] = 'I-DRUG'
                    ner_tags_numeric[i] = 2

        return ne_tags, ner_tags_numeric

    def get_relation_pairs(self, data_instance):
        relation_pairs = []
        for r in data_instance['relations']:
            ae = data_instance['entities'][r['head']]
            drug = data_instance['entities'][r['tail']]
            relation_pairs.append([drug['end'] - 1, ae['end'] - 1])

        return relation_pairs

    def save_json(self, token_list, ne_tags, ner_tags_numeric, relation_pairs, data_instance):
        dict_out = {'tokens': token_list,
                    'ne tags': ne_tags,
                    'relation pairs': relation_pairs,
                    'embeddings': [],
                    'ner_tags_numeric': ner_tags_numeric
                    }

        f_name = str(data_instance['orig_id']) + '.json'

        if not os.path.isdir(self.out_path):
            os.makedirs(self.out_path)

        # Extract the last file
        with open(self.out_path + f_name, 'w') as fp:
            json.dump(dict_out, fp)

        return

    def execute_preprocessing(self):
        counter = 0
        for data_instance in self.ade_full:
            # Take a lower-cased version of the token list.
            token_list = self.lower_casing(data_instance)

            # Create the ne tag list.
            ne_tags, ner_tags_numeric = self.get_ne_tags(data_instance)

            # Create the relation-pair list.
            relation_pairs = self.get_relation_pairs(data_instance)

            # Extracted the processed json file
            self.save_json(token_list, ne_tags, ner_tags_numeric, relation_pairs, data_instance)

            counter += 1
            if counter % 100 == 0:
                print('{} sentences processed.'.format(counter))

        return