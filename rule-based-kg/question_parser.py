class QuestionPaser:

    '''构建实体节点'''

    def build_entitydict(self, args):
        entity_dict = {}
        for arg, types in args.items():
            for type in types:
                if type not in entity_dict:
                    entity_dict[type] = [arg]
                else:
                    entity_dict[type].append(arg)

        return entity_dict

    def parser_main(self, res_classify):
        args = res_classify['args']
        entity_dict = self.build_entitydict(args)
        entity_dict = {'product': ['Apple iPhone 12']}
        question_types = res_classify['question_types']
        cyphers = []
        for question_type in question_types:
            _cypher = {}
            _cypher['question_type'] = question_type
            cypher = []
            if question_type == 'sku_info':
                cypher = self.cypher_transfer(
                    question_type, entity_dict.get('product'))

            if cypher:
                _cypher['cypher'] = cypher
                cyphers.append(_cypher)

        return cyphers

    def cypher_transfer(self, question_type, entities):
        if not entities:
            return []

        cypher = []
        if question_type == 'sku_info':
            cypher = ["MATCH (m:SkuInfo {{spu_name:'{0}'}}) RETURN m".format(
                i) for i in entities]

        return cypher


if __name__ == '__main__':
    handler = QuestionPaser()
