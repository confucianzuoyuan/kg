from neo4j import GraphDatabase

NEO4J_URI = "neo4j://localhost"
NEO4J_AUTH = ("neo4j", "BaiYuan329064BY")


def query_tx(tx, q):
    result = tx.run(q)

    return [record for record in result]


class AnswerSearcher:
    def __init__(self):
        self.d = GraphDatabase.driver(uri=NEO4J_URI, auth=NEO4J_AUTH)
        self.d.verify_connectivity()
        self.session = self.d.session(database="neo4j")

    def search_main(self, cyphers):
        final_answers = []
        for c in cyphers:
            question_type = c['question_type']
            answers = []
            queries = c['cypher']
            for q in queries:
                results = self.session.execute_read(
                    query_tx, q)
                answers += results
            final_answer = self.answer_prettify(question_type, answers)
            if final_answer:
                final_answers.append(final_answer)
        return final_answers

    def answer_prettify(self, question_type, answers):
        final_answer = []
        if not answers:
            return ''
        if question_type == 'sku_info':
            desc = [i['m']['sku_name'] for i in answers]
            product = answers[0]['m']['spu_name']
            final_answer = '{0}的信息：{1}'.format(
                product, '；'.join(list(set(desc))[:5]))

        return final_answer
