import os


class QuestionClassifier:
    def __init__(self):
        cur_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.product_path = os.path.join(cur_dir, 'dict/product.txt')
        self.product_words = [i.strip()
                              for i in open(self.product_path) if i.strip()]
        self.qwds = ['内存多大', '机身内存', '内存', '运行内存', '运存']

    def classify(self, question):
        data = {}
        data['args'] = {}
        types = {'product': 1}
        question_type = 'others'

        question_types = []

        # sku info
        if self.check_words(self.qwds, question) and ('product' in types):
            question_type = 'sku_info'
            question_types.append(question_type)

        data['question_types'] = question_types

        return data

    '''基于特征词进行分类'''

    def check_words(self, wds, sent):
        for wd in wds:
            if wd in sent:
                return True
        return False


if __name__ == '__main__':
    handler = QuestionClassifier()
    while 1:
        question = input('input an question:')
        data = handler.classify(question)
        print(data)
