import unittest
from types import SimpleNamespace

from taoyao.app import TaoYaoBot
from taoyao.knowledge import knowledge_inventory_question


class RememberQuestionTests(unittest.TestCase):
    def memory_text(self, text):
        return TaoYaoBot._memory_request_text(SimpleNamespace(), text)

    def test_what_did_you_remember_is_a_question_not_a_request(self):
        for text in ('你记住了什么', '你都记住了些什么？', '记住了哪些', '你记下了什么'):
            self.assertEqual(self.memory_text(text), '', text)

    def test_other_question_wordings_are_not_requests(self):
        for text in ('你记了些啥', '记住的有哪些？', '你都记住过什么', '记着些什么呢', '你记住什么了？'):
            self.assertEqual(self.memory_text(text), '', text)

    def test_statements_containing_question_words_are_still_requests(self):
        self.assertEqual(self.memory_text('记住我爱吃啥都行'), '我爱吃啥都行')
        self.assertEqual(self.memory_text('帮我记住我不喜欢吃什么香菜之类的东西'), '我不喜欢吃什么香菜之类的东西')

    def test_real_remember_requests_still_work(self):
        self.assertEqual(self.memory_text('帮我记住明天开会'), '明天开会')
        self.assertEqual(self.memory_text('记住守夜是女生'), '守夜是女生')

    def test_what_did_you_remember_queries_knowledge(self):
        for text in ('你记住了什么', '你都记住了些什么？', '记住了哪些'):
            self.assertTrue(knowledge_inventory_question(text), text)

    def test_remembered_about_me_is_not_knowledge_inventory(self):
        self.assertFalse(knowledge_inventory_question('你记住了我什么'))
        self.assertFalse(knowledge_inventory_question('我记住了什么'))


if __name__ == '__main__':
    unittest.main()
