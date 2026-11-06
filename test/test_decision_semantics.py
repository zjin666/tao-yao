import unittest

from taoyao.personality import (
    BUCKET_DELTA_MAP, BUCKET_EXAMPLES, BUCKET_DEFINITIONS,
    BUCKET_DECISION_ORDER, ClassificationResult, PersonalityScoringModel,
    RuleClassifier, describe_scores, describe_tone,
)


class DecisionSemanticsTests(unittest.TestCase):
    def test_buckets_describe_user_expression(self):
        classifier = RuleClassifier()
        cases = [
            ('我今天特别难过', 'INT', '强烈情感暴露'),
            ('我今天特别难过', 'RAT', '强情绪表达'),
            ('有点想你', 'INT', '轻度情感流露'),
            ('我只把你当工具', 'INT', '明显疏离'),
            ('我习惯把情绪藏起来', 'INT', '轻度回避'),
            ('我很珍惜你', 'INT', '强烈情感暴露'),
            ('我们按事实讨论', 'RAT', '严密论证'),
            ('直觉告诉我不要这样', 'RAT', '轻度感性'),
            ('好耶！！', 'ACT', '亢奋高能'),
            ('我们一起去看看', 'ACT', '积极活跃'),
            ('没什么精神', 'ACT', '低能量'),
            ('累死了', 'ACT', '明显疲惫'),
        ]
        for text, dim, bucket in cases:
            self.assertEqual(classifier.classify(text)[dim].bucket, bucket, (text, dim))

    def test_no_evidence_and_length_instructions_are_not_traits(self):
        classifier = RuleClassifier()
        for text in ('嗯', '哦', '一句话告诉我结论', '详细讲讲，我想听完整的分析'):
            self.assertEqual(classifier.classify(text)['ACT'].bucket, '中性')
        for text in ('让我先休息一下', '这个隐私我不想说'):
            self.assertEqual(classifier.classify(text)['INT'].bucket, '中性')
        self.assertEqual(BUCKET_DELTA_MAP['INT']['社交礼仪'], 0)
        self.assertEqual(classifier.classify('我很珍惜你')['RAT'].bucket, '中性')

    def test_normal_disagreement_does_not_accumulate_retaliation(self):
        result = RuleClassifier().classify('我有不同看法，但是愿意听你说')
        self.assertEqual(BUCKET_DELTA_MAP['AGG'][result['AGG'].bucket], 0)

    def test_pressure_concession_and_non_confrontational_requests(self):
        classifier = RuleClassifier()
        for text in ('就按我说的做', '别解释，照做就行', '你别找理由，先给我照办'):
            self.assertEqual(classifier.classify(text)['AGG'].bucket, '轻度施压')
        for text in ('算了，不争了，听你的', '这次我让步，就按你的方案来', '我还是不太同意，但这次先照你的做'):
            self.assertEqual(classifier.classify(text)['AGG'].bucket, '妥协让步')
        for text in ('帮我看看这个', '请直接告诉我结果', '只要答案，不用展开', '都可以，你挑吧',
                     '我选不出来，你帮我选', '我太累了，不想选了，你决定吧', '我相信你的眼光，你决定吧'):
            result = classifier.classify(text)['AGG']
            self.assertEqual(BUCKET_DELTA_MAP['AGG'][result.bucket], 0, text)
        self.assertEqual(BUCKET_DELTA_MAP['AGG'][classifier.classify('你决定就好')['AGG'].bucket], 0)
        self.assertEqual(classifier.classify('我错了')['AGG'].bucket, '顺从认错')
        self.assertEqual(classifier.classify('我不会退让')['AGG'].bucket, '强硬对抗')
        self.assertLess(0, BUCKET_DELTA_MAP['AGG']['轻度施压'])
        self.assertLess(BUCKET_DELTA_MAP['AGG']['轻度施压'], BUCKET_DELTA_MAP['AGG']['强硬对抗'])
        self.assertLess(BUCKET_DELTA_MAP['AGG']['顺从认错'], BUCKET_DELTA_MAP['AGG']['妥协让步'])
        self.assertLess(BUCKET_DELTA_MAP['AGG']['妥协让步'], 0)
        tone = describe_tone({'AGG':80})
        for phrase in ('面对轻度施压', '不自动反击', '立场和边界'):
            self.assertIn(phrase,tone)

    def test_other_axes_require_specific_expression_evidence(self):
        classifier = RuleClassifier()
        cases = [
            ('抱歉打扰一下', 'AGG', '中性'),
            ('垃圾文件放哪里', 'AGG', '中性'),
            ('我错了', 'AGG', '顺从认错'),
            ('你就是个废物', 'AGG', '攻击挑衅'),
            ('笑死了', 'HUM', '轻松玩笑'),
            ('这个包袱我接', 'HUM', '接梗互动'),
            ('确实', 'HUM', '中性'),
            ('你在吗', 'DEP', '中性'),
            ('我撑不住了', 'DEP', '中性'),
            ('不需要这个附件', 'DEP', '中性'),
            ('我自己解决', 'DEP', '自主解决'),
            ('别插手我的决定', 'DEP', '强硬独立'),
            ('陪我说会儿话', 'DEP', '隐性依赖'),
            ('只有你能帮我了', 'DEP', '强烈求助'),
        ]
        for text, dim, expected in cases:
            self.assertEqual(classifier.classify(text)[dim].bucket, expected, (text, dim))

    def test_neutral_round_does_not_inherit_previous_delta(self):
        model = PersonalityScoringModel()
        first, _ = model.update_with_details({'INT': ClassificationResult('强烈情感暴露', 1)}, first=True)
        second, details = model.update_with_details({})
        self.assertEqual(first, second)
        self.assertEqual(details['INT']['delta'], 0)

    def test_shared_vocabulary_and_prompt_branches(self):
        for dim in ('AGG', 'INT', 'DEP', 'HUM', 'RAT', 'ACT'):
            names = set(BUCKET_DELTA_MAP[dim]) - {'中性'}
            self.assertEqual(names, set(BUCKET_DEFINITIONS[dim]))
            self.assertEqual(names, set(BUCKET_DECISION_ORDER[dim]))
            self.assertEqual(names, set(BUCKET_EXAMPLES[dim]) - {'中性'})
        tone = describe_tone({'INT': 80, 'RAT': 20, 'ACT': 80})
        for word in ('破防', '闹别扭', 'ACT长短', '无挑衅不反击'):
            self.assertIn(word, tone)
        self.assertIn('偏展开长回复', describe_scores({'ACT': 80}))


if __name__ == '__main__':
    unittest.main()
