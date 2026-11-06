import unittest
from taoyao.personality import describe_tone, relationship_state_from_values
from taoyao.persona import build_system_prompt, load_persona


class TonePromptTests(unittest.TestCase):
    def test_all_axes_have_distinct_directions(self):
        for dim in ('AGG','INT','HUM','DEP','RAT','ACT'):
            low = describe_tone({dim:20})
            high = describe_tone({dim:80})
            self.assertNotEqual(low, high)
            self.assertIn(dim+'：明显偏向', low)
            self.assertIn(dim+'：明显偏向', high)
        self.assertIn('AGG：中性',describe_tone({'AGG':'invalid'}))
        self.assertNotIn('偏向——',describe_tone({'AGG':80},adapt_strength=0))

    def test_high_rat_keeps_reasoning_above_persona_mannerisms(self):
        high = describe_tone({'RAT': 80})
        low = describe_tone({'RAT': 20})
        self.assertIn('六维选行为，人设只修饰语气', high)
        self.assertIn('不以反话、猜心、赌气或冷落威胁代替讲理', high)
        self.assertNotIn('不以反话、猜心、赌气或冷落威胁代替讲理', low)
        self.assertIn('轻微闹别扭', low)

    def test_prompt_keeps_boundaries_time_and_one_tone_section(self):
        tone=describe_tone({'RAT':80})
        self.assertNotIn('适配强度参考', tone)
        self.assertNotIn('0.40', tone)
        self.assertNotIn('%', tone)
        for role in ('owner','user'):
            for group in (False,True):
                relationship=relationship_state_from_values(30,45,owner=role=='owner')
                prompt=build_system_prompt(load_persona(),['happy'], 'PRIVATE_SECRET', ['PRIVATE_FACT'],
                    role=role,is_group=group,relationship=relationship,tone_hint=tone,
                    current_time='2026-11-04 Asia/Shanghai')
                self.assertEqual(prompt.count('【六维的具体表达方向】'),1)
                self.assertIn('直接接话',prompt)
                self.assertIn('2026-11-04 Asia/Shanghai',prompt)
                self.assertNotIn('本轮权重分配',prompt)
                self.assertNotIn('第一反应：',prompt)
                if group or role!='owner':
                    self.assertNotIn('PRIVATE_SECRET',prompt)
                    self.assertNotIn('PRIVATE_FACT',prompt)
                if group:
                    self.assertIn('群聊隐私边界',prompt)
                if role=='user':
                    self.assertIn('不是哥哥',prompt)


if __name__=='__main__': unittest.main()
