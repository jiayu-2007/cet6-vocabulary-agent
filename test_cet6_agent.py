import json
import random
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

from cet6_agent import (
    EN_TO_ZH,
    ZH_TO_EN,
    LearningSession,
    Vocabulary,
    WrongQuestionStore,
    parse_task_count,
)


@contextmanager
def project_temp_directory():
    """Use a project-local test directory for locked-down Windows runners."""

    directory = Path(__file__).resolve().parent / f".test-data-{uuid.uuid4().hex}"
    directory.mkdir()
    try:
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


class FakeAI:
    def __init__(self, correct_answers=None):
        self.correct_answers = correct_answers or {}
        self.grade_calls = []
        self.note_calls = []

    def generate_question(self, term, question_type):
        return f"题面：{term if question_type == EN_TO_ZH else '请写出对应英文'}"

    def grade_answer(self, term, question_type, question, user_answer):
        self.grade_calls.append((term, question_type, user_answer))
        return {
            "correct": user_answer == self.correct_answers.get(term, "right"),
            "rationale": "fake rationale",
        }

    def learning_note(self, term, question_type, user_answer, correct, rationale):
        self.note_calls.append((term, correct))
        return {
            "message": "fake note",
            "example_en": f"An example with {term}.",
            "example_zh": "例句翻译",
            "confusable": "fake distinction",
        }


class AgentTests(unittest.TestCase):
    def test_parse_task_count(self):
        self.assertEqual(parse_task_count("本次最终答对 12 个有效单词"), 12)
        with self.assertRaises(ValueError):
            parse_task_count("先熟悉一下词汇")

    def test_vocabulary_is_selection_only_list(self):
        vocab = Vocabulary(["coherent", "comply with", "coherent"])
        self.assertEqual(vocab.words, ("coherent", "comply with"))
        self.assertEqual(vocab.pick({"coherent"}, random.Random(1)), "comply with")

    def test_wrong_pool_persists_and_resolves(self):
        with project_temp_directory() as directory:
            path = Path(directory) / "wrong.json"
            store = WrongQuestionStore(path)
            store.add("abrupt")
            store.add("abrupt")
            self.assertTrue(store.contains("ABRUPT"))
            reloaded = WrongQuestionStore(path)
            self.assertEqual(reloaded.records()[0]["attempts"], 2)
            reloaded.resolve("abrupt")
            self.assertEqual(reloaded.terms(), [])

    def test_wrong_answer_is_persisted_and_later_correct_answer_counts(self):
        with project_temp_directory() as directory:
            store = WrongQuestionStore(Path(directory) / "wrong.json")
            ai = FakeAI()
            outputs = []
            session = LearningSession(
                target_count=1,
                vocabulary=Vocabulary(["abrupt"]),
                wrong_store=store,
                ai=ai,
                rng=random.Random(0),
                write=outputs.append,
            )
            answers = iter(["wrong", "right"])
            session.run(lambda _: next(answers))
            self.assertEqual(session.correct_count, 1)
            self.assertFalse(store.contains("abrupt"))
            self.assertEqual([call[2] for call in ai.grade_calls], ["wrong", "right"])
            self.assertIn("自动结束", "\n".join(outputs))

    def test_switch_type_keeps_current_term(self):
        with project_temp_directory() as directory:
            store = WrongQuestionStore(Path(directory) / "wrong.json")
            ai = FakeAI()
            session = LearningSession(
                target_count=1,
                vocabulary=Vocabulary(["abrupt"]),
                wrong_store=store,
                ai=ai,
                rng=random.Random(1),
                write=lambda _: None,
            )
            answers = iter(["切换题型", "right"])
            session.run(lambda _: next(answers))
            self.assertEqual(session.correct_count, 1)
            self.assertEqual(ai.grade_calls[0][0], "abrupt")
            self.assertEqual(ai.grade_calls[0][1], ZH_TO_EN)


if __name__ == "__main__":
    unittest.main()
