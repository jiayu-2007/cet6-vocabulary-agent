"""CET-6 vocabulary self-test Agent powered by DeepSeek.

The local vocabulary file is deliberately used only for selecting new targets.
Answer grading, question generation, examples, confusing-word notes, and
mistake explanations are delegated to DeepSeek.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol


EN_TO_ZH = "英文→中文"
ZH_TO_EN = "中文→英文"
QUESTION_TYPES = (EN_TO_ZH, ZH_TO_EN)
END_COMMAND = "结束本次学习"
VIEW_WRONG_COMMAND = "查看错题"
SWITCH_TYPE_COMMAND = "切换题型"


class AgentError(Exception):
    """Expected, user-facing application error."""


class ConfigurationError(AgentError):
    pass


class DeepSeekError(AgentError):
    pass


def parse_task_count(text: str) -> int:
    """Extract the requested number of final correct words from a task.

    The task text may be natural Chinese, for example ``本次完成 10 个单词``.
    Parsing the number locally is only session control; it never evaluates an
    answer or supplies a vocabulary meaning.
    """

    match = re.search(r"(?<!\d)(\d+)(?!\d)", text)
    if not match:
        raise ValueError("任务要求中没有找到要完成的单词数量。")
    count = int(match.group(1))
    if count < 1:
        raise ValueError("要完成的单词数量必须大于 0。")
    return count


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_from_model(content: str) -> dict[str, Any]:
    """Parse a JSON object from a model response, including fenced JSON."""

    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise DeepSeekError("DeepSeek 返回的内容不是有效 JSON。")
        try:
            value = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise DeepSeekError("DeepSeek 返回的内容不是有效 JSON。") from exc
    if not isinstance(value, dict):
        raise DeepSeekError("DeepSeek 返回的 JSON 不是对象。")
    return value


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    model: str = "deepseek-chat"
    timeout: float = 45.0

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> "Settings":
        """Load required API settings from .env and the process environment."""

        if env_file is not None and env_file.exists():
            _load_env_file(env_file)
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        base_url = os.getenv("DEEPSEEK_BASE_URL", "").strip()
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()
        timeout_text = os.getenv("DEEPSEEK_TIMEOUT", "45").strip()
        if not api_key:
            raise ConfigurationError("缺少 DEEPSEEK_API_KEY，请在 .env 中配置。")
        if not base_url:
            raise ConfigurationError("缺少 DEEPSEEK_BASE_URL，请在 .env 中配置。")
        try:
            timeout = float(timeout_text)
        except ValueError as exc:
            raise ConfigurationError("DEEPSEEK_TIMEOUT 必须是数字。") from exc
        return cls(api_key=api_key, base_url=base_url, model=model, timeout=timeout)


def _load_env_file(path: Path) -> None:
    """Small .env loader so the CLI remains usable without python-dotenv."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConfigurationError(f"无法读取 .env：{exc}") from exc
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip().strip("\"'")
        # A process-level variable takes precedence over .env.
        os.environ.setdefault(name, value)


class Vocabulary:
    """Local list of English words/phrases used only for random selection."""

    def __init__(self, words: Iterable[str]):
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in words:
            if not isinstance(item, str) or not item.strip():
                raise AgentError("cet6_words.json 必须只包含非空英文单词或短语字符串。")
            word = item.strip()
            key = word.casefold()
            if key not in seen:
                cleaned.append(word)
                seen.add(key)
        if not cleaned:
            raise AgentError("cet6_words.json 不能为空。")
        self.words = tuple(cleaned)

    @classmethod
    def from_file(cls, path: Path) -> "Vocabulary":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise AgentError(f"找不到词库文件：{path}") from exc
        except json.JSONDecodeError as exc:
            raise AgentError(f"词库不是有效 JSON：{path}") from exc
        if not isinstance(data, list):
            raise AgentError("cet6_words.json 顶层必须是英文单词/短语字符串数组。")
        return cls(data)

    def pick(self, excluded: set[str], rng: random.Random) -> str:
        candidates = [word for word in self.words if word.casefold() not in excluded]
        if not candidates:
            raise AgentError("当前词库中没有可抽取的新单词。")
        return rng.choice(candidates)


class WrongQuestionStore:
    """Persistent independent wrong-question pool.

    It stores only target strings and attempt metadata. It never stores a local
    answer key; the API remains the sole judge.
    """

    def __init__(self, path: Path):
        self.path = path
        self._items: dict[str, dict[str, Any]] = {}
        self._load()

    @staticmethod
    def _key(term: str) -> str:
        return term.casefold()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AgentError(f"错题池不是有效 JSON：{self.path}") from exc
        if not isinstance(data, list):
            raise AgentError("错题池顶层必须是数组。")
        for item in data:
            if isinstance(item, str) and item.strip():
                term = item.strip()
                self._items[self._key(term)] = {
                    "term": term,
                    "attempts": 1,
                    "last_failed_at": None,
                }
            elif isinstance(item, dict) and isinstance(item.get("term"), str):
                term = item["term"].strip()
                if term:
                    record = dict(item)
                    record["term"] = term
                    record["attempts"] = int(record.get("attempts", 1))
                    self._items[self._key(term)] = record

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = sorted(self._items.values(), key=lambda item: item["term"].casefold())
        # Replace atomically when possible so an interrupted run does not leave
        # half-written persistent state.
        handle, temp_name = tempfile.mkstemp(
            prefix=f"{self.path.stem}-", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as temp_file:
                json.dump(payload, temp_file, ensure_ascii=False, indent=2)
                temp_file.write("\n")
            os.replace(temp_name, self.path)
        except OSError:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise

    def terms(self) -> list[str]:
        return [item["term"] for item in self._items.values()]

    def contains(self, term: str) -> bool:
        return self._key(term) in self._items

    def add(self, term: str) -> None:
        key = self._key(term)
        record = self._items.get(
            key,
            {"term": term, "attempts": 0, "last_failed_at": None},
        )
        record["attempts"] = int(record.get("attempts", 0)) + 1
        record["last_failed_at"] = _now()
        self._items[key] = record
        self._save()

    def resolve(self, term: str) -> None:
        if self._items.pop(self._key(term), None) is not None:
            self._save()

    def records(self) -> list[dict[str, Any]]:
        return sorted(self._items.values(), key=lambda item: item["term"].casefold())


class AIClient(Protocol):
    def generate_question(self, term: str, question_type: str) -> str: ...

    def grade_answer(
        self, term: str, question_type: str, question: str, user_answer: str
    ) -> dict[str, Any]: ...

    def learning_note(
        self,
        term: str,
        question_type: str,
        user_answer: str,
        correct: bool,
        rationale: str,
    ) -> dict[str, Any]: ...


class DeepSeekClient:
    """DeepSeek adapter. All semantic operations are made through this class."""

    def __init__(self, settings: Settings):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ConfigurationError(
                "未安装 openai 依赖，请先执行：pip install -r requirements.txt"
            ) from exc
        self._client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.timeout,
        )
        self._model = settings.model

    def _json_call(self, system: str, user: str) -> dict[str, Any]:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
        except Exception as exc:  # SDK exceptions vary between versions.
            raise DeepSeekError(f"调用 DeepSeek 失败：{exc}") from exc
        return _json_from_model(content)

    def generate_question(self, term: str, question_type: str) -> str:
        system = (
            "你是大学英语六级单词自测出题员。你只能根据用户给出的目标英文词/短语生成一道题，"
            "不得输出答案、同义词答案、词根提示或额外讲解。必须返回 JSON："
            '{"question":"..."}。题面用中文，简洁自然。'
        )
        if question_type == EN_TO_ZH:
            instruction = (
                f"题型是英文→中文。目标词/短语是 {term!r}。题面必须要求用户给出它的中文释义，"
                "可以在题面中显示该英文目标。"
            )
        else:
            instruction = (
                f"题型是中文→英文。目标词/短语是 {term!r}。请写出准确、自然的中文释义或场景描述，"
                "让用户据此写出英文目标；题面绝对不能出现目标英文、其明显变形或直接拼写提示。"
            )
        data = self._json_call(system, instruction)
        question = data.get("question")
        if not isinstance(question, str) or not question.strip():
            raise DeepSeekError("DeepSeek 没有生成有效题面。")
        if question_type == ZH_TO_EN and term.casefold() in question.casefold():
            raise DeepSeekError("DeepSeek 生成的中文→英文题面泄露了目标答案。")
        return question.strip()

    def grade_answer(
        self, term: str, question_type: str, question: str, user_answer: str
    ) -> dict[str, Any]:
        system = (
            "你是六级词汇自测的唯一判题人。必须使用你的英语语言知识判断用户答案与目标词/短语的"
            "语义是否相同或在当前题型下等价；允许合理的同义表达、词形变化和自然译法，不能只做字符串匹配。"
            "不要引用或假设存在任何本地词库答案。若答案只沾边、过于宽泛、词性不符或改变核心含义，判错。"
            '必须返回 JSON：{"correct":true或false,"rationale":"简短中文理由"}。'
        )
        user = json.dumps(
            {
                "target_english": term,
                "question_type": question_type,
                "question": question,
                "user_answer": user_answer,
            },
            ensure_ascii=False,
        )
        data = self._json_call(system, user)
        raw_correct = data.get("correct", data.get("is_correct"))
        if isinstance(raw_correct, bool):
            correct = raw_correct
        elif isinstance(raw_correct, str):
            correct = raw_correct.strip().lower() in {"true", "yes", "correct"}
        else:
            raise DeepSeekError("DeepSeek 没有返回明确的判题结果。")
        rationale = data.get("rationale", "")
        if not isinstance(rationale, str):
            rationale = str(rationale)
        return {"correct": correct, "rationale": rationale.strip()}

    def learning_note(
        self,
        term: str,
        question_type: str,
        user_answer: str,
        correct: bool,
        rationale: str,
    ) -> dict[str, Any]:
        system = (
            "你是英语六级词汇教练。请根据目标词/短语和这次作答生成学习反馈，所有解释用简体中文。"
            "必须返回 JSON，字段必须包括：message、example_en、example_zh、confusable。"
            "example_en 必须是包含目标词/短语的自然英文例句，example_zh 是其中文翻译。"
            "confusable 用于易混词/短语辨析；若没有典型易混项也要简短说明。"
            "不要臆造词义，不要把本地词库当作答案来源。"
        )
        user = json.dumps(
            {
                "target_english": term,
                "question_type": question_type,
                "user_answer": user_answer,
                "correct": correct,
                "judge_rationale": rationale,
                "feedback_requirement": "正确时巩固用法；错误时解释错因并帮助下次重测。",
            },
            ensure_ascii=False,
        )
        data = self._json_call(system, user)
        result: dict[str, Any] = {}
        for field in ("message", "example_en", "example_zh", "confusable"):
            value = data.get(field, "")
            result[field] = value.strip() if isinstance(value, str) else str(value)
        return result


@dataclass
class Question:
    term: str
    question_type: str
    text: str


class LearningSession:
    """One confirmed task and its one-question-at-a-time interaction state."""

    def __init__(
        self,
        target_count: int,
        vocabulary: Vocabulary,
        wrong_store: WrongQuestionStore,
        ai: AIClient,
        rng: random.Random | None = None,
        write: Callable[[str], None] = print,
    ):
        self.target_count = target_count
        self.vocabulary = vocabulary
        self.wrong_store = wrong_store
        self.ai = ai
        self.rng = rng or random.Random()
        self.write = write
        self.completed_terms: set[str] = set()
        self.correct_count = 0
        self.question_type: str | None = None
        self.current: Question | None = None
        self.active = True

    def _next_term(self) -> str:
        # Wrong items have priority, so they are independently re-tested until
        # a DeepSeek judgment marks them correct.
        wrong = [
            term
            for term in self.wrong_store.terms()
            if term.casefold() not in self.completed_terms
        ]
        if wrong:
            return self.rng.choice(wrong)
        return self.vocabulary.pick(
            excluded=self.completed_terms | {term.casefold() for term in self.wrong_store.terms()},
            rng=self.rng,
        )

    def _new_question(self, term: str | None = None) -> Question:
        if term is None:
            term = self._next_term()
        question_type = self.question_type or self.rng.choice(QUESTION_TYPES)
        self.question_type = question_type
        text = self.ai.generate_question(term, question_type)
        self.current = Question(term=term, question_type=question_type, text=text)
        return self.current

    def _show_question(self, question: Question) -> None:
        self.write(f"\n[{question.question_type}] {question.text}")
        if question.question_type == EN_TO_ZH:
            self.write(f"英文词/短语：{question.term}")

    def _show_wrong(self) -> None:
        records = self.wrong_store.records()
        if not records:
            self.write("错题池为空。")
            return
        self.write("当前持久错题池：")
        for record in records:
            self.write(f"- {record['term']}（累计答错 {record.get('attempts', 0)} 次）")

    def _switch_type(self) -> None:
        if self.question_type is None:
            self.question_type = self.rng.choice(QUESTION_TYPES)
        else:
            self.question_type = ZH_TO_EN if self.question_type == EN_TO_ZH else EN_TO_ZH
        if self.current is not None:
            self.current = self._new_question(self.current.term)
            self.write(f"已切换为 {self.question_type}，当前单词重新出题。")

    def _handle_answer(self, answer: str) -> None:
        assert self.current is not None
        question = self.current
        result = self.ai.grade_answer(
            question.term, question.question_type, question.text, answer
        )
        correct = bool(result.get("correct"))
        rationale = str(result.get("rationale", "")).strip()
        if correct:
            self.wrong_store.resolve(question.term)
            key = question.term.casefold()
            if key not in self.completed_terms:
                self.completed_terms.add(key)
                self.correct_count += 1
            self.write(
                f"回答正确！本次进度：{self.correct_count}/{self.target_count}。"
            )
        else:
            self.wrong_store.add(question.term)
            self.write("这次未通过，已加入持久错题池，后续会随机重测。")
            if rationale:
                self.write(f"判题说明：{rationale}")

        try:
            note = self.ai.learning_note(
                question.term,
                question.question_type,
                answer,
                correct,
                rationale,
            )
            self._show_note(note)
        except DeepSeekError as exc:
            # The answer has already been judged and persisted; a feedback
            # failure must not corrupt progress or stop the session.
            self.write(f"学习反馈暂时生成失败：{exc}")

        self.current = None
        if self.correct_count >= self.target_count:
            self.active = False
            self.write("已完成本次任务设定的最终答对数量，学习会话自动结束。")

    def _show_note(self, note: dict[str, Any]) -> None:
        if note.get("message"):
            self.write(f"讲解：{note['message']}")
        if note.get("example_en"):
            self.write(f"例句：{note['example_en']}")
        if note.get("example_zh"):
            self.write(f"译文：{note['example_zh']}")
        if note.get("confusable"):
            self.write(f"易混辨析：{note['confusable']}")

    def run(self, input_func: Callable[[str], str] = input) -> None:
        self._new_question()
        while self.active:
            assert self.current is not None
            self._show_question(self.current)
            answer = input_func(
                "你的答案（支持：查看错题 / 切换题型 / 结束本次学习）："
            ).strip()
            if answer == END_COMMAND:
                self.active = False
                self.write("已结束本次学习。错题池已保存。")
            elif answer == VIEW_WRONG_COMMAND:
                self._show_wrong()
            elif answer == SWITCH_TYPE_COMMAND:
                self._switch_type()
            elif not answer:
                self.write("答案不能为空，请继续作答。")
            else:
                self._handle_answer(answer)
            if self.active and self.current is None:
                self._new_question()


def confirm_task(task_text: str, input_func: Callable[[str], str] = input) -> int:
    """Receive and confirm a task before any question is generated."""

    while True:
        try:
            count = parse_task_count(task_text)
        except ValueError as exc:
            print(f"无法识别任务要求：{exc}")
            task_text = input_func("请重新输入本次任务要求：").strip()
            continue
        confirmation = input_func(
            f"已识别为：最终答对 {count} 个有效单词。确认开始吗？（Y/n）："
        ).strip().lower()
        if confirmation in {"", "y", "yes", "确认", "是"}:
            return count
        task_text = input_func("请重新输入本次任务要求：").strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeepSeek 六级单词自测 Agent")
    base = Path(__file__).resolve().parent
    parser.add_argument("--vocab", type=Path, default=base / "cet6_words.json")
    parser.add_argument(
        "--wrong-pool", type=Path, default=base / "wrong_questions.json"
    )
    parser.add_argument("--env-file", type=Path, default=base / ".env")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        vocabulary = Vocabulary.from_file(args.vocab)
        wrong_store = WrongQuestionStore(args.wrong_pool)
        settings = Settings.from_env(args.env_file)
        ai = DeepSeekClient(settings)
    except AgentError as exc:
        print(f"启动失败：{exc}")
        return 2

    print("六级单词自测 Agent 已就绪。")
    print("请先输入本次任务要求，例如：本次最终答对 10 个单词。")
    task_text = input("本次任务要求：").strip()
    try:
        target_count = confirm_task(task_text)
        if target_count > len(vocabulary.words):
            print(
                f"任务需要 {target_count} 个有效单词，但当前词库只有 {len(vocabulary.words)} 个不同词条。"
            )
            return 2
        LearningSession(
            target_count=target_count,
            vocabulary=vocabulary,
            wrong_store=wrong_store,
            ai=ai,
        ).run()
    except (AgentError, KeyboardInterrupt) as exc:
        if isinstance(exc, KeyboardInterrupt):
            print("\n已中断本次学习，错题池中已保存此前结果。")
        else:
            print(f"本次学习未能继续：{exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
