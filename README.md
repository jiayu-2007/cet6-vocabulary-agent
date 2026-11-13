# CET-6 单词自测 Agent

这是一个一问一答的命令行六级词汇自测 Agent。它只从本地 `cet6_words.json` 随机抽取英文单词/短语；题目生成、同义判题、例句、易混词辨析和错题讲解全部通过 DeepSeek API 完成。

## 使用

1. 安装 Python 3.10 或更高版本。
2. 安装依赖：

   ```powershell
   pip install -r requirements.txt
   ```

3. 复制 `.env.example` 为 `.env`，填写自己的 `DEEPSEEK_API_KEY` 和 `DEEPSEEK_BASE_URL`。密钥不会写入代码。
4. 按需替换 `cet6_words.json`。该文件必须是只包含英文单词/短语字符串的 JSON 数组，不要放中文释义、例句或答案。
5. 启动：

   ```powershell
   python cet6_agent.py
   ```

首次启动只会要求输入本次任务，例如“本次最终答对 10 个单词”，并在确认后才出第一道题。

## 交互规则

- 题型随机为“英文→中文”或“中文→英文”，每次只显示一道题。
- 只有 DeepSeek 判定正确的不同词条才计入完成数量；答错会写入 `wrong_questions.json`。
- 错题优先进入后续抽题，并持续重测；答对后才从错题池移除并计数。
- 在答案输入处支持：`查看错题`、`切换题型`、`结束本次学习`。
- 达到任务要求的最终答对数量后自动结束。

## 可选参数

```powershell
python cet6_agent.py --vocab .\cet6_words.json --wrong-pool .\wrong_questions.json --env-file .\.env
```

`wrong_questions.json` 是运行时生成的持久数据文件，里面只保存错题词条和次数，不保存判题答案。

