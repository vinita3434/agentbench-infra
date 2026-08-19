# Resolved tasks

Episodes the verifier scored **resolved**. Split out of `master_table.md`; regenerate with `python run/master_table.py`.

Runs with no verdict (ERROR, not verified) appear in neither table — a harness failure is not a model failure.

| # | Task ID | Task name | Lang | Model | Passed on | Turns | Wall | Cost |
|---|---|---|---|---|---|---|---|---|
| 1 | `prettier__prettier-3382` | Typescript: decorator + readonly + comment leads to un… | JavaScript | claude-opus-4.8 | attempt 1 | 52 | 5m30s | $0.9670 |
| 2 | `sveltejs__svelte-1310` | Spread properties cause CSS to be DCE'd incorrectly | JavaScript | claude-opus-4.8 | attempt 1 | 32 | 3m17s | $0.4174 |
| 3 | `sveltejs__svelte-3141` | :not(...) styles are broken | JavaScript | claude-opus-4.8 | attempt 1 | 23 | 3m27s | $0.3404 |
| 4 | `prettier__prettier-12930` | Syntax error after formatting when using computed key … | JavaScript | claude-opus-4.8 | attempt 1 | 32 | 3m23s | $0.3304 |
| 5 | `serverless__serverless-4794` | AWS deploy fails with empty error message if S3.headOb… | JavaScript | claude-opus-4.8 | attempt 1 | 12 | 1m09s | $0.2453 |
| 6 | `prettier__prettier-3405` | 1.9: Trailing comma not adding in code blocks in markd… | JavaScript | claude-opus-4.8 | attempt 1 | 8 | 52s | $0.1416 |
| 7 | `langchain-ai__langchain-4646` | PydanticOutputParser has high chance failing when comp… | Python | claude-opus-4.8 | attempt 1 | 6 | 24s | $0.0736 |
| 8 | `sveltejs__svelte-906` | Attribute name only (no value) CSS selector throws if … | JavaScript | claude-opus-4.8 | attempt 1 | 4 | 17s | $0.0437 |
| 9 | `langchain-ai__langchain-6765` | Recent tags change causes AttributeError: 'str' object… | Python | claude-opus-4.8 | attempt 1 | 3 | 11s | $0.0384 |
| 10 | `langchain-ai__langchain-7653` | SQLite LLM cache clear does not take effect | Python | claude-opus-4.8 | attempt 1 | 4 | 16s | $0.0372 |
| 11 | `serverless__serverless-5602` | Parent paths no longer working for package inclusions/… | JavaScript | claude-sonnet-5 | attempt 1 | 56 | 7m37s | $0.5037 |
| 12 | `serverless__serverless-5571` | schedule from file with default value removes space | JavaScript | claude-sonnet-5 | attempt 1 | 40 | 6m17s | $0.3029 |
| 13 | `sveltejs__svelte-2092` | Proposal: `dev: true` only affects runtime warnings an… | JavaScript | claude-sonnet-5 | attempt 1 | 40 | 85m03s | $0.2533 |
| 14 | `serverless__serverless-7374` | serverlessrc keeps changing | JavaScript | claude-sonnet-5 | attempt 1 | 23 | 2m18s | $0.1680 |
| 15 | `prettier__prettier-361` | Export extension syntax not formatted correctly | JavaScript | claude-sonnet-5 | attempt 1 | 31 | 3m59s | $0.1522 |
| 16 | `serverless__serverless-5775` | Fallback value for unset variable is removing spaces | JavaScript | claude-sonnet-5 | attempt 1 | 27 | 2m23s | $0.1388 |
| 17 | `langchain-ai__langchain-3367` | Terminal tool gives `ValueError: Could not parse LLM o… | Python | claude-sonnet-5 | attempt 1 | 19 | 1m47s | $0.0943 |
| 18 | `serverless__serverless-7031` | AWS - ability to request DynamoDB.DocumentClient in a … | JavaScript | claude-sonnet-5 | attempt 1 | 14 | 1m01s | $0.0564 |
| 19 | `langchain-ai__langchain-6456` | ChatPromptTemplate with partial variables is giving va… | Python | claude-sonnet-5 | attempt 1 | 13 | 55s | $0.0522 |
| 20 | `serverless__serverless-5775` | Fallback value for unset variable is removing spaces | JavaScript | kimi-k2.7-code | attempt 2 | 35 | 4m35s | $0.4502 |
| 21 | `serverless__serverless-5602` | Parent paths no longer working for package inclusions/… | JavaScript | kimi-k2.7-code | attempt 2 | 40 | 3m53s | $0.3635 |
| 22 | `sveltejs__svelte-3141` | :not(...) styles are broken | JavaScript | kimi-k2.7-code | attempt 3 (after 1 fail) | 65 | 8m24s | $0.3575 |
| 23 | `langchain-ai__langchain-6456` | ChatPromptTemplate with partial variables is giving va… | Python | kimi-k2.7-code | attempt 2 | 48 | 4m47s | $0.2139 |
| 24 | `prettier__prettier-3405` | 1.9: Trailing comma not adding in code blocks in markd… | JavaScript | kimi-k2.7-code | attempt 2 | 36 | 3m03s | $0.2100 |
| 25 | `langchain-ai__langchain-3367` | Terminal tool gives `ValueError: Could not parse LLM o… | Python | kimi-k2.7-code | attempt 2 | 31 | 1m35s | $0.1124 |
| 26 | `prettier__prettier-12930` | Syntax error after formatting when using computed key … | JavaScript | kimi-k2.7-code | attempt 2 | 41 | 4m29s | $0.0937 |
| 27 | `serverless__serverless-7031` | AWS - ability to request DynamoDB.DocumentClient in a … | JavaScript | kimi-k2.7-code | attempt 2 | 14 | 2m34s | $0.0597 |
| 28 | `langchain-ai__langchain-7653` | SQLite LLM cache clear does not take effect | Python | kimi-k2.7-code | attempt 2 | 17 | 1m02s | $0.0557 |

**28 episodes, $6.2735 total.**
 Turns: min 3, median 31, max 65.
