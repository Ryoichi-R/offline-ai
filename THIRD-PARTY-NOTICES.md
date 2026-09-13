# Third-Party Notices

この文書はactiveなoffline-ai配布候補に関する第三者componentの台帳です。実際の配布物はmanifestとchecksumで確認してください。

| Component              | Version                       | License / condition  | Active handling                                                 |
| ---------------------- | ----------------------------- | -------------------- | --------------------------------------------------------------- |
| Ollama                 | manifestで固定                | upstream terms       | installerのhash、版、署名を検証する                             |
| Python                 | 3.11.9                        | PSF License          | installerのhash、版、署名を検証する                             |
| PowerShell 7           | optional                      | MIT                  | 手動同梱時に版・hash・条件を確認する                            |
| GPT-OSS model          | gpt-oss:20b（manifestで固定） | model license layer  | 現行balanced候補。license layerのhashとsizeを検証して可視化する |
| Qwen3.5 model          | manifestで固定                | model license layer  | license layerのhashとsizeを検証して可視化する                   |
| BGE-M3 model           | manifestで固定                | model license layer  | license layerのhashとsizeを検証して可視化する                   |
| Local reranker payload | optional transport only       | artifactごとに要審査 | 標準同梱・自動起動しない                                        |

PDF変換runtime、Xpdf、Git for Windows fallback、structured parserはactive componentではありません。
