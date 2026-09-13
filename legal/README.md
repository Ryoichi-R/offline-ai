# Legal artifact policy

現行のuser-built transport packageでは、Ollama model manifestのlicense layerをhash・size検証し、`legal/models/<manifest-relative-path>/LICENSE.txt`へ可視化します。

Publisherがprebuilt bundleを公開する場合は、版固定したSBOM、license原文、署名/hash、Python license stack、installerのpass-through termsをこの配下へ追加し、再配布審査後に`release-metadata.json`の承認を更新してください。

Local rerankerは搬送契約だけを提供します。runtime/modelの標準同梱、再配布、自動起動は、artifact固有のhash、license、完全オフライン動作、CPU性能が承認されるまで行いません。
