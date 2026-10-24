import os

os.environ.pop("OLLAMA_HOST", None)

import search  # noqa: E402


def test_rrf_merges_windows_and_posix_path_keys():
    keyword = [
        {
            "path": r"rules\leave.md",
            "snippet": "休暇",
            "score": 10,
            "keyword_score": 10,
            "source": "keyword",
        }
    ]
    embed = [
        {
            "path": "rules/leave.md",
            "snippet": "",
            "score": 0.9,
            "embedding_score": 0.9,
            "source": "embedding",
        }
    ]

    merged = search.merge_results(keyword, embed)

    assert len(merged) == 1
    assert merged[0]["path"] == "rules/leave.md"
    assert merged[0]["snippet"] == "休暇"
    assert merged[0]["keyword_score"] == 10
    assert merged[0]["embedding_score"] == 0.9
    assert merged[0]["rrf_score"] > 1 / 60
