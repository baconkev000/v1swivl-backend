"""Rank extraction for DataForSEO Labs ranked_keywords/live item shapes."""

from accounts.dataforseo_utils import _rank_from_ranked_keywords_item, compute_ranked_metrics


def test_rank_from_nested_ranked_serp_element_serp_item():
    item = {
        "keyword_data": {
            "keyword": "ai consulting austin",
            "keyword_info": {"search_volume": 500},
        },
        "ranked_serp_element": {
            "serp_item": {
                "rank_absolute": 7,
                "rank_group": 7,
            }
        },
    }
    assert _rank_from_ranked_keywords_item(item) == 7


def test_rank_prefers_top_level_absolute_over_nested_group():
    item = {
        "rank_absolute": 2,
        "ranked_serp_element": {"serp_item": {"rank_group": 9}},
    }
    assert _rank_from_ranked_keywords_item(item) == 2


def test_compute_ranked_metrics_sets_rank_for_nested_shape():
    items = [
        {
            "keyword_data": {
                "keyword": "local seo",
                "keyword_info": {"search_volume": 1200},
            },
            "ranked_serp_element": {"serp_item": {"rank_absolute": 4}},
        }
    ]
    metrics = compute_ranked_metrics(items)
    ranked = [k for k in metrics["top_keywords"] if k.get("keyword") == "local seo"]
    assert len(ranked) == 1
    assert ranked[0].get("rank") == 4
    assert metrics["top3_positions"] == 0
    assert metrics["top10_positions"] == 1
