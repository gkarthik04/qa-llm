from qa.metrics import exact_match, f1_score, normalize_answer, recall_at_k, squad_scores, mean_reciprocal_rank


def test_normalize():
    assert normalize_answer("The  Denver Broncos!") == "denver broncos"


def test_exact_match_ignores_articles_and_case():
    assert exact_match("the Denver Broncos", "Denver Broncos") == 1.0
    assert exact_match("Broncos", "Denver Broncos") == 0.0


def test_f1_partial_overlap():
    # pred tokens {denver, broncos, team}, truth {denver, broncos}: P=2/3, R=1 -> F1=0.8
    assert abs(f1_score("Denver Broncos team", "Denver Broncos") - 0.8) < 1e-9
    assert f1_score("Carolina", "Denver Broncos") == 0.0


def test_squad_scores_takes_best_reference():
    s = squad_scores(["Broncos", "Paris"], [["Denver Broncos", "Broncos"], ["London"]])
    assert s["exact_match"] == 50.0
    assert s["n"] == 2


def test_recall_and_mrr():
    retrieved = [[3, 1, 2], [5, 6, 7], [9, 8, 4]]
    gold = [1, 7, 0]
    r = recall_at_k(retrieved, gold, [1, 2, 3])
    assert r == {"recall@1": 0.0, "recall@2": 100 / 3, "recall@3": 200 / 3}
    assert abs(mean_reciprocal_rank(retrieved, gold) - (1 / 2 + 1 / 3) / 3) < 1e-9
