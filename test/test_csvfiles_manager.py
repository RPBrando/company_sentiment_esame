import src.csvfiles_manager as csv

def test_df_summary():
    summary = csv.df_summary()
    assert isinstance(summary, dict)
    assert "n. interaction" in summary
    assert "sentiment count" in summary
    assert "average score" in summary

    