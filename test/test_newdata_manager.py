import src.newdata_manager as manager

name_file = "train.csv"

def test_newdata_manager():

    data_obj = manager.Dataset(name_file)

    x, y = data_obj.return_text_target()

    assert  len(x) == len(y)
