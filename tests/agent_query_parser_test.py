from src.agent.query_parser import QueryParser


def test_query_parser():
    parser = QueryParser()

    query = "I want a 3 BHK flat under 2 Cr for investment"

    result = parser.parse(query)

    print(result)

    assert result["bhk"] == 3
    assert result["budget"] == 2.0
    assert result["property_type"] == "flat"
    assert result["purpose"] == "investment"

# test_query_parser()

def test_property_name_extraction():

    parser = QueryParser()

    result = parser.parse(
        "Recommend properties similar to DLF The Crest"
    )

    print(result)

    assert result["property_name"] == "Dlf The Crest"

# test_property_name_extraction()