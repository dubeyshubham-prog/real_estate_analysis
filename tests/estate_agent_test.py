from src.agent.estate_agent import EstateAgent

def test_estate_agent_process_query():
    agent = EstateAgent()

    query = "I want a 3 BHK flat under 2 Cr for investment"

    result = agent.process_query(query)

    print(result)

    assert result["parsed_query"]["bhk"] == 3
    assert result["parsed_query"]["budget"] == 2.0

    assert result["selected_tool"] in [
        "analysis",
        "recommendation"
    ]

    assert result["tool_result"]["status"] in [
        "success",
        "missing_input"
    ]

    assert result["parsed_query"]["property_name"] is None


def test_estate_agent_real_property_recommendation():

    agent = EstateAgent()

    query = "Recommend properties similar to DLF The Arbour"

    result = agent.process_query(query)

    print(result)

    assert result["selected_tool"] == "recommendation"

    assert result["tool_result"]["status"] == "success"

    assert len(
        result["tool_result"]["recommendations"]
    ) > 0


def test_estate_agent_analysis_tool():

    agent = EstateAgent()

    query = "Analyze Gurgaon real estate market"

    result = agent.process_query(query)

    print(result)

    assert result["selected_tool"] == "analysis"

    assert result["tool_result"]["status"] == "success"

    assert result["parsed_query"]["intent"] == "analysis"

    assert "summary" in result["tool_result"]

    summary = result["tool_result"]["summary"]

    assert summary["total_properties"] > 0

    assert summary["average_price_cr"] > 0


def test_price_prediction_redirect():

    agent = EstateAgent()

    query = "Predict price of 3 BHK flat"

    result = agent.process_query(query)

    print(result)

    assert result["selected_tool"] == "price"

    assert result["tool_result"]["status"] == "redirect"

    assert result["tool_result"]["redirect_url"] == "/predict"

def test_average_price_query_uses_analysis_tool():

    agent = EstateAgent()

    query = "What is the average price in Gurgaon?"

    result = agent.process_query(query)

    print(result)

    assert result["selected_tool"] == "analysis"
    assert result["tool_result"]["status"] == "success"
    assert "Average Property Price" in result["final_response"]