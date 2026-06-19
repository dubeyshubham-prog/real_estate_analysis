from src.agent.estate_agent import EstateAgent

def test_web_search():

    agent = EstateAgent()

    result = agent.process_query(
        "latest real estate news"
    )

    print(result)

    assert (
        "web_search"
        in result["execution_results"]
    )