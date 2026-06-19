from src.agent.llm_router import LLMRouter

def test_router():

    router = LLMRouter()

    result = router.route(
        "Analyze Gurgaon real estate market"
    )

    print(result)

def test_average_price_query():

    router = LLMRouter()

    result = router.route(
        "What is the average price in Gurgaon?"
    )

    print(result)