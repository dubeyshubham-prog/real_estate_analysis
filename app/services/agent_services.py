from src.agent.estate_agent import EstateAgent

class AgentService:

    def __init__(self):

        self.agent = EstateAgent()

    def ask_agent(
            self,
            query
    ):

        result = self.agent.process_query(
            query
        )

        return result