class WebSearchTool:

    def search(
            self,
            query
    ):

        return {
            "status": "success",
            "message": "Web search completed",
            "query": query,
            "results": [
                f"Latest information about {query}"
            ]
        }