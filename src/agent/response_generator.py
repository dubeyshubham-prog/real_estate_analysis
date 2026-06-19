class AgentResponseGenerator:

    def generate_market_analysis_response(
            self,
            tool_result
    ):

        summary = tool_result["summary"]

        expensive = ", ".join(
            summary["top_expensive_sectors"].keys()
        )

        affordable = ", ".join(
            summary["top_affordable_sectors"].keys()
        )

        response = (
            f"📊 Gurgaon Market Summary\n\n"

            f"Total Properties: "
            f"{summary['total_properties']}\n\n"

            f"Average Property Price: "
            f"₹{summary['average_price_cr']} Cr\n\n"

            f"Most Expensive Sectors:\n"
            f"{expensive}\n\n"

            f"Most Affordable Sectors:\n"
            f"{affordable}"
        )

        return response

    def generate_missing_fields_response(
            self,
            missing_fields
    ):
        readable_name_map = {
            "property_type": "property type",
            "sector": "sector",
            "bedRoom": "bedroom",
            "bathroom": "bathroom",
            "balcony": "balcony",
            "agePossession": "property age / possession status",
            "floor_num": "floor number",
            "total_floors": "total floors",
            "built_up_area": "built-up area",
            "servant_room": "servant room availability",
            "store_room": "store room availability",
            "furnishing_type": "furnishing type",
            "luxury_category": "luxury category",
            "floor_category": "floor category"
        }

        readable_fields = [
            readable_name_map.get(field, field)
            for field in missing_fields
        ]

        fields_text = ", ".join(readable_fields)

        return (
            "I need some more details before I can predict the price. "
            f"Please provide: {fields_text}."
        )

    def generate_recommendation_response(
            self,
            tool_result
    ):

        property_name = tool_result["property_name"]

        recommendations = tool_result["recommendations"]

        property_names = [
            item["PropertyName"]
            for item in recommendations
        ]

        recommendation_text = "\n".join(
            [
                f"{index + 1}. {name}"
                for index, name in enumerate(property_names)
            ]
        )

        response = (
            f"Based on {property_name}, I found these similar properties:\n\n"
            f"{recommendation_text}"
        )
        return response

    def generate_investment_guidance_response(
            self,
            recommendation_result
    ):
        summary = recommendation_result["summary"]

        return (
            f"For a budget of ₹{recommendation_result['budget']} Cr "
            f"and {recommendation_result['bhk']} BHK investment requirement, "
            f"Gurgaon average property price is ₹{summary['average_price_cr']} Cr. "
            f"Consider exploring affordable sectors such as "
            f"{', '.join(summary['top_affordable_sectors'].keys())}."
        )