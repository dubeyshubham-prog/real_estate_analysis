PRICE_PREDICTION_REQUIRED_FIELDS = [
    "property_type",
    "sector",
    "bedRoom",
    "bathroom",
    "balcony",
    "agePossession",
    "floor_num",
    "total_floors",
    "built_up_area",
    "servant_room",
    "store_room",
    "furnishing_type",
    "luxury_category",
    "floor_category"
]

def get_missing_price_fields(parsed_query):

    missing_fields = []

    for field in PRICE_PREDICTION_REQUIRED_FIELDS:

        if parsed_query.get(field) is None:

            missing_fields.append(field)

    return missing_fields