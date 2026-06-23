import pytest
from fastapi.testclient import TestClient

from app.main import create_app


pytestmark = pytest.mark.filterwarnings(
    "ignore:Using `httpx` with `starlette.testclient` is deprecated"
)


@pytest.fixture
def client() -> TestClient:
    with TestClient(create_app()) as test_client:
        yield test_client


def valid_prediction_form() -> dict[str, str]:
    return {
        "property_type": "flat",
        "sector": "sector 7",
        "bedRoom": "2",
        "bathroom": "2",
        "balcony": "1",
        "agePossession": "Moderately Old",
        "floor_num": "4",
        "total_floors": "4",
        "built_up_area": "1171",
        "servant_room": "0",
        "store_room": "0",
        "furnishing_type": "unfurnished",
        "luxury_category": "Low",
        "floor_category": "Low Floor",
    }


@pytest.mark.parametrize(
    "path",
    [
        "/",
        "/recommendation",
        "/analysis",
        "/deep-vision",
        "/rag",
        "/ai-assistant",
        "/docs",
    ],
)
def test_active_pages_are_available(
    client: TestClient,
    path: str,
) -> None:
    assert client.get(path).status_code == 200


def test_rag_page_exposes_pdf_upload_control(
    client: TestClient,
) -> None:
    response = client.get("/rag")

    assert 'type="file"' in response.text
    assert 'name="file"' in response.text
    assert 'action="/rag/upload#rag-upload-result"' in response.text
    assert "PDF Knowledge Base" in response.text


def test_agent_page_keeps_pdf_module_separate(
    client: TestClient,
) -> None:
    response = client.get("/ai-assistant")

    assert "Open PDF Upload" not in response.text
    assert "Ask Questions From Your PDF" not in response.text


@pytest.mark.parametrize(
    ("path", "form_action"),
    [
        ("/", "/predict#prediction-result"),
        ("/recommendation", "/recommendation#recommendation-result"),
        ("/deep-vision", "/deep-vision#vision-result"),
        ("/ai-assistant", "/ai-assistant#agent-result"),
        ("/rag", "/rag/upload#rag-upload-result"),
    ],
)
def test_main_forms_preserve_result_scroll_position(
    client: TestClient,
    path: str,
    form_action: str,
) -> None:
    response = client.get(path)

    assert f'action="{form_action}"' in response.text


def test_navigation_marks_current_page_active(
    client: TestClient,
) -> None:
    response = client.get("/analysis")

    assert 'href="/analysis" class="nav-item active"' in response.text


def test_product_branding_is_visible(
    client: TestClient,
) -> None:
    response = client.get("/")

    assert "<h2>PropLens</h2>" in response.text
    assert "Real Estate Intelligence, Clearly." in response.text
    assert "<title>PropLens" in response.text


def test_browser_validation_errors_render_readable_page(
    client: TestClient,
) -> None:
    form = valid_prediction_form()
    form["built_up_area"] = "3"

    response = client.post(
        "/predict",
        data=form,
        headers={"accept": "text/html"},
    )

    assert response.status_code == 422
    assert "Please check your input" in response.text
    assert "Built Up Area" in response.text


def test_health_endpoints_report_ready(client: TestClient) -> None:
    assert client.get("/health/live").json() == {"status": "ok"}

    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_prediction_form_validates_and_renders_result(
    client: TestClient,
) -> None:
    response = client.post("/predict", data=valid_prediction_form())

    assert response.status_code == 200
    assert "Estimated Market Price" in response.text
    assert "What influenced this estimate?" in response.text


def test_prediction_page_exposes_numeric_input_limits(
    client: TestClient,
) -> None:
    response = client.get("/")

    assert 'name="floor_num" min="-2" max="100"' in response.text
    assert (
        'name="built_up_area" min="101" max="100000"'
        in response.text
    )


def test_prediction_rejects_invalid_floor_relationship(
    client: TestClient,
) -> None:
    form = valid_prediction_form()
    form["floor_num"] = "10"
    form["total_floors"] = "4"

    response = client.post("/predict", data=form)

    assert response.status_code == 422


def test_recommendation_validates_limits_and_unknown_names(
    client: TestClient,
) -> None:
    invalid_limit = client.post(
        "/recommendation",
        data={"property_name": "Ireo Victory Valley", "top_n": "50"},
    )
    unknown_property = client.post(
        "/recommendation",
        data={"property_name": "Not A Real Property", "top_n": "5"},
    )

    assert invalid_limit.status_code == 422
    assert unknown_property.status_code == 404


def test_upload_routes_reject_unsupported_files_before_model_loading(
    client: TestClient,
) -> None:
    invalid_pdf = client.post(
        "/rag/upload",
        files={"file": ("unsafe.txt", b"not a pdf", "text/plain")},
    )
    invalid_image = client.post(
        "/deep-vision",
        files={"image": ("unsafe.txt", b"not an image", "text/plain")},
    )

    assert invalid_pdf.status_code == 400
    assert invalid_image.status_code == 400


def test_disabled_vision_renders_deployment_notice(
    client: TestClient,
    monkeypatch,
) -> None:
    from config.settings import Config

    monkeypatch.setattr(Config, "ENABLE_VISION", False)

    page = client.get("/deep-vision")
    upload = client.post(
        "/deep-vision",
        files={"image": ("room.jpg", b"image", "image/jpeg")},
        headers={"accept": "text/html"},
    )

    assert page.status_code == 200
    assert "Deep Vision is unavailable" in page.text
    assert upload.status_code == 400
