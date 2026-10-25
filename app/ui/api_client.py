
from pathlib import Path
import requests


class ProductionAPIClient:
    """
    Client for the validated VinBigData_XAI FastAPI service.
    """

    def __init__(
        self,
        base_url="http://127.0.0.1:8765",
        timeout=300,
    ):

        self.base_url = (
            base_url.rstrip("/")
        )

        self.timeout = timeout

    def health(self):

        response = requests.get(
            self.base_url + "/health",
            timeout=self.timeout,
        )

        response.raise_for_status()

        return response.json()

    def info(self):

        response = requests.get(
            self.base_url + "/info",
            timeout=self.timeout,
        )

        response.raise_for_status()

        return response.json()

    def predict(
        self,
        image_path,
    ):

        image_path = Path(
            image_path
        )

        if not image_path.exists():

            raise FileNotFoundError(
                image_path
            )

        with image_path.open(
            "rb"
        ) as handle:

            files = {

                "file": (

                    image_path.name,

                    handle,

                    "image/jpeg",
                )
            }

            response = requests.post(
                self.base_url + "/predict",
                files=files,
                timeout=self.timeout,
            )

        response.raise_for_status()

        return response.json()

    def prototype_explanation(
        self,
        image_path,
    ):
        """Request real prototype similarity reasoning for one uploaded image."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        with image_path.open("rb") as handle:
            response = requests.post(
                self.base_url + "/visual-xai/prototype",
                files={
                    "image": (
                        image_path.name,
                        handle,
                        "image/jpeg",
                    )
                },
                timeout=self.timeout,
            )

        response.raise_for_status()
        return response.json()
