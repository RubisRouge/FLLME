try:
    from google.auth import default as get_google_default_credentials
    from google.auth.credentials import Credentials as GoogleCredentials
    from google.auth.transport.requests import Request as GoogleRequest
except ImportError:
    raise ImportError(
        "google-auth is required for Google ADC authentication. "
        "Install it with: pip install google-auth"
    )

from .header import HTTPHeaderAuthResolver


class GoogleADCAuthResolver(HTTPHeaderAuthResolver):
    def __init__(
        self,
    ) -> None:
        self._credentials: GoogleCredentials | None = None
        super().__init__(self.fetch_bearer)

    def fetch_bearer(self) -> str:
        if self._credentials is None:
            self._credentials, _ = get_google_default_credentials()
        if not self._credentials.valid:
            self._credentials.refresh(GoogleRequest())
        return f"Bearer {self._credentials.token}"
