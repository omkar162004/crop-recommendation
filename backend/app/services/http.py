import requests
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app import config
from app.services.errors import ServiceUnavailable


def _retryable(exc: BaseException) -> bool:
    # retry network errors, timeouts, 5xx and 429; never retry other 4xx (our request is wrong)
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        code = exc.response.status_code
        return code >= 500 or code == 429
    return isinstance(exc, requests.RequestException)


@retry(
    stop=stop_after_attempt(config.RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=2, min=3, max=20),
    retry=retry_if_exception(_retryable),
    reraise=True,
)
def _get(url, params, timeout):
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def get_json(url, params, timeout):
    """GET with timeout + retries + backoff. Raises ServiceUnavailable if it still fails."""
    try:
        return _get(url, params, timeout)
    except (requests.RequestException, ValueError) as exc:
        raise ServiceUnavailable(f"{url} failed: {exc}") from exc