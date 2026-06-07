"""HTTP session helpers with retry, caching, and rate-limiting behavior."""

from urllib3.util.retry import Retry

from requests import Session
from requests.adapters import HTTPAdapter
from requests_cache import CacheMixin
from requests_ratelimiter import LimiterMixin

__all__ = ()


class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
    """
    Session class with caching and rate-limiting behavior. Accepts arguments
    for both LimiterSession and CachedSession.
    """

    def __init__(
            self,
            *args,
            max_retries=3,
            backoff_factor=0.25,
            status_forcelist=(408, 413, 500, 503, 504),
            **kwargs,
            ):
        super().__init__(*args, **kwargs)
        retry_adapter = HTTPAdapter(max_retries=Retry(
            total=max_retries,
            # Force retries only on these statuses.
            status_forcelist=status_forcelist,
            backoff_factor=backoff_factor,
        ))
        self.mount('http://', retry_adapter)
        self.mount('https://', retry_adapter)
