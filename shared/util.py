import logging
import typing
from time import sleep, perf_counter
from typing import Sequence, Generator, TypeVar
from xml.etree.ElementTree import Element

import httpx
from httpx import Client, codes, Response, URL
from httpx._client import UseClientDefault, USE_CLIENT_DEFAULT
from httpx._types import (
    RequestContent, RequestData, RequestFiles, QueryParamTypes, HeaderTypes, CookieTypes,
    AuthTypes, TimeoutTypes, RequestExtensions
)

T = TypeVar('T')


def batched(lst: Sequence[T] | Generator[T, None, None], batch_size: int) -> Generator[list[T], None, None]:
    batch = []
    for li in lst:
        batch.append(li)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch


class RequestClient(Client):
    def __init__(self, *,
                 max_req_per_sec: int = 5, max_retries: int = 5, timeout_rate: float = 5.,
                 retry_on_status: list[int] | None = None,
                 **kwargs):
        super().__init__(**kwargs)

        self.max_req_per_sec = max_req_per_sec
        self.time_per_request = 1 / max_req_per_sec
        self.max_retries = max_retries
        self.timeout_rate = timeout_rate
        self.last_request: float | None = None
        self.retry_on_status = retry_on_status or [
            codes.INTERNAL_SERVER_ERROR,  # 500
            codes.BAD_GATEWAY,  # 502
            codes.SERVICE_UNAVAILABLE,  # 503
            codes.GATEWAY_TIMEOUT,  # 504
        ]
        self.kwargs = kwargs
        self.callbacks = {}

    def switch_proxy(self, proxy: str | None = None):
        if proxy != self.kwargs.get('proxy'):
            client = self.__class__(**{
                **self.kwargs,
                'proxy': proxy,
                'max_req_per_sec': self.max_req_per_sec,
                'max_retries': self.max_retries,
                'timeout_rate': self.timeout_rate,
                'retry_on_status': self.retry_on_status})
            self.__dict__.update(client.__dict__)

    def on(self, status: int, func: typing.Callable[[Response], dict[str, typing.Any]]):
        self.callbacks[status] = func

    @typing.override
    def request(
            self,
            method: str,
            url: URL | str,
            *,
            content: RequestContent | None = None,
            data: RequestData | None = None,
            files: RequestFiles | None = None,
            json: typing.Any | None = None,
            params: QueryParamTypes | None = None,
            headers: HeaderTypes | None = None,
            cookies: CookieTypes | None = None,
            auth: AuthTypes | UseClientDefault | None = USE_CLIENT_DEFAULT,
            follow_redirects: bool | UseClientDefault = USE_CLIENT_DEFAULT,
            timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
            extensions: RequestExtensions | None = None,
    ) -> Response:
        for retry in range(self.max_retries):
            # Check if we need to wait before the next request so we are staying below the rate limit
            time = perf_counter() - (self.last_request or 0)
            if time < self.time_per_request:
                logging.debug(f'Sleeping to keep rate limit: {self.time_per_request - time:.4f} seconds')
                sleep(self.time_per_request - time)

            # Log latest request
            self.last_request = perf_counter()

            response = super().request(method=method, url=url, content=content, data=data, files=files, json=json,
                                       params=params, headers=headers, cookies=cookies, auth=auth,
                                       follow_redirects=follow_redirects, timeout=timeout, extensions=extensions)

            try:
                response.raise_for_status()

                # reset counters after successful request
                self.time_per_request = 1 / self.max_req_per_sec

                return response

            except httpx.HTTPError as e:
                if e.response.status_code in self.callbacks:
                    logging.debug(f'Found status handler for {e.response.status_code}')
                    update = self.callbacks[e.response.status_code](e.response)
                    if update and update.get('content'):
                        content = update.get('content')
                    if update and update.get('data'):
                        data = update.get('data')
                    if update and update.get('json'):
                        json.update(update.get('json', {}))
                    if update and update.get('params'):
                        params.update(update.get('params', {}))
                    if update and update.get('headers'):
                        headers.update(update.get('headers', {}))

                # if this error is not on the list, pass on error right away; otherwise log and retry
                elif e.response.status_code not in self.retry_on_status and len(self.retry_on_status) > 0:
                    raise e

                else:
                    logging.warning(f'Retry {retry} after failing to retrieve from {url}: {e}')
                    logging.warning(e.response.text)
                    logging.exception(e)

                    # grow the sleep time between requests
                    self.time_per_request = (self.time_per_request + 1) * self.timeout_rate
        else:
            raise RuntimeError('Maximum number of retries reached')


def xml2dict(element: Element) -> dict[str, typing.Any]:
    base = {}
    for child in element:
        if child.tag not in base:
            base[child.tag] = []
        base[child.tag].append(xml2dict(child))
    base |= {f'@{attr}': val for attr, val in element.attrib.items()}
    if element.text and len(element.text.strip()) > 0:
        base['_text'] = element.text.strip()
    return base
