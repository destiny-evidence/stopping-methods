import logging
from contextlib import ContextDecorator
from time import sleep, perf_counter
from typing import Any

import httpx

logger = logging.getLogger('shared.openalex')


def revert_index(inverted_index: dict[str, list[int]] | None) -> str | None:
    if inverted_index is None:
        return None

    token: str
    position: int
    positions: list[int]

    abstract_length: int = len([1 for idxs in inverted_index.values() for _ in idxs])
    abstract: list[str] = [''] * abstract_length

    for token, positions in inverted_index.items():
        for position in positions:
            if position < abstract_length:
                abstract[position] = token

    return ' '.join(abstract)


class rate_limit(ContextDecorator):
    def __init__(self, min_time_ms: int = 100):
        self.min_time = min_time_ms / 1000

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.start
        self.readout = f'Time: {self.time:.3f} seconds'
        if self.time < self.min_time:
            logger.debug(f'Sleeping to keep rate limit: {self.min_time - self.time:.4f} seconds')
            sleep(self.min_time - self.time)


def fetch_works(params: dict[str, Any], fields: list[str] | None = None):
    if fields is None:
        fields = FIELDS_TO_FETCH
    cursor = '*'
    ids = 0
    page_i = 0
    while cursor is not None:
        page_i += 1
        with rate_limit(min_time_ms=100) as t:
            res = httpx.get(
                'https://api.openalex.org/works',
                params={
                           'cursor': cursor,
                           'per-page': 200,
                           'select': ','.join(fields)
                       } | params,
                # headers={'api_key': os.getenv('API_KEY')},
                timeout=None,
            )
            page = res.json()
            cursor = page['meta']['next_cursor']
            logger.info(f'Retrieved {ids:,}/{page['meta']['count']:,}; currently on page {page_i}')

            for res in page['results']:
                if res.get('abstract_inverted_index'):
                    res['abstract'] = revert_index(res['abstract_inverted_index'])
                    del res['abstract_inverted_index']

                yield res
                ids += 1

FIELDS_TO_FETCH = [
    'id',
    'doi',
    'title',
    'display_name',
    'publication_year',
    'publication_date',
    'ids',
    'language',
    'primary_location',
    'type',
    'type_crossref',
    'indexed_in',
    'open_access',
    'authorships',
    'institution_assertions',
    'countries_distinct_count',
    'institutions_distinct_count',
    'corresponding_author_ids',
    'corresponding_institution_ids',
    'apc_list',
    'apc_paid',
    'fwci',
    'has_fulltext',
    'fulltext_origin',
    # 'cited_by_count',
    # 'citation_normalized_percentile',
    # 'cited_by_percentile_year',
    # 'biblio',
    'is_retracted',
    'is_paratext',
    'primary_topic',
    # 'topics',
    'keywords',
    # 'concepts',
    # 'mesh',
    'locations_count',
    'locations',
    'best_oa_location',
    # 'sustainable_development_goals',
    'grants',
    'datasets',
    'versions',
    # 'referenced_works_count',
    # 'referenced_works',
    # 'related_works',
    'abstract_inverted_index',
    # 'cited_by_api_url',
    # 'counts_by_year',
    'updated_date',
    'created_date'
]
