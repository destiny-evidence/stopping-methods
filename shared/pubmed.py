import logging
from typing import Any, Generator
from xml.etree.ElementTree import Element, fromstring as parse_xml

from shared.config import settings
from shared.util import RequestClient, batched, xml2dict

logger = logging.getLogger('wrapper-pubmed')
PAGE_SIZE = 10


def get_title(article: Element) -> str | None:
    hits = article.findall('.//ArticleTitle')
    if len(hits) > 0:
        return ' '.join(hits[0].itertext())
    return None


def get_abstract(article: Element) -> str | None:
    hits = article.findall('.//Abstract')
    if len(hits) > 0:
        return '\n\n'.join(hits[0].itertext())
    return None


def get_doi(article: Element) -> str | None:
    hits = article.findall('.//ArticleId[@IdType="doi"]')
    if len(hits) > 0:
        return hits[0].text
    return None


def get_id(article: Element) -> str | None:
    hits = article.findall('.//PMID')
    if len(hits) > 0:
        return hits[0].text
    return None


def fetch(ids: list[dict[str, str]]) -> Generator[dict[str, Any], None, None]:
    if settings.PUBMED_API_KEY is None:
        raise RuntimeError('`PUBMED_API_KEY` not set')

    parts = []
    for reference in ids:
        if reference.get('pubmed_id'):
            parts.append(f'{reference['pubmed_id']}[PMID]')
        if reference.get('doi'):
            parts.append(f'"{reference['doi']}"[DOI]')

    if len(parts) == 0:
        raise ValueError('Found no scopus ids or DOIs to query pubed')

    # direct lookup via
    # https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?api_key=KEY&db=pubmed&id=17975326
    # can be comma separated!
    # DOCS: https://www.ncbi.nlm.nih.gov/books/NBK25497/

    n_records = 0
    with RequestClient(timeout=120, max_req_per_sec=3) as request_client:
        for n_pages, parts_batch in enumerate(batched(parts, batch_size=PAGE_SIZE)):
            logger.info(f'Fetching search context (page {n_pages})...')
            search_page = request_client.get(
                'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi',
                params={
                    'api_key': settings.PUBMED_API_KEY,
                    'db': 'pubmed',
                    'term': ' OR '.join(parts_batch),
                    'usehistory': 'y',
                },
            )
            tree = parse_xml(search_page.text)
            web_env = tree.find('WebEnv').text
            query_key = tree.find('QueryKey').text

            result_page = request_client.get(
                'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi',
                params={
                    'api_key': settings.PUBMED_API_KEY,
                    'db': 'pubmed',
                    'WebEnv': web_env,
                    'query_key': query_key,
                },
            )

            tree = parse_xml(result_page.text)
            for article in tree.findall('PubmedArticle'):
                entry = xml2dict(article)
                n_records += 1
                yield {
                    'title': get_title(article),
                    'abstract': get_abstract(article),
                    'doi': get_doi(article),
                    'pubmed_id': get_id(article),
                    'raw': entry,
                }
            logger.debug(f'Found {n_records:,} records after processing page {n_pages}')


if __name__ == '__main__':
    for ri, record in enumerate(fetch(
            ids=[
                {'pubmed_id': '17975327'},
                {'doi': '10.1046/j.1464-410x.1997.02667.x'},
            ])):
        print(record)
        if ri > 100:
            break
    print('Force stopped')
