from pydantic import BaseModel, ConfigDict
from shared.collection import AbstractCollection
from shared.dataset import Dataset


class Record(BaseModel):
    model_config = ConfigDict(extra='allow')

    # A unique identifier for each reference within the dataset. Example: 1, 2, 3, 4, 5. Missing Values: Not allowed
    id: int
    # Title of the reference. Missing Values: Represented as None.
    title: str | None = None
    # Abstract of the reference. Missing Values: Represented as  None.
    abstract: str | None = None
    # PubMed ID of the reference (if available). Missing Values: Represented as None.
    pubmed_id: str | None = None
    # OpenAlex identifier of the reference (if available). Missing values: Represented as None.
    openalex_id: str | None = None
    # Digital Object Identifier for the reference (if available). Missing Values: Allowed.
    doi: str | None = None
    # Keywords for the reference, separated by semi-colons. Example: "Keyword1; Keyword2; Keyword3". Missing Values: Represented as None.
    keywords: str | None = None

    # Label indicating inclusion/exclusion of the reference at title-abstract level screening. Values: {0, 1}.
    # Missing Values: Not allowed (must always have a value).
    label_abs: int
    # Label indicating some classification (e.g., full-text relevance). Values: {0, 1}. Missing Values: None.
    label_ft: int | None = None


class GenericCollection(AbstractCollection):
    BASE: str = 'generic-jsonl'

    def fetch_collection(self):
        pass  # we assume the raw files were put in the directory already

    def prepare_datasets(self):
        pass

    def generate_datasets(self):
        for file in self.raw_folder.glob('*.jsonl'):
            with open(file, 'r') as f:
                records = [Record.model_validate_json(line) for line in f]
                yield Dataset(key=f'generic-{file.stem}',
                              labels=[rec.label_abs for rec in records],
                              texts=[(rec.title or '') + ' ' + (rec.abstract or '') for rec in records])
