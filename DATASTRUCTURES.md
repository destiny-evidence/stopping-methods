The idea is to have a single metadataset (or index file) for all review datasets which contains the following information for each review. 
For now, we are creating metadatasets for collections of reviews as we go on (e.g. Cohen, CLEF data) 
We will then combine these metadatasets into a single metadataset. 

To avoid issues with merging, each individual metadataset should contain the following fields: 

* name | Type: String. Description: File name of the dataset. Example: "heart_dataset.csv".
* source | Type: String. Description: Where the dataset came from. If not from a 'collection' of datasets (e.g., CLEF) give the title of the paper from which they are derived. Examples: "CLEF datasets", "FASTREAD datasets", "CAMARADES datasets", "Screening Smarter, Not Harder: A Comparative Analysis of Machine Learning Screening Algorithms and Heuristic Stopping Criteria for Systematic Reviews in Educational Research"
* evidence_synthesis_type | Type: String. Description: The type of review conducted. Must be one of the following predefined options:
	•	"Systematic Review"
	•	"Evidence Map"
	•	"Scoping Review"
	•	"Rapid Review"
	•	"Horizon Scan"
	•	"Other"
* domain | Type: String. Description: Description of the topic and type of the review. Example: "Clinical", "Preclinical", "Computer Science", "Education", "Toxicology".
* review_type: Type: String. Description: Description of the review type. Example: "Prognostic", "Diagnostic Test Accuracy", "Intervention". Missing values: Can be initially be left empty as ("") if information not obvious. 
* eligibility_criteria | Type: String. Description: A short description of the inclusion criteria for the dataset. Missing values: Can be initially be left empty as (""). 
* n_total | Type: Integer. Description: Total number of records/documents in the review dataset. Missing values: Not allowed.
* n_includes_ab | Type: Integer. Description: Total number of included records/documents in the review dataset based on abstract screening. Missing values: Not allowed.
* n_includes_ft | Type: Integer. Description: Total number of included records/documents in the review dataset based on full-text screening. Missing values: NA. 
* include_ratio_abs | Type: Float. Description: Proportion of included records based on abstract screening, calculated as: n_includes_abs/n_total. Missing values: Not allowed. 
* include_ratio_ft | Type: Float. Description: Proportion of included records based on full-text screening, calculated as: calculated as: n_includes_abs/n_total. Missing values: NA. 

See shared/dataset.py for datastructure for simulation dataset fields (copied below for ease of reference) 

Required variables: id, title, abstract, pubmed_id, openalex_id, doi, keywords, year, label_abs, label_ft: 

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
    # Publication year
    year: int | None = None

    # Label indicating inclusion/exclusion of the reference at title-abstract level screening. Values: {0, 1}.
    # Missing Values: Not allowed (must always have a value).
    label_abs: int
    # Label indicating some classification (e.g., full-text relevance). Values: {0, 1}. Missing Values: None.
    label_ft: int | None = None
