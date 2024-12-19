# Structure of metadataset
The idea is to have a single metadataset (or index file) for all review datasets which contains the following information for each review. 
For now, we are creating metadatasets for collections of reviews as we go on (e.g. Cohen, CLEF data) 
We will then combine these metadatasets into a single metadataset. 

To avoid issues with merging, each individual metadataset should contain the following fields: 

* Name | Type: String. Description: File name of the dataset. Example: "heart_dataset.csv".
* Evidence_synthesis_type | Type: String. Description: The type of review conducted. Must be one of the following predefined options:
	•	"Systematic Review"
	•	"Evidence Map"
	•	"Scoping Review"
	•	"Rapid Review"
	•	"Horizon Scan"
	•	"Other"
* Domain | Type: String. Description: Description of the topic and type of the review. Example: "Clinical", "Preclinical", "Computer Science", "Education", "Toxicology".
* Review_type: Type: String. Description: Description of the review type. Example: "Prognostic", "Diagnostic Test Accuracy", "Intervention". Missing values: Can be initially be left empty as ("") if information not obvious. 
* Eligibility_criteria | Type: String. Description: A short description of the inclusion criteria for the dataset. Missing values: Can be initially be left empty as (""). 
* N_total | Type: Integer. Description: Total number of records/documents in the review dataset. Missing values: Not allowed.
* N_includes | Type: Integer. Description: Total number of included records/documents in the review dataset. Missing values: Not allowed.
* Include_ratio | Type: Float. Description: Proportion of included records, calculated as: N_includes/N_total. Missing values: Not allowed. 

# Structure of simulation datasets
Minimum Variables
* ID | Type: Integer. Description: A unique identifier for each reference within the dataset. Example: 1, 2, 3, 4, 5. Missing Values: Not allowed
* Title | Type: String. Description: Title of the reference. Missing Values: Represented as an empty string (""). 
* Abstract | Type: String. Description: Abstract of the reference. Missing Values: Represented as an empty string ("").
* TIAB_Label | Type: Integer. Description: Label indicating inclusion/exclusion of the reference at title-abstract level screening. Values: {0, 1}. Missing Values: Not allowed (must always have a value).

Optional Variables
* PMID | Type: Integer. Description: PubMed ID of the reference (if available). Missing Values: Represented as a 0. 
* OA | Type: String. Description: OpenAlex identifier of the reference (if available). Missing values: Represented as an empty string ("").
* DOI | Type: String. Description: Digital Object Identifier for the reference (if available). Missing Values: Represented as an empty string (""). 
* Keywords | Type: String. Description: Keywords for the reference, separated by semi-colons. Example: "Keyword1; Keyword2; Keyword3". Missing Values: Represented as an empty string ("").
* FT_Label | Type: Integer. Description: Label indicating some classification (e.g., full-text relevance). Values: {0, 1}. Missing Values: Not allowed.
