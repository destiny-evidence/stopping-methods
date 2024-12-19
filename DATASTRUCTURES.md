The idea is to have a single metadataset (or index file) for all review datasets which contains the following information for each review. 
For now, we are creating metadatasets for collections of reviews as we go on (e.g. Cohen, CLEF data) 
We will then combine these metadatasets into a single metadataset. 

To avoid issues with merging, each individual metadataset should contain the following fields: 

* name | Type: String. Description: File name of the dataset. Example: "heart_dataset.csv".
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

See generic_json.py for datastructure for simulation dataset fields
