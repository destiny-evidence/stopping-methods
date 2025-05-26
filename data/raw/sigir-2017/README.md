# SIGIR2017 SysRev Query Collection
This is taken from:
https://github.com/ielab/SIGIR2017-SysRev-Collection/tree/master

This directory contains the collection of queries for the paper "A Test Collection for Evaluating Retrieval of Studies for Inclusion in Systematic Reviews".

The following files have been made available:

 - `systematic_reviews.json`: A list of urls pointing to the systematic reviews used in this collection.
 - `citations.json`: A list of citations that correspond to the systematic reviews. The `document_id` refers to the `id` in `systematic_reviews.json`. The `included` field is true if the study was included in the review.

If you use this collection in your own work, please cite it as:

```
@inproceedings{scells2017collection,
	Author = {Scells, Harrisen and Zuccon, Guido and Koopman, Bevan and Deacon, Anthony and Geva, Shlomo and Azzopardi, Leif},
	Booktitle = {Proceedings of the 40th international ACM SIGIR conference on Research and development in Information Retrieval},
	Organization = {ACM},
	Title = {A Test Collection for Evaluating Retrieval of Studies for Inclusion in Systematic Reviews},
	Year = {2017}
}
```
