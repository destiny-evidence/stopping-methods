from itertools import count 
from copy import deepcopy
import os
#from generic import Record
import pandas as pd
import rispy

### Function to read 2 RIS files, assuming two such files represent a review, one file contains includes, the other excludes
### When the RIS file contains PMID and only PMIDs in a given field, we can use the PMIDField paramater to grab these values
### Same applies to OpenAlex IDs

print(os.getcwd())

def RIS_ineclue_And_Exclude_to_TSV(IncludesFile, ExcludesFile, PMIDField = "", OpenAlexField = "", outputFile = "out.csv"):
    
    ###Function to read one line of a reference, as imported via rispy
    def ProcessLine(entry, data, counter, isInclude = 0, PMIDField = "", OpenAlexField = ""):
        data['id'][counter] = counter
        if 'abstract' in entry: data['abstract'][counter] =  entry['abstract']
        else: data['abstract'][counter] = ""
        
        if 'title' in entry: 
            data['title'][counter] =  entry['title']
            #print (entry['title'])
        elif 'primary_title' in entry:
            data['title'][counter] =  entry['primary_title']
            #print (entry['primary_title'])            
        else: data['title'][counter] = ""
        
        data['label_abs'][counter] = isInclude
        
        if 'keywords' in entry: 
            outp="; ".join(entry['keywords'])
            data['keywords'][counter] = outp
        else: data['keywords'][counter] = ""
        
        if 'doi' in entry: 
            data['doi'][counter] =  entry['doi']
        else: data['doi'][counter] = ""

        if PMIDField != "":
            if "pmid" in entry: data['pubmed_id'][counter] =  entry["pmid"]
            else: data['pubmed_id'][counter] = float('NaN')
        else: data['pubmed_id'][counter] = float('NaN')
        
        if OpenAlexField != "":
            if "OAid" in entry: data['openalex_id'][counter] =  entry["OAid"]
            else: data['openalex_id'][counter] = float('NaN')
        else: data['openalex_id'][counter] = float('NaN')
        
        data['label_ft'][counter] = float('NaN')
        
    dataDict = { "id": {}, "title": {}, "abstract": {}, "label_abs": {}, "pubmed_id": {}, "openalex_id": {}, "doi": {}, "keywords": {}, "label_ft": {}}
    counter = 0
    mapping = deepcopy(rispy.TAG_KEY_MAPPING)
    
    if PMIDField != "":        
        mapping[PMIDField] = "pmid"
        
    if OpenAlexField != "":        
        mapping[OpenAlexField] = "OAid"
        
    with open(IncludesFile, 'r', encoding="UTF-8") as bibliography_file:
        entries = rispy.load(bibliography_file, mapping=mapping)

    for entry in entries:
        ProcessLine(entry, dataDict, counter, 1, PMIDField = PMIDField)
        #print('---' + str(counter) + '---')
        counter += 1

    with open(ExcludesFile, 'r', encoding="UTF-8") as bibliography_file:
        entries = rispy.load(bibliography_file, mapping=mapping)
    for entry in entries:
        ProcessLine(entry, dataDict, counter, 0)
        #print('---' + str(counter) + '---')
        counter += 1
        
    df = pd.DataFrame(dataDict)
    #print(df["Title"])
    
    # with open("filename.jsonl", "w") as fp:
    #     for _, row in df.iterrows():
    #         fp.write(Record(title=row["title"]).dump_json()+"\n")
        
    
    #df.to_csv(outputFile, index=False, sep = "\t")
    df.to_csv(outputFile, index=False)

  
IncFilepath = 'data/raw/Digital interventions for social isolation and loneliness EGM 28165 Includes.txt'
ExcFilepath = 'data/raw/Digital interventions for social isolation and loneliness EGM 28165 Excludes.txt'


#here we show how to convert a pair of RIS files, while fishing the PMID values from the U1 RIS field
#note: for this pair of files, U1 does contain ID values, but they aren't real PMIDs!
#RIS_ineclue_And_Exclude_to_TSV(IncFilepath, ExcFilepath, outputFile= "../data/review-1253.tsv", PMIDField = "U1")

#without (fake) PMIDs...
#RIS_ineclue_And_Exclude_to_TSV("RIS Files/cancer_INCLUDES.ris", "RIS Files/cancer_EXCLUDES.ris", outputFile= "cancer.tsv")

RIS_ineclue_And_Exclude_to_TSV(IncFilepath, ExcFilepath, outputFile= "data/raw/EGM 28165.csv")