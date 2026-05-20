# Email spam classify
## Analysis problems with bussiness and technical req
1. Problems
- Teacher: Given a dataset of emails, determine whether each email is "spam" or "not spam." - SO large.
    - so i think, we have to do with much source of data: *LKML*, *freeBDS*, *Kangle*, *...*. 
    because! model will get general to classify spam email.
    - we have to crawl all data and storage into MongoDB: *ELT*
    - Finally, we should merge it in a big tablur to prepare data processing. 


- Data Flow: Crawl data from many source -> ELT -> EDA -> process and merge -> EDA, visualization again with new data 
### Notice: In EDA and visualization always eval and analysis data. IMPORTANT!
## Tools
- Data crawl: Scapry
- ELT: MongoDB in local host is database.
- Visualization: Matplotlib

## code requirements
- dont long and complex
- dont comment
- create folder and file better

## Input and OutPut
- Input Dataflow: Source is address of web.
- Output dataflow: A .csv file have all data to train model
- other outputs: plots EDA ... ect

## model
- Navies Bayesian

## AIM
- model can classify general case.