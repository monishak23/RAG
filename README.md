# RAG
Retrieval Augmented Generation

## Concepts

Phase 1: Indexing (document is loaded)
 - Chunking: Split the entire document into pieces with overlap.Overlap ensures both chunks contain that sentence. (preferable chunk size is 500 and overlap is 100 - focused chunk - goldilocks principle )
 - Emdeddings: Convert the chunks into list of numbers( vectors)
 - Vectore database: Store the embedding result into the vector database.

Phase 2: Quering (on every user question)
  - Embed the question: Generate vector for the question text.
  - Similarity search: Use similarity search to get the 3 chunks that are closest to the result.
  - Feed to model: Send the similarity search result to model as the content.
