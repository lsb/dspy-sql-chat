TODO:

[x] copy https://github.com/martenlienen/icml-neurips-iclr-dataset/blob/master/papers.csv into the repository
[x] use Pandas and SQLite to read this csv into an in-memory database (call the table `paper_authorships` )and to be able to query against the data (for example: `select count(distinct year) from paper_authorships` should yield whatever number of years we have in the database)
[x] use dspy.Predict() (this is complicated, use the docs at https://dspy.ai) to use the `qwen3:4b-instruct-2507-q4_K_M` model in Ollama to take a human language description of a data query and to translate that to SQL (for example: `how many years do we have in the database` should yield like the sql in the last todo, and the sql should run and return the same value as `select count(distinct year) from paper_authorships`). make sure that the maximum new tokens that we allow from the LLM is enough to include all of the thinking that the model does. running this model locally takes more than a few minutes; we might want to change the model in the future.
[x] generate 200 pairs of "question to ask about this paper_authorships table" and "sql that represents that query". use the json-lines format
[ ] following https://dspy.ai/tutorials/gepa_aime/ , evaluate the chat to sql function from before on this dataset, optimize the DSPy class, and print everything about the optimized class (we may not be using chain of thought like in the tutorial but everything else applies)
[ ] use Gradio to make a chat app to view this


General: use uv, keep git up to date, keep pytest up to date, only do one todo at a time