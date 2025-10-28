# Read Me

Execute below commands for:

### Create Virtual Env

```Shell
# deprecated pip and migrated to use uv
#python3 -m venv .venv
vu venv
```

### Activate Virtual Env

```Shell
source .venv/bin/activate
```

#### Install requirements

```Shell
# deprecated pip and migrated to use uv
#python3 -m pip install -r requirements.txt
uv add -r requirements.txt
uv sync
```

**NOTE:** Please add `.env` file and add your own OpenAI API Key "OPENAI_API_KEY=\<Key Here\>"

#### Run UI Agent

```Shell
streamlit run demo_app.py
```

#### Run CLI Agent (if you have it!)

```Shell
python3 myagent_cli.py
```

## Source Structure

Data Folder `demo_data` contains all the data files read to embed/tokenize the data and store into a Vector DB

## Vector Database

In this small demo/poc, I am leveraging [Chroma DB](https://www.trychroma.com/).
Chroma DB is a free, simple to use open-source vector/embedding database.

## Model

Use the [OpenAI Models](https://platform.openai.com/docs/models) you're interested, obviously considering the cost and limitations.

## Embedding Model

Here are list of [OpenAI Embedding Models](https://platform.openai.com/docs/guides/embeddings#embedding-models) model that generate embeddings. An embedding is a vector (list) of floating point numbers. The distance between two vectors measures their relatedness. Small distances suggest high relatedness and large distances suggest low relatedness.
