import argparse
import dotenv
import json
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
dotenv.load_dotenv()


from llama_index import Document, download_loader, VectorStoreIndex, load_index_from_storage, StorageContext, ServiceContext
from llama_index.llms import OpenAI


def build_vector_store():
    """Build the vector store by reading the data directory and indexing all issues."""
    JiraReader = download_loader("JiraReader")
    reader = JiraReader(
        email=os.environ["JIRA_EMAIL"],
        api_token=os.environ["JIRA_API_TOKEN"],
        server_url=f"{os.environ['JIRA_DOMAIN']}"
    )
    documents = reader.load_data(query="project = AL")
    vector_store = VectorStoreIndex.from_documents(documents)
    vector_store.set_index_id("jira")
    vector_store.storage_context.persist("./vector_store")
    print("Saved vector store")


def ask_question(question):
    storage_context = StorageContext.from_defaults(persist_dir="./vector_store")
    index = load_index_from_storage(storage_context, index_id="jira")
    query_engine = index.as_query_engine()
    response = query_engine.query(question)
    print(response)

def chat():
    storage_context = StorageContext.from_defaults(persist_dir="./vector_store")
    service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo-0613"))
    index = load_index_from_storage(storage_context, index_id="jira", service_context=service_context)
    chat_engine = index.as_chat_engine(chat_mode="context", system_prompt="You are an assistant who answers questions based on the user's JIRA data. Limit your answers to only the provided JIRA context, and expand details about tickets as applicable.")
    user_input = input("You (type exit to end): ")
    while user_input != "exit":
        response = chat_engine.chat(user_input)
        print(response)
        user_input = input("\nYou (type exit to end): ")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("command", choices=["init", "ask", "chat"])
    parser.add_argument("question", nargs="?")
    args = parser.parse_args()

    if args.command == "init":
        build_vector_store()
    elif args.command == "chat":
        chat()
    elif args.command == "ask":
        ask_question(args.question)
    

if __name__ == "__main__":
    main()
