import argparse
import dotenv
import json
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
dotenv.load_dotenv()


from llama_index import Document, VectorStoreIndex, load_index_from_storage, StorageContext, ServiceContext
from llama_index.llms import OpenAI
from jira import get_all_issues_paginated, JIRA_DOMAIN


def build_jira_index():
    """Build the JIRA index by dumping all issues into JSON files in the data directory."""
    for issue in get_all_issues_paginated(limit=None):
        issue_key = issue["key"]
        os.makedirs("data", exist_ok=True)
        with open(f"data/{issue_key}.json", "w") as f:
            json.dump(issue, f, indent=2)
            print(f"Wrote {issue_key}.json")

def build_vector_store():
    """Build the vector store by reading the data directory and indexing all issues."""
    documents = []
    for filename in os.listdir("data"):
        with open(f"data/{filename}") as f:
            issue = json.load(f)
            issue_key = issue["key"]
            try:
                document = Document(
                    id=issue_key,
                    metadata={
                        "ticket_id": issue_key,
                        "author": issue["fields"].get("creator", {}).get("displayName", "Unknown"),
                        "assignee": issue["fields"].get("assignee", {}).get("displayName", "Unknown"),
                        "created_on": issue["fields"].get("created", "Unknown"),
                        "updated_on": issue["fields"].get("updated", "Unknown"),
                        "status": issue["fields"].get("status", {}).get("name", "Unknown"),
                        "jira_url": f"https://{JIRA_DOMAIN}/browse/{issue_key}"
                    },
                    text=f"""
                        Issue Summary: {issue["fields"]["summary"]}
                        Description: {issue["fields"]["description"]}
                    """
                )
                documents.append(document)
            except (KeyError, AttributeError):
                print(f"Failed to build document for {issue_key}")
                continue
            print(f"Built document for {issue_key}")
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
        build_jira_index()
        build_vector_store()
    elif args.command == "chat":
        chat()
    elif args.command == "ask":
        ask_question(args.question)
    

if __name__ == "__main__":
    main()
