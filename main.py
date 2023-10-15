import argparse
from concurrent.futures import ThreadPoolExecutor
import dotenv
import json
import pprint
import os
import requests
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
dotenv.load_dotenv()

from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import TokenTextSplitter
from langchain.document_loaders import JSONLoader
from langchain.vectorstores import Chroma


def cache_jira(project_key="AL"):
    """Download all Jira issues and cache them in a local file."""
    auth = requests.auth.HTTPBasicAuth(
        os.environ["JIRA_EMAIL"], os.environ["JIRA_API_TOKEN"]
    )

    def remap_issue(issue):
        mapped = {
            "key": issue.get("key", ""),
            "summary": issue["fields"].get("summary", ""),
            "creator": issue["fields"].get("creator", {}).get("displayName", ""),
            "assignee": issue["fields"]["assignee"]["displayName"]
            if issue["fields"].get("assignee")
            else "Unassigned",
            "status": issue["fields"].get("status", {}).get("name", ""),
            "created": issue["fields"].get("created", ""),
            "updated": issue["fields"].get("updated", ""),
            "related_issues": ", ".join(
                [
                    link["outwardIssue"]["key"]
                    for link in issue["fields"].get("issuelinks", [])
                    if link.get("outwardIssue")
                ]
            ),
        }

        url = f"https://{os.environ['JIRA_DOMAIN']}/rest/api/3/issue/{issue['key']}/comment"
        headers = {"Accept": "application/json"}
        response = requests.get(url, headers=headers, auth=auth)
        response.raise_for_status()

        if not issue["fields"]["description"] and not response.json()["comments"]:
            return None

        mapped["text"] = {
            "description": issue["fields"]["description"],
            "comments": [
                {"body": comment["body"], "author": comment["author"]["displayName"]}
                for comment in response.json()["comments"]
            ],
        }
        print(f"Downloaded {issue['key']} from Jira...")
        return mapped

    url = f"https://{os.environ['JIRA_DOMAIN']}/rest/api/3/search"
    params = {"jql": f"project = {project_key}", "maxResults": 1000, "startAt": 0}
    headers = {"Accept": "application/json"}
    total_issues = 0
    issues = []
    next_issues = True
    while next_issues:
        response = requests.get(url, params=params, headers=headers, auth=auth)
        response.raise_for_status()
        data = response.json()
        if not data["issues"]:
            break
        total_issues = data["total"]
        with ThreadPoolExecutor(max_workers=3) as executor:
            remapped_issues = [
                remapped_issue
                for remapped_issue in list(executor.map(remap_issue, data["issues"]))
                if remapped_issue
            ]
            issues.extend(remapped_issues)
        params["startAt"] += len(data["issues"])
        next_issues = params["startAt"] <= total_issues
        print(f"Downloaded {len(issues)} issues from Jira...")

    with open("jira.json", "w") as f:
        json.dump(issues, f)


def build_vector_store():
    def metadata_func(record, metadata):
        for key, value in record.items():
            if key == "text":
                continue
            metadata[key] = value
        return metadata

    loader = JSONLoader(
        file_path="jira.json",
        jq_schema=".[]",
        content_key="text",
        metadata_func=metadata_func,
        text_content=False,
    )
    data = loader.load()
    token_splitter = TokenTextSplitter("gpt2")
    texts = token_splitter.split_documents(data)
    docsearch = Chroma.from_documents(texts, OpenAIEmbeddings(), persist_directory="jira_chroma")
    docsearch.persist()
    # docsearch.save_local("jira")


def ask_question(question):
    # docsearch = Chroma.load_local("jira", OpenAIEmbeddings())
    docsearch = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory="jira_chroma")
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(model="gpt-3.5-turbo-instruct", temperature=0),
        chain_type="map_reduce",
        retriever=docsearch.as_retriever(search_type="mmr"),
        verbose=True,
    )
    print(qa.run(question))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("command", choices=["init", "ask", "chat"])
    parser.add_argument("question", nargs="?")
    args = parser.parse_args()

    if args.command == "init":
        # cache_jira()
        build_vector_store()
    elif args.command == "chat":
        pass
        # chat()
    elif args.command == "ask":
        ask_question(args.question)


if __name__ == "__main__":
    main()
