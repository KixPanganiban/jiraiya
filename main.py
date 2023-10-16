import argparse
import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import dotenv
import requests

warnings.filterwarnings("ignore", category=UserWarning)
dotenv.load_dotenv()

from langchain.document_loaders import JSONLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAIChat
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import Chroma


def cache_jira(project_key="AL"):
    """Download all Jira issues and cache them in a local file."""
    auth = requests.auth.HTTPBasicAuth(
        os.environ["JIRA_EMAIL"], os.environ["JIRA_API_TOKEN"]
    )

    def flatten_jira_text(text):
        """Flatten Jira text into a single string."""
        result = ""
        if "type" in text:
            if text["type"] == "text":
                result += text["text"] + " "
            elif text["type"] == "listItem":
                result += "- "
            elif text["type"] == "inlineCard":
                result += text["attrs"]["url"].split("/")[-1] + " "
        if "content" in text:
            for item in text["content"]:
                result += flatten_jira_text(item)
            if text["type"] == "paragraph":
                result += "\n"
        return result

    def extract_text_from_description(description):
        """Extract text from a Jira issue description."""
        if not description:
            return ""
        return flatten_jira_text(description)

    def extract_text_from_comments(comments):
        """Extract text from Jira issue comments."""
        result = ""
        for comment in comments:
            result += (
                comment["author"]["displayName"]
                + " said: "
                + flatten_jira_text(comment["body"])
                + "\n"
            )
        return result

    def remap_issue(issue):
        """Remap a Jira issue into a simpler more compact format."""
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

        mapped[
            "text"
        ] = """
        Summary: {summary}
        --
        Key: {key}
        --
        Description:
        {description}
        
        Comments:
        {comments}
        """.format(
            key=issue["key"],
            description=extract_text_from_description(issue["fields"]["description"]),
            comments=extract_text_from_comments(response.json()["comments"]),
        )
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
    """Build a vector store from the Jira issues."""

    def metadata_func(record, metadata):
        """Save the mapped Jira issue as metadata, except for the text."""
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
    docsearch = Chroma.from_documents(
        texts, OpenAIEmbeddings(), persist_directory="jira_chroma"
    )
    docsearch.persist()



def ask_question(question):
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma(embedding_function=embeddings, persist_directory="jira_chroma")

    template = """
    You are a helpful knowledge assistant that provides answers to questions using issues from Jira.
    If you don't know that answer, say so. Do not make up an answer.
    Important: Do not ask the user to consult people or documentation. Do not create hyperlinks.
    Expand on your answers as much as possible.
    At the end of your answer, always cite your sources using the Issue key like so: "Sources: [key 1, key 2, ...]"
    Use these Jira issues as context:
    
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    model = OpenAIChat(model="gpt-3.5-turbo-16k", temperature=0)
    chain = (
        {
            "context": docsearch.as_retriever(search_type="mmr", lambda_mult=0),
            "question": RunnablePassthrough(),
        }
        | prompt
        | model
        | StrOutputParser()
    )

    return chain.invoke(question)


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
        print(ask_question(args.question))


if __name__ == "__main__":
    main()
