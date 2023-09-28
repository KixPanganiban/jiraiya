import os

import requests
from requests.auth import HTTPBasicAuth


JIRA_EMAIL = os.environ["JIRA_EMAIL"]
JIRA_API_TOKEN = os.environ["JIRA_API_TOKEN"]
JIRA_DOMAIN = os.environ["JIRA_DOMAIN"]


def get_issues_page(project_key="AL", max_results=100, start_at=0):
    """Get a page of issues for a given project key."""
    jql = f"project={project_key}"
    url = f"https://{JIRA_DOMAIN}/rest/api/3/search"
    auth = HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN)
    headers = {
        "Accept": "application/json"
    }
    params = {
        "jql": jql,
        "maxResults": max_results,
        "startAt": start_at
    }
    response = requests.request(
        "GET",
        url,
        headers=headers,
        params=params,
        auth=auth
    )
    return response.json()


def get_all_issues_paginated(project_key="AL", max_results=100, limit=1000):
    """Generator that yields all issues for a given project key."""
    start_at = 0
    while True:
        response = get_issues_page(project_key, max_results, start_at)
        issues = response["issues"]
        start_at += max_results
        if not issues or (limit and start_at > limit):
            break
        yield from issues
