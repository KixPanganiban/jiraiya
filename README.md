# Jiraiya
### The JIRA Ninja

## Setup
To set up this project, follow the steps below:

1. Clone the repository to your local machine.
2. Install the required dependencies listed in the `requirements.txt` file. You can use the following command to install them:
    
    ```
    pip install -r requirements.txt
    ```

3. Make sure you have the necessary environment variables set up in `.env`:
    
    - `OPENAI_API_KEY`: Your OpenAI API key.
    - `JIRA_EMAIL`: Your JIRA email address.
    - `JIRA_API_TOKEN`: Your JIRA API token.
    - `JIRA_DOMAIN`: The domain of your JIRA instance.

## Initialization
To initialize the project, run the main.py script with the init command. This will build the JIRA index and the vector store. Use the following command:

```
python main.py init
```

## Asking Questions
To ask questions based on the JIRA data, use the ask command followed by your question. For example:
```
python main.py ask "What is Kix working on?"
```
This will query the JIRA data in the vector store and provide a response based on the provided question.
