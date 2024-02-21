# <img src="docs/robot.png" alt="Image Description" width="50"/>  Chat with a Website using RAG - LangChain Chatbot with Streamlit GUI
<a href="https://shivr-webchat-rag-langchain.streamlit.app/"><img src="https://img.shields.io/badge/deployment-website-blue" alt="Website"/></a>

## 📝 Overview 

This project implements a chatbot that interacts with a website using the RAG (Retrieval-Augmented Generation) model powered by the LangChain framework. The chatbot is integrated into a Streamlit GUI (Graphical User Interface) for easy usage. It provides a seamless conversational experience, enabling users to interact with the website content through natural language.

## 🏗️ Features

- **Website Integration:** Interacts with website content through a conversational interface.
- **RAG Model:** Employs the RAG (Retrieval-Augmented Generation) model for generating responses based on retrieved knowledge.
- **Streamlit GUI:** Provides a user-friendly interface powered by Streamlit, allowing users to interact with the chatbot effortlessly.
- **Streamlit Cloud:** Deployed on the web using Streamlit Cloud.

## 👨‍🏫 How RAG works
A RAG bot is short for Retrieval-Augmented Generation. This means that we are going to "augment" the knowledge of our LLM with new information that we are going to pass in our prompt. We first vectorize all the text that we want to use as "augmented knowledge" and then look through the vectorized text to find the most similar text to our prompt. We then pass this text to our LLM as a prefix.

![RAG Diagram](docs/HTML-rag-diagram.jpg)

## 🛠 Deployment 

The chatbot can be accessed [here](https://shivr-webchat-rag-langchain.streamlit.app/) on the web using Streamlit Cloud.

## 👨‍💻 Installation
Ensure you have Python installed on your system. Then clone this repository:
```bash
git clone [repository-link]
cd [repository-directory]
```
Install the required packages:
```bash
pip install -r requirements.txt
```
## 💻 Usage

### Website
- Obtain & enter OpenAI API key
- Enter the webpage to chat with
- Start typing your query, once the bot initalizes on the screen
  
### Local
To run the Streamlit app:

```bash
streamlit run app.py
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.
