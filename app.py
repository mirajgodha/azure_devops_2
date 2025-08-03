from flask import Flask, request, render_template, jsonify
import os
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from openai import AzureOpenAI
import openai
import logging

load_dotenv()

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

api_type = "azure"
api_url = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
api_key = os.getenv("AZURE_OPENAI_KEY")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")


def get_ai_response(question, api_type="api1"):
    try:
        client = openai.AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=api_url,
        )

        logging.info(f"API type: {api_type}")
        # You can customize the behavior based on the API type
        if api_type == "api1":
            # Default behavior for API 1
            messages = [{"role": "user", "content": question}]
        else:
            # Different behavior for API 2
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question}
            ]

        response = client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            max_tokens=500,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error in {api_type}: {str(e)}")
        raise


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/api1", methods=["POST"])
def api1():
    try:
        logging.info(f"API type: api1")
        user_question = request.form.get("question", "").strip()
        logging.debug(f"API1 - User input: {user_question}")

        if not user_question:
            return jsonify({"error": "Please enter a question."}), 400

        answer = get_ai_response(user_question, "api1")
        return jsonify({"answer": answer})

    except Exception as e:
        logging.error(f"Error in API1: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api2", methods=["POST"])
def api2():
    try:
        logging.info(f"API type: api1")
        user_question = request.form.get("question", "").strip()
        logging.debug(f"API2 - User input: {user_question}")
        logging.info(f"API type: {api_type}")

        if not user_question:
            return jsonify({"error": "Please enter a question."}), 400

        # BASIC RAG-style logic
        custom_context = """
Quantum Root is a platform created by data engineering professionals.
It offers training and project-based learning in GenAI, Big Data, and NoSQL technologies.
It focuses on hands-on experience and real-world systems used in modern data stacks. 
Further details can be found at https://quantumroot.in.
Their course details can be found at https://quantumroot.in/courses
Refund policy can be found at https://quantumroot.in/refund
Blog can be found at https://quantumroot.in/blog
"""

        client = openai.AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=api_url,
        )

        messages = [
            {"role": "system", "content": custom_context},
            {"role": "user", "content": user_question}
        ]

        response = client.chat.completions.create(
            model=deployment_name,
            messages=messages,
        )

        logging.debug(f"OpenAI response: {response}")
        answer = response.choices[0].message.content
        return jsonify({"answer": answer})

    except Exception as e:
        logging.exception("Error during OpenAI API call")
        return jsonify({"error": str(e)}), 500


@app.route("/api3", methods=["GET", "POST"])
def api3():
    answer = ""
    if request.method == "POST":
        q = request.form.get("question", "").strip()

        if not q:
            return render_template("index.html", answer="Ask something!")

        # Retrieve relevant documents
        vectordb = Chroma(persist_directory=DB_DIR, embedding_function=HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-small"))
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        results = retriever.get_relevant_documents(q)

        context = "\n\n".join([doc.page_content for doc in results])

        messages = [
            {"role": "system", "content": f"Use the following context to answer:\n\n{context}"},
            {"role": "user", "content": q}
        ]

        client = AzureOpenAI(api_key=api_key, azure_endpoint=api_url, api_version=api_version)
        resp = client.chat.completions.create(model=deployment_name, messages=messages)
        answer = resp.choices[0].message.content

    return render_template("index.html", answer=answer)


if __name__ == "__main__":
    app.run(debug=True)