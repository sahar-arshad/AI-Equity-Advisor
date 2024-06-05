# AI Equity Advisor

## Table of contents
* [Introduction](#introduction)
* [Features](#features)
* [Technologies](#technologies)
* [Getting Started](#getting-started)
* [Steps Guide](#steps-guide)
* [Feedback](#feedback)
  
## Introduction 

Welcome to the AI Equity Advisor! This solution aims to revolutionize the investment recommendation process for financial stakeholders, providing transparency, efficiency, and accuracy in decision-making. By combining real-time analysis of news based event impacts in conjuction with technical indicators, our tool offers well-informed stock recommendations in the dynamic stock market environment.

## Features

- Automatic detection of real-time news based events and their impact on the market trend.
- Integration of technical indicators with actionable trading insights for comprehensive analysis.
- Efficient and personalized stock recommendations via. Custom Generative module.
- QAbot integration allowing analysts to ask questions and recieve instant guidance from AI-stock Advisor.

## Technologies

- Programming Language: Python (version: 3.10.4)
- UI Framework: Streamlit (version: 1.33.0)
- Generative AI: LLama2, Mistral
- Framework: langchain, huggingface_hub, torch
- Technical Indicators Library: TA-Lib
- Data Visualization: plotly (version: 5.21.0)
- Embeddings: HuggingFaceInstructEmbeddings
- Embeddings Storage: faiss  
  
## Getting Started 
1. To ensure smooth execution of our code, we recommend you to access our AI Equity Advisor tool via. the Hugging Face Hub [here](https://huggingface.co/spaces/GenInstigators/NLFF-AIChallenge). This will help prevent any potential conflicts with libraries on a local setup and ensure optimal performance.
2. Alternatively, you can run our Streamlit app by navigating to the directory containing `app.py` in your terminal and execute the command: streamlit run app.py.
4. We have embedded the inference model results of test data already in the app.py file. However,to utilize the GenAI module for recommendation inference in real time, ensure to execute the ["GenAI_recommendation_model_inference"](https://github.com/sahar-arshad/AI-Equity-Advisor/blob/main/GenAI_recommendation_model_inference.ipynb) notebook with GPU access enabled.
