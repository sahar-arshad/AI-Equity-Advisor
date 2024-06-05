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
4. We have embedded the inference model results of test data already in the app.py file. However,to utilize the GenAI module for recommendation inference in real time, ensure to execute the ["GenAI_recommendation_model_inference"](https://github.com/Huma-Ameer10/2024-AI-Challenge-GenInstigators/blob/main/GenAI_recommendation_model_inference.ipynb) notebook with GPU access enabled.

## Steps Guide

1. <b>Market Analysis</b> module provides actionable market insights for the following day based on the Current Date selected. Each indicator generates trading signals based on data driven trading startegies employed. Preference based signal range can be selected to plot on the close trend.
   
   ![market](https://github.com/Huma-Ameer10/2024-AI-Challenge-GenInstigators/assets/88269723/499a6af8-1cb9-4800-bae6-4d11f1cc3357)

2. <b>News Analysis</b> module can automatically detect the events from the real-time news to highlight their impact on the market trend. Detail of events with high impacts is also in display for comprehensive analysis.

   ![nws](https://github.com/Huma-Ameer10/2024-AI-Challenge-GenInstigators/assets/88269723/2b2fa0cf-f8c3-45a0-b63e-c0f066bc74e6)   

   ![events_identified](https://github.com/Huma-Ameer10/2024-AI-Challenge-GenInstigators/assets/88269723/5a84b589-90f1-41f8-8a82-91a6f54c37b8)


3. <b>GenAI Recommendations</b> is a module built with GenAI technology which provides Custom recommendations to the user based on his role as an equity analyst or an active trader. By analysing both the technical indicators and news based event impact, a risk tolerance based recommendation along with its rational is generated for the following day.For quick accuracy check of the generated recommendation market close price for current day and following day are also provided.

   ![genAIrecomm](https://github.com/Huma-Ameer10/2024-AI-Challenge-GenInstigators/assets/88269723/774dcc60-6656-4249-a68d-06925c40683b)

6. <b>Ask AI Advisor</b> allows the user to have an instant guide on a matter of investment from an AI powered Stock Advisor.

   ![CHAT](https://github.com/Huma-Ameer10/2024-AI-Challenge-GenInstigators/assets/88269723/50ce3548-7cc4-477a-8d78-08d4037a21a6) 

