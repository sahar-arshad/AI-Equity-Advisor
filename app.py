# Import Necessary Libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import ast
from pagination import paginator
import style as cs
import random
import time
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os

# HuggingFace API Key for MistralAI
hk = "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"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hk

# Disclaimer to be dispalyed at the bottom of each tab
disclaimer = """
    <div style='background-color:#b43c42; color:#ffffff; padding:8px; border-radius:3px; font-size:12px''>
        <strong>Disclaimer:</strong> For demo purpose, the tool is currently populated with 10 months (Nov 2020 - Aug 2021) news 
                and historical data of oil sector from PSX. This data is intended to illustrate the tool's functionality and is not 
                intended for actual investment decisions.
    </div>
    """

# Utils Functions
def signals_to_plot(selected_indicator, num_signals, signal_column, data):
    if selected_indicator != 'RSI':
        if num_signals == 'All':
            buy_dates = data[data[signal_column] == 3.0]
            sell_dates = data[data[signal_column] == -3.0]
            hold_dates = data[data[signal_column] == 0]
        elif num_signals == 'Last 5 Days':
            last5 = data.tail(5)
            buy_dates = last5[last5[signal_column] == 3.0]
            sell_dates = last5[last5[signal_column] == -3.0]
            hold_dates = last5[last5[signal_column] == 0]
        elif num_signals == 'Last 15 Days':
            last15 = data.tail(15)
            buy_dates = last15[last15[signal_column] == 3.0]
            sell_dates = last15[last15[signal_column] == -3.0]
            hold_dates = last15[last15[signal_column] == 0]
        elif num_signals == 'Last 20 Days':
            last20 = data.tail(20)
            buy_dates = last20[last20[signal_column] == 3.0]
            sell_dates = last20[last20[signal_column] == -3.0]
            hold_dates = last20[last20[signal_column] == 0]

    elif selected_indicator == 'RSI':
        if num_signals == 'All':
            buy_dates = data[data[signal_column] >= 1.0]
            sell_dates = data[data[signal_column] <= -1.0]
            hold_dates = data[data[signal_column] == 0]
        elif num_signals == 'Last 5 Days':
            last5 = data.tail(5)
            buy_dates = last5[last5[signal_column] >= 1.0]
            sell_dates = last5[last5[signal_column] <= -1.0]
            hold_dates = last5[last5[signal_column] == 0]
        elif num_signals == 'Last 15 Days':
            last15 = data.tail(15)
            buy_dates = last15[last15[signal_column] >= 1.0]
            sell_dates = last15[last15[signal_column] <= -1.0]
            hold_dates = last15[last15[signal_column] == 0]
        elif num_signals == 'Last 20 Days':
            last20 = data.tail(20)
            buy_dates = last20[last20[signal_column] >= 1.0]
            sell_dates = last20[last20[signal_column] <= -1.0]
            hold_dates = last20[last20[signal_column] == 0]
    return buy_dates, sell_dates, hold_dates

def convert_str_to_list(string):
    try:
        # Use ast.literal_eval to safely evaluate the string as a list
        return ast.literal_eval(string)
    except (ValueError, SyntaxError):
        # If the string cannot be converted to a list, return it as is
        return string


# Extract Answer from LLM response
def get_answer(text):
  text = response['result']
  helpful_answer_index = text.find('Helpful Answer:')
  if helpful_answer_index != -1:
      helpful_answer = text[helpful_answer_index + len('Helpful Answer:'):].strip()
      print(helpful_answer)
  else:
      print("No helpful answer found.")
  return helpful_answer

# Streamed response emulator
def response_generator(answer):
    response = answer
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


# ---- WebApp ----
# Add Title and Logo
title_container = st.container(border=False)        # Create a container to hold the tile and logo 
col1, col2 = title_container.columns([0.2, 0.8], gap='medium')        # Create columns to display logo and title side-by-side
col1.image("logo.png")        # Add logo to the 1st column
col2.title("AI Equity Advisor")        # Add title to the 2nd column

# Add credits below the title
c1, c2 = col2.columns([0.5, 0.5], gap="large")
c1.markdown("Powered by GenInstigators")

# Load Technical Data
data_file_path = r"technicalRecommendation.csv"  # Update this with your file path
data = pd.read_csv(data_file_path)

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Set date limit for end date picker
date_limit = pd.to_datetime(data['Date'].max())   

# Set default current date
current_date = pd.to_datetime('2021-08-12')

# Create Tabs
market_analysis, news_analysis, final_recs, chat = st.tabs(["Market Analysis", "News Analysis", "GenAI Recommendations", "Ask AI Advisor"])

with market_analysis:
    st.header("Market Analysis", help = "This module provides market analysis for the following day based on the current date.")
    # st.write("This module provides market analysis for the following day based on the current date.")

    # Add date picker
    date_container = st.container(border=False) 
    col1, col2 = date_container.columns([0.5, 0.5], gap='medium')
    # start_date = col1.date_input('Start Date', value=default_start_date, min_value=data['Date'].min(), max_value=date_limit)
    end_date = col1.date_input("Current Date", value=current_date, min_value=data['Date'].min(), max_value=date_limit)

    # Filter data based on the date selected by the user
    start_date = pd.to_datetime(data['Date'].min())
    end_date = pd.to_datetime(end_date)
    data2 = data[data['Date'].between(start_date, end_date)]

    # Dropdown for selecting the indicator
    selected_indicator = st.selectbox("Select an Indicator", ['EMA 9', 'EMA 55', 'MACD', 'RSI'])
    
    # Dropdown for selecting the Number of Signal Days
    num_signals = st.selectbox("Signals to Show", ['None', 'All', 'Last 5 Days', 'Last 15 Days', 'Last 20 Days'])

    # Rename columns to maintain naming convention
    data2.rename(columns={'Close_price': 'Close Price', 'EMA_9': 'EMA 9', 'EMA_55': 'EMA 55'}, inplace=True)

    # Plot Close Price vs the indicator selected by the user
    if selected_indicator == 'EMA 9':
        # Plot close price and EMA 9
        fig = px.line(data2, x='Date', y=['Close Price', 'EMA 9'], title='Close Price vs EMA 9',
                    labels={'Date': 'Date', 'value': 'Price in Rs.', 'variable': 'Type'})
        fig.update_traces(selector=dict(type='scatter'))
        
        # Plot buy/sell signals
        if num_signals != 'None':
            # get signal values using the signals_to_plot utils function
            strong_buy_dates, strong_sell_dates, strong_hold_dates = signals_to_plot(
                selected_indicator=selected_indicator, 
                num_signals=num_signals, 
                signal_column='EMA9_Signal', 
                data=data2)
            
            # Add Buy signals
            fig.add_scatter(x=strong_buy_dates['Date'], y=strong_buy_dates['EMA 9'], mode='markers', 
                            marker=dict(symbol='triangle-up', size=10, color=cs.pos_impacts_color), name='Strong buy')  
            # Add Sell signals
            fig.add_scatter(x=strong_sell_dates['Date'], y=strong_sell_dates['EMA 9'], mode='markers', 
                            marker=dict(symbol='triangle-down', size=10, color=cs.neg_impacts_color), name='Strong sell')
        
        # Add date range selection buttons to chart
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        # Update y-axis to allow vertical scrolling and dragging
        fig.update_yaxes(fixedrange=False)
        
        # Show chart on WebApp
        st.plotly_chart(fig)
    
    elif selected_indicator == 'EMA 55':
        # Plot close price and EMA 9
        fig = px.line(data2, x='Date', y=['Close Price', 'EMA 55'], title='Close Price vs EMA 55',
                    labels={'Date': 'Date', 'value': 'Price in Rs.', 'variable': 'Type'})
        fig.update_traces(selector=dict(type='scatter'))
        
        # Plot buy/sell signals
        if num_signals != 'None':
            # get signal values using the signals_to_plot utils function
            strong_buy_dates, strong_sell_dates, strong_hold_dates = signals_to_plot(
                selected_indicator=selected_indicator, 
                num_signals=num_signals, 
                signal_column='EMA55_Signal', 
                data=data2)

            # Add Buy signals
            fig.add_scatter(x=strong_buy_dates['Date'], y=strong_buy_dates['EMA 55'], mode='markers', 
                            marker=dict(symbol='triangle-up', size=10, color=cs.pos_impacts_color), name='Strong buy')
            # Add Sell signals
            fig.add_scatter(x=strong_sell_dates['Date'], y=strong_sell_dates['EMA 55'], mode='markers', 
                            marker=dict(symbol='triangle-down', size=10, color=cs.neg_impacts_color), name='Strong sell')
        
        # Add date range selection buttons to chart
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        # Update y-axis to allow vertical scrolling and dragging
        fig.update_yaxes(fixedrange=False)
        
        # Show chart on WebApp
        st.plotly_chart(fig)

    elif selected_indicator == 'MACD':
        # Set up the figure and subplots
        fig = make_subplots(rows=2, cols=1)
        
        # Add subplot for Close Price and Signals
        fig.add_trace(go.Scatter(x=data2['Date'], y=data2['Close Price'], mode='lines', name='Close Price'),
                        row=1, col=1)
        
        # Plot buy/sell signals
        if num_signals != 'None':
            # get signal values using the signals_to_plot utils function
            strong_buy_dates, strong_sell_dates, strong_hold_dates = signals_to_plot(
                selected_indicator=selected_indicator, 
                num_signals=num_signals, 
                signal_column='MACD_Signals', 
                data=data2)
            
            # Add Buy signals
            fig.add_trace(go.Scatter(x=strong_buy_dates['Date'], y=strong_buy_dates['Close Price'], mode='markers', 
                        marker=dict(symbol='triangle-up', size=10, color=cs.pos_impacts_color), name='Strong Buy'), row=1, col=1)
            # Add Sell signals
            fig.add_trace(go.Scatter(x=strong_sell_dates['Date'], y=strong_sell_dates['Close Price'], mode='markers', 
                        marker=dict(symbol='triangle-down', size=10, color=cs.neg_impacts_color), name='Strong Sell'), row=1, col=1)
            # Add Hold signals
            fig.add_trace(go.Scatter(x=strong_hold_dates['Date'], y=strong_hold_dates['Close Price'], mode='markers', 
                        marker=dict(symbol='circle', size=10, color='orange'), name='Hold'), row=1, col=1)

        # Add subplot for MACD
        # fig2 = go.Figure()
        fig.add_trace(go.Scatter(x=data2['Date'], y=data2['MACD_12_26_9'], mode='lines', name='MACD', yaxis='y2',
                                line=dict(dash='solid', color=cs.macd_color, width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=data2['Date'], y=data2['MACDs_12_26_9'], mode='lines', name='Signal', yaxis='y2',
                                line=dict(dash='solid', color=cs.macd_signal_color, width=2)), row=2, col=1)
        fig.add_trace(go.Bar(x=data2['Date'], y=data2['MACDh_12_26_9'], name='Histogram', yaxis='y2',
                            marker=dict(color=cs.macd_hist)), row=2, col=1)

        # Update layout
        fig.update_layout(title='Close Price vs MACD')
        
        # Add date range selection buttons to chart
        fig.update_xaxes(
            rangeslider_visible=False,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        # Update y-axis to allow vertical scrolling and dragging
        fig.update_yaxes(fixedrange=False)
        
        # Show chart on WebApp
        st.plotly_chart(fig, use_container_width=True)
        

    elif selected_indicator == 'RSI':
        # Set up the figure
        fig = go.Figure()

        # Add RSI line
        fig.add_trace(go.Scatter(x=data2['Date'], y=data2['RSI'], mode='lines', name='RSI',
                     line=dict(dash='solid', color=cs.rsi_color, width=2)))

        # Add overbought and oversold lines
        overbought_strong = 79
        oversold_strong = 22
        fig.add_shape(type="line", x0=data2['Date'].min(), y0=overbought_strong, x1=data2['Date'].max(), y1=overbought_strong, line=dict(color="red", width=1, dash="dash"), name="Overbought")
        fig.add_shape(type="line", x0=data2['Date'].min(), y0=oversold_strong, x1=data2['Date'].max(), y1=oversold_strong, line=dict(color="green", width=1, dash="dash"), name="Oversold")

        # Plot buy/sell signals
        if num_signals != 'None':
            # get signal values using the signals_to_plot utils function
            strong_buy_dates, strong_sell_dates, strong_hold_dates = signals_to_plot(
                selected_indicator=selected_indicator,
                num_signals=num_signals, 
                signal_column='RSI_Signals', 
                data=data2)

            # Add Buy signals
            fig.add_trace(go.Scatter(x=strong_buy_dates['Date'], y=strong_buy_dates['RSI'], mode='markers', 
                                     marker=dict(symbol='triangle-up', size=10, color=cs.pos_impacts_color), name='Strong Buy'))
            # Add Sell signals
            fig.add_trace(go.Scatter(x=strong_sell_dates['Date'], y=strong_sell_dates['RSI'], mode='markers', 
                                     marker=dict(symbol='triangle-down', size=10, color=cs.neg_impacts_color), name='Strong Sell'))
            # fig.add_trace(go.Scatter(x=strong_hold_dates['Date'], y=strong_hold_dates['RSI'], mode='markers', marker=dict(symbol='circle', size=10, color='orange'), name='Hold'))
        
        fig.update_layout(title='RSI Analysis', showlegend=True)
        
        # Add date range selection buttons to chart
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        # Update y-axis to allow vertical scrolling and dragging
        fig.update_yaxes(fixedrange=False)
        st.plotly_chart(fig)
    # st.write(data2)

    
    # Add discalimer
    st.markdown(disclaimer, unsafe_allow_html=True)
    
with news_analysis:
    st.header("News Analysis", help="This module provides news based event impact for the following day based on the current date.")
    # st.write("This module provides news based event impact for the following day based on the current date.")
    
    # Load News Events data
    data_file_path = r"Events_SameDay.csv"  # Update this with your file path
    events = pd.read_csv(data_file_path, encoding="ISO-8859-1", lineterminator='\n')

    # Convert 'Date' column to datetime format
    events['Date'] = pd.to_datetime(events['Date'])

    # Filter data based on the date selected by the user
    events = events[(events['Date'] >= start_date) & (events['Date'] <= end_date)]

    # Use convert_str_to_list utils function to restore list value data type
    cols = ['Raw_Headline', 'Bold_KW', 'Feature', 'Raw_News', 'Sources', 'Urls']
    for col in cols:
        events[col] = events[col].apply(convert_str_to_list)
    
    # Get unique features
    events['SetFeature'] = events['Feature'].apply(lambda x: str(set(x)))
    
    # Add a new column for positive values of column A
    events['Positive_Impacts'] = events[events['Events_Impact'] >= 4.7]['Events_Impact']

    # Add a new column for negative values of column A
    events['Negative_Impacts'] = events[events['Events_Impact'] <= -4.7]['Events_Impact']

    # Fill NaN values in the new columns with 0
    events['Positive_Impacts'].fillna("", inplace=True)
    events['Negative_Impacts'].fillna("", inplace=True)

    # Filter out subset dataframes to plot positive & negative impacts
    plot_sub_pos = events[events['Positive_Impacts']!='']
    plot_sub_neg = events[events['Negative_Impacts']!='']
    
    # Create the line trace for stock prices
    line_stock = go.Scatter(x=events['Date'], y=events['Price'], mode='lines', name='OGDCL Close Price',
        line=dict(dash='solid', color=cs.close_line_color, width=2),
        customdata=events['SetFeature'],
        hovertemplate='%{x}<br>Close: %{y}<br>Feature: %{customdata}<br>',
        )
    title = 'OGDCL Close Price vs News Impact'
    layout = go.Layout(
        title=title,
        xaxis=dict(
            title='Date',
            tickformat='%b %d, %Y',
            # gridcolor='lightgray',
            range=[start_date, end_date],
            # tickvals=list(range(dateA, dateB, 3)),
        ),
        yaxis=dict(
            title='Price in Rs.',
            # gridcolor='lightgray',
            range=[90, 120],
            tickvals=list(range(90, 120, 5)),
        )
    )

    # Add all traces to the figure
    figure = go.Figure(data=[line_stock], layout=layout)
    # Add Positive impacts
    figure.add_scatter(x=plot_sub_pos['Date'], y=plot_sub_pos['Price'], mode='markers', 
        marker=dict(symbol='triangle-up', size=10, color=cs.pos_impacts_color), name='Positive Impact', 
        customdata=plot_sub_pos['SetFeature'], hovertemplate='%{x}<br>Close: %{y}<br>Feature: %{customdata}<br>')  
    # Add Negative impacts
    figure.add_scatter(x=plot_sub_neg['Date'], y=plot_sub_neg['Price'], mode='markers', 
        marker=dict(symbol='triangle-down', size=10, color=cs.neg_impacts_color), name='Negative Impact', 
        customdata=plot_sub_neg['SetFeature'], hovertemplate='%{x}<br>Close: %{y}<br>Feature: %{customdata}<br>',)
    
    # Update Layout
    figure.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'y': 0.95,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=12),
        },
        hovermode='closest',
        margin=dict(l=40, r=40, t=80, b=40),
        modebar_add="togglespikelines",
    )
    # Add date range selection buttons to chart
    figure.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    # Update y-axis to allow vertical scrolling and dragging
    figure.update_yaxes(fixedrange=False)
    st.plotly_chart(figure)

    # Add subheader for news section
    st.subheader("News Events")
    """In this section, news events for each date in the data will be displayed along the features for that date"""
    # Filter data for news events
    news = events[events['Date'].between(start_date, end_date, inclusive='both')]
    news = news[['Date', 'Raw_Headline', 'Bold_KW', 'Feature', 'Raw_News', 'Sources', 'Urls']]
   
    # Extract only the date from the datetime
    news['Date'] = news['Date'].dt.date
    
    # Sort DataFrame based on the 'Date' column in descending order
    news = news.sort_values(by='Date', ascending=False)
   
    # Reset index to reflect the new order
    news.reset_index(drop=True, inplace=True)

    # Get all the unique dates to iterate over
    dates = list(news['Date'].unique())
    
    # Sort the date list
    dates = np.sort(dates)
    
    # Reverse the array to have the latest date at index 0
    dates = dates[::-1]

    # Decide number of items to display per page
    num_dates = len(dates)
    items_per_page = min(num_dates, 5)

    # iterate over the paginator
    for i, date in paginator("Select Page Number", dates,  items_per_page=items_per_page, on_sidebar=False, ukey='news_pages'):
        # Display the date
        st.write(f'<span style="font-size: large;"><b>Date:</b> <u>{date}</u></span>', unsafe_allow_html=True)

        # Filter data for each date in the loop
        filtered_news = news[news['Date'] == date]

        # Extract the details required
        features = filtered_news['Feature'].sum()
        headlines = filtered_news['Raw_Headline'].sum()
        news_list = filtered_news['Raw_News'].sum() 
        sources = filtered_news['Sources'].sum()
        urls = filtered_news['Urls'].sum()

        # Create a container to display news for each date
        main_container = st.container(height = 250, border=True)

        # Create columns to display news on one side and features on the other
        col1, col2 = main_container.columns([0.7, 0.3], gap='medium')

        # Display each headline in the extracted headlines in the container
        for index, headline in enumerate(headlines):
            # Link news article's Url to the headline to redirect to the source article webpage on click
            col1.page_link(urls[index], label=f"**:blue[{headline}]**")
            # Display news source in the container
            col1.write(f"<span style='font-size: small;'>By {sources[index]}</span><br>", unsafe_allow_html=True)
            # Display news content on click
            with col1:              
                with st.expander("Show Full Article"):
                    st.write(news_list[index])

        # Display features on click
        with col2:
            with st.expander("Oil Sector Features"):
                st.write(set(features))
                

    # Add Disclaimer
    st.markdown(disclaimer, unsafe_allow_html=True) 


with final_recs:
    help = """This module provides trading recommendation for the following day based on the current date.
    For demo purpose this is restricted to test data from (Aug 12, 2021- Aug 31,2021). 
    The results shown here are based on our model's inference on this test data, which is available in the Colab Notebook provided along GitHub submission. 
    """
    st.header("GenAI Recommendations", help=help)
    # st.write("""This module provides trading recommendation for the following day based on the current date.
    # For demo purpose this is restricted to test data from (Aug 12, 2021- Aug 31,2021). 
    # The results shown here are based on our model's inference on this test data, which is available in the Colab Notebook provided along GitHub submission. 
    # """)

    # Load generated recommendations data
    recs = pd.read_csv("test_recom1.csv")

    # Convert date column to datetime values
    recs['Date'] = pd.to_datetime(recs['Date'])
    
    # Get only the date from datetime
    recs['Date'] = recs['Date'].dt.date
    
    # Get all the unique dates to add to the selectbox and to iterate over
    rec_dates = np.sort(list(recs['Date'].unique()))
    
    # Create the date select box
    pred_date = st.selectbox("Pick the Test Date", rec_dates)
    
    # Store the close price value of the following day for each date in a dictionary to call later
    fp = {}   # initialize an empty dictionary
    for index, d in enumerate(rec_dates[:-1]):    # iterate over the unique dates
        fr = recs[recs['Date'] == rec_dates[index+1]]   # get data of the following day
        fr.reset_index(inplace=True, drop=True)    # reset index
        following_price = fr['Price'][0]    # get close price
        fp[d] = following_price    # append dictionary

    # As no following day data is available for the latest date in the list, assign it 'Not Available'
    fp[rec_dates[-1]] = 'Not Available'

    # Add radio buttons to select role
    role = st.radio(
    "Show recommendation summary as:",
    ["Active Trader", "Equity Analyst"], horizontal=True)

    # filter data based on the date selected by the user
    filter_recs = recs[recs['Date'] == pred_date]

    # filter required data based on the role selected by the user
    if role == 'Active Trader':
        trade_recs = filter_recs[['Date', 'Recommendations_Active_Trader', 'Price']]
        # Convert back to Dictionaries from strings
        trade_recs['Recommendations_Active_Trader'] = trade_recs['Recommendations_Active_Trader'].apply(convert_str_to_list)
        trade_recs.rename(columns={'Recommendations_Active_Trader': 'Recommendations'}, inplace=True)
    elif role == 'Equity Analyst':
        trade_recs = filter_recs[['Date', 'Recommendations_Equity_Analyst', 'Price']]
        # Convert back to Dictionaries from strings
        trade_recs['Recommendations_Equity_Analyst'] = trade_recs['Recommendations_Equity_Analyst'].apply(convert_str_to_list)
        trade_recs.rename(columns={'Recommendations_Equity_Analyst': 'Recommendations'}, inplace=True)

    # reset index after filteration
    trade_recs.reset_index(inplace=True, drop=True)  

    # create container to display generated recommendations
    genrec_container = st.container(border=False)

    # create columns to display date, current close price, and following day close price side-by-side
    rec_col1, rec_col2, rec_col3 = genrec_container.columns(3, gap='medium')

    # Show selected date
    rec_col1.write(f'<span style="font-size: large;"><b>Current Date:</b> <u>{pred_date}</u></span>', unsafe_allow_html=True)
    # Show selected date close price
    current_price = trade_recs['Price'][0]
    rec_col2.write(f'<span style="font-size: large;"><b>Current Close Price:</b> {current_price}</span>', unsafe_allow_html=True)
    # Show following day close price
    rec_col3.write(f'<span style="font-size: large;"><b>Following Close Price:</b> {fp[pred_date]}</span>', unsafe_allow_html=True)

    # Show generated recommendations
    genrec_container.subheader("Generated Recommendation")
    genrec_container.write(trade_recs['Recommendations'][0])

    # Show Market and News Analysis w.r.t. OGDCL Close Price chart
    # Create the line trace for stock prices
    line_stock = go.Scatter(x=events['Date'], y=events['Price'], mode='lines', name='OGDCL Close Price',
        line=dict(dash='solid', color=cs.close_line_color, width=2),
        text=events['EMA9_Signal'],
        hovertext=events['EMA55_Signal'],
        meta = events["RSI_Signals"],
        customdata=events['MACD_Signals'],
        hovertemplate='%{x}<br>Close: %{y}<br> EMA9 Signal: %{text}<br>EMA55 Signal: %{hovertext}<br> RSI Signal: %{meta}<br>MACD Signal: %{customdata}<br>',
        # hoverlabel=dict(font=dict(color=events
        # ['FeatureSentiment'].apply(lambda x: 'red' if x == 'Negative' else 'blue' if x == 'Neutral' else 'green'))),  # Customize the line style, color, and width
    )
    title = 'Market and News Analysis w.r.t. OGDCL Close Price'
    layout = go.Layout(
        title=title,
        xaxis=dict(
            title='Date',
            tickformat='%b %d, %Y',
            # gridcolor='lightgray',
            range=[start_date, end_date],
            # tickvals=list(range(dateA, dateB, 3)),
        ),
        yaxis=dict(
            title='Price in Rs.',
            # gridcolor='lightgray',
            range=[90, 120],
            tickvals=list(range(90, 120, 5)),
        )
    )

    # Add all traces to the figure
    figure = go.Figure(data=[line_stock], layout=layout)
    # Add positive impact
    figure.add_scatter(x=plot_sub_pos['Date'], y=plot_sub_pos['Price'], mode='markers', 
        marker=dict(symbol='triangle-up', size=10, color=cs.pos_impacts_color), name='Positive Impact', 
        # customdata=plot_sub_pos['SetFeature'], 
        text=events['EMA9_Signal'],
        hovertext=events['EMA55_Signal'],
        meta = events["RSI_Signals"],
        customdata=events['MACD_Signals'],
        hovertemplate='%{x}<br>Close: %{y}<br> EMA9 Signal: %{text}<br>EMA55 Signal: %{hovertext}<br> RSI Signal: %{meta}<br>MACD Signal: %{customdata}<br>',)  
    # Add negative impact
    figure.add_scatter(x=plot_sub_neg['Date'], y=plot_sub_neg['Price'], mode='markers', 
        marker=dict(symbol='triangle-down', size=10, color=cs.neg_impacts_color), name='Negative Impact', 
        text=events['EMA9_Signal'],
        hovertext=events['EMA55_Signal'],
        meta = events["RSI_Signals"],
        customdata=events['MACD_Signals'],
        hovertemplate='%{x}<br>Close: %{y}<br> EMA9 Signal: %{text}<br>EMA55 Signal: %{hovertext}<br> RSI Signal: %{meta}<br>MACD Signal: %{customdata}<br>',)

    # Update layout
    figure.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'y': 0.95,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=12),
        },
        hovermode='closest',
        margin=dict(l=40, r=40, t=80, b=40),
        modebar_add="togglespikelines",
    )
    # Add date range selection buttons to chart
    figure.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    # Update y-axis to allow vertical scrolling and dragging
    figure.update_yaxes(fixedrange=False)
    st.plotly_chart(figure)

    # Add Disclaimer
    st.markdown(disclaimer, unsafe_allow_html=True)
    
with chat:
    # st.header("Chat with AI Stock Advisor")

    # loader = CSVLoader("Events_SameDay.csv",encoding='iso-8859-1')
    
    # Initialize HuggingFace Instruct Embeddings
    embeddings = HuggingFaceInstructEmbeddings()

    # Load saved Vector Store
    persist_directory = 'FAISS_VectorStore'
    db = FAISS.load_local(persist_directory, embeddings, allow_dangerous_deserialization=True)
    
    # Initialize GenAI LLM Model
    repo_id = "mistralai/Mistral-7B-Instruct-v0.1"
    llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_new_tokens": 1024})

    # Define Prompt Template
    system_prompt = """You are a financial expert for stock market who can perform multiple tasks for the intended user including trading 
    recommendations with reasoning, retrieving articles with their impact in the market, retrieving or enlisting features affecting market 
    trends (could be positive or negative).However, if a user is asking for trading recommendation, then you need to generate trading signal 
    recommendations utilizing insights from two approaches. One is the technical indicators signals EMA55, RSI, EMA9, and MACD (all ranging 
    from -3 to 3, where –3 is strong sell, -2 is moderate sell, -1 is weak sell, 0 is for hold, 1 is for weak buy, 2 is for moderate buy 
    and 3 is for strong buy) from the respective signal while other insight is from news impacts (either positive or negative between -5 to 5).
    Provide your recommendation with balanced approach if news impact is too much positive or negative, technical indicator can be ignored and 
    buy or sell suggestion based on news impact can be given. On the contrary, if technical indicators are opposite to news impact, 
    a hold position is a reasonable suggestion. If technical indicators are all positive along news impact, strong buy signal can be 
    generated. If technical indicators and news impact are all negative a strong sell signal can be generated. If news impact is too low, 
    then generate recommendation based on technical indicator specially with more weightage to ema 55 in all the technical indicators. 
    Your response should cover all technical aspects including the analysis of technical indicators as well as the news impact. Also cover 
    logical financial rational as well as the explanations with your answer."""
    B_INST, E_INST = "<s>[INST] ", " [/INST]"
    template = (
                    B_INST
                    + system_prompt
                    + """
                Context: {context}
                User: {question}
                """
                    + E_INST +
                "\nHelpful Answer: \n"
                )
    sys_prompt = PromptTemplate(input_variables=["context", "question"], template=template)

    # Create QA Chain
    chain = RetrievalQA.from_chain_type(
    llm=llm,    # Add LLM
    chain_type="stuff", 
    retriever=db.as_retriever(),     # Add Vector Store
    input_key="question",
    chain_type_kwargs={"prompt": sys_prompt})    # Add prompt template

    # Add Container to display chat history
    chat_container = st.container(height = 265, border=False)
    with chat_container:
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
    
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    
    # st.divider()    # Divider to separate chat history and chat input
    # Accept user input
    if prompt := st.chat_input("Enter your query here.", key='input2'):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with chat_container.chat_message("user"):
            st.markdown(prompt)

        # Get Response to user query from LLM
        response = chain({"question": prompt})
        # Extract the answer from the response
        result = get_answer(response['result'])

        # Display assistant response in chat message container
        with chat_container.chat_message("assistant"):
            response = st.write_stream(response_generator(result))
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    

    

    
    