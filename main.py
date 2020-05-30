import json
import os
from flask import Flask
from flask import request
from flask import make_response
from flask import redirect, url_for
import matplotlib
matplotlib.use('Agg')
import time
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from linebot import (LineBotApi, WebhookHandler)
from linebot.models import (
    ImageSendMessage,
    TextSendMessage
)
# from mlxtend.frequent_patterns import apriori, association_rules 
# from gcloudconnector import BigQueryConnector
# config_file_path = 'config/config.json'
# gbq_connector = BigQueryConnector()
# gbq_connector.init_conn_with_config_file(config_file_path)

# def supermarket_data(path):
#     sql_code = f"SELECT * FROM {path}"
#     query_data = gbq_connector.query(sql_code, output_type='df')
#     return query_data

# path = '`superstore.supermarket_data`'
# df = supermarket_data(path)
df = pd.read_csv('supermarket_data.csv')
df['SHOP_DATE'] = pd.to_datetime(df['SHOP_DATE'], format = '%Y%m%d')

app = Flask(__name__)
line_bot_api = LineBotApi('')
handler = WebhookHandler('')
with open('config/ngrok.json') as f:
  config = json.load(f)

@app.route('/', methods=['POST'])
# ----- Main Function -----
def MainFunction():
    #Getting intent from Dailogflow
    question_from_dailogflow_raw = request.get_json(silent=True, force=True)
    #Call generating_answer function to classify the question
    answer_from_bot = generating_answer(question_from_dailogflow_raw)
    #Make a respond back to Dailogflow
    r = make_response(answer_from_bot)
    #Setting Content Type
    r.headers['Content-Type'] = 'application/json' 
    return r

# ----- Generate Answer -----
def generating_answer(question_from_dailogflow_dict):
    #Print intent that recived from dialogflow.
    print(json.dumps(question_from_dailogflow_dict, indent=4 ,ensure_ascii=False))
    #Getting intent name form intent that recived from dialogflow.
    intent_group_question_str = question_from_dailogflow_dict["queryResult"]["intent"]["displayName"] 
    #Select function for answering question
    if intent_group_question_str == 'ยอดขายรวมย้อนหลัง x วัน':
        answer_str = sales_1(question_from_dailogflow_dict)
    elif intent_group_question_str == 'ยอดขายรวมของวันที่ xxxx-xx-xx':
        answer_str = sales_2(question_from_dailogflow_dict)
    elif intent_group_question_str == 'ยอดขายรวมระหว่างวันที่ xxxx-xx-xx ถึง xxxx-xx-xx':
        answer_str = sales_3(question_from_dailogflow_dict)
    elif intent_group_question_str == 'ยอดขายรวมย้อนหลัง x วัน ของร้านที่ x x':
        answer_str = sales_4(question_from_dailogflow_dict)
    elif intent_group_question_str == 'ยอดขายรวมของวันที่ xxxx-xx-xx  ของร้านที่ x':
        answer_str = sales_5(question_from_dailogflow_dict)
    elif intent_group_question_str == 'ยอดขายรวมของวันที่ xxxx-xx-xx  ของร้านที่ x x':
        answer_str = sales_6(question_from_dailogflow_dict)
    elif intent_group_question_str == 'ยอดขายรวมระหว่างวันที่ xxxx-xx-xx ถึง xxxx-xx-xx ของร้านที่ x':
        answer_str = sales_7(question_from_dailogflow_dict)
    elif intent_group_question_str == 'ยอดขายรวมระหว่างวันที่ xxxx-xx-xx ถึง xxxx-xx-xx ของร้านที่ x x':
        answer_str = sales_8(question_from_dailogflow_dict)
    elif intent_group_question_str == 'ยอดขายรวมย้อนหลัง x วัน ของร้านที่ x':
        answer_str = sales_9(question_from_dailogflow_dict)
    elif intent_group_question_str == 'สินค้าไหนถูกซื้อบ่อยสุด x อันดับแรก':
        answer_str = product_1(question_from_dailogflow_dict)
    elif intent_group_question_str == 'สินค้าไหนถูกซื้อบ่อยสุด x อันดับแรก ของร้านที่ x':
        answer_str = product_2(question_from_dailogflow_dict)
    elif intent_group_question_str == 'จำนวนลูกค้าทั้งหมด':
        answer_str = customer_1(question_from_dailogflow_dict)
    elif intent_group_question_str == 'จำนวนลูกค้าทั้งหมดของร้านที่ x':
        answer_str = customer_2(question_from_dailogflow_dict)
    elif intent_group_question_str == 'สินค้าที่ทำเงิน x% จากยอดขายทั้งหมด':
        answer_str = item_1(question_from_dailogflow_dict)
    elif intent_group_question_str == 'สินค้าที่ทำเงิน x% จากยอดขายทั้งหมด ของร้านที่ x':
        answer_str = item_2(question_from_dailogflow_dict)
    elif intent_group_question_str == 'สินค้าไหนถูกซื้อคู่กันบ่อยที่สุด':
        answer_str = product_3(question_from_dailogflow_dict)
    else: 
        answer_str = "ผมไม่เข้าใจ คุณต้องการอะไร"
    #Build answer dict 
    answer_from_bot = {"fulfillmentText": answer_str}
    #Convert dict to JSON
    answer_from_bot = json.dumps(answer_from_bot) 
    return answer_from_bot

# ----- Answer String -----
def sales_1(respond_dict):
    days = float(respond_dict["queryResult"]["outputContexts"][0]["parameters"]["days.original"])
    answer_str = f"ยอดขายย้อนหลัง {days:.0f} วัน คือ {total_previous_sales(df, days):,.2f} บาทจ้า"
    user_id = respond_dict['originalDetectIntentRequest']['payload']['data']['source']['userId']
    # Generate Graph
    image_name = each_previous_sales(df, days)
    # Response Text
    line_bot_api.push_message(user_id, ImageSendMessage(
        original_content_url=f"{config['url']}/static/{image_name}",
        preview_image_url=f"{config['url']}/static/{image_name}"
    ))
    return answer_str

def sales_2(respond_dict):
    date = respond_dict["queryResult"]["outputContexts"][0]["parameters"]["date.original"]
    answer_str = f"ยอดขายวันที่ {date} คือ {total_sales_on_date(df, date):,.2f} บาทจ้า"
    return answer_str

def sales_3(respond_dict):
    date_start = respond_dict["queryResult"]["outputContexts"][0]["parameters"]["date_start.original"]
    date_stop = respond_dict["queryResult"]["outputContexts"][0]["parameters"]["date_stop.original"]
    answer_str = f"ยอดขายระหว่างวันที่ {date_start} ถึง วันที่ {date_stop} คือ {total_sales_between_date(df, start = date_start, stop = date_stop):,.2f} บาทจ้า"
    user_id = respond_dict['originalDetectIntentRequest']['payload']['data']['source']['userId']
    # Generate Graph
    image_name = each_total_sales_between_date(df, date_start, date_stop)
    # Response Text
    line_bot_api.push_message(user_id, ImageSendMessage(
        original_content_url=f"{config['url']}/static/{image_name}",
        preview_image_url=f"{config['url']}/static/{image_name}"
    ))
    return answer_str

def sales_4(respond_dict):
    days = float(respond_dict["queryResult"]["outputContexts"][0]["parameters"]["days.original"])
    store1 = str(respond_dict["queryResult"]["outputContexts"][0]["parameters"]["store1.original"])
    store2 = str(respond_dict["queryResult"]["outputContexts"][0]["parameters"]["store2.original"])
    answer_str = f"ยอดขายย้อนหลัง {days:.0f} วัน ของร้านที่ {store1} คือ {total_previous_sales(df, days = days, store = store1):,.2f} และ {store2} คือ {total_previous_sales(df, days = days, store = store2):,.2f} บาทจ้า"
    print(answer_str)
    user_id = respond_dict['originalDetectIntentRequest']['payload']['data']['source']['userId']
    # Generate Graph
    image_name = each_previous_sales(df, days, store = [store1,store2])
    # Response
    line_bot_api.push_message(user_id, ImageSendMessage(
        original_content_url=f"{config['url']}/static/{image_name}",
        preview_image_url=f"{config['url']}/static/{image_name}"
    ))
    return answer_str

def sales_5(respond_dict):
    date = respond_dict["queryResult"]["outputContexts"][0]["parameters"]["date.original"]
    store = respond_dict["queryResult"]["outputContexts"][0]["parameters"]["store.original"]
    answer_str = f"ยอดขายวันที่ {date} ของร้าน {store} คือ {total_sales_on_date(df, date = date, store = [store]):,.2f} บาทจ้า"
    return answer_str

def sales_6(respond_dict):
    date = respond_dict["queryResult"]["outputContexts"][0]["parameters"]["date.original"]
    store1 = respond_dict["queryResult"]["outputContexts"][0]["parameters"]["store1.original"]
    store2= respond_dict["queryResult"]["outputContexts"][0]["parameters"]["store2.original"]
    answer_str = f"ยอดขายวันที่ {date} ของร้าน {store1} คือ {total_sales_on_date(df, date = date, store = [store1]):,.2f} และ {store2} คือ {total_sales_on_date(df, date = date, store = [store2]):,.2f} บาทจ้า"
    return answer_str

def sales_7(respond_dict):
    date_start = respond_dict["queryResult"]["outputContexts"][0]["parameters"]["date_start.original"]
    date_stop = respond_dict["queryResult"]["outputContexts"][0]["parameters"]["date_stop.original"]
    store = respond_dict["queryResult"]["outputContexts"][0]["parameters"]["store.original"]
    
    answer_str = f"ยอดขายระหว่างวันที่ {date_start} ถึง วันที่ {date_stop} ของร้าน {store} คือ {total_sales_between_date(df, start = date_start, stop = date_stop, store=store):,.2f} บาทจ้า"
    user_id = respond_dict['originalDetectIntentRequest']['payload']['data']['source']['userId']
    # Generate Graph
    image_name = each_total_sales_between_date(df, date_start, date_stop,[store])
    # Response Text
    line_bot_api.push_message(user_id, ImageSendMessage(
        original_content_url=f"{config['url']}/static/{image_name}",
        preview_image_url=f"{config['url']}/static/{image_name}"
    ))
    return answer_str

def sales_8(respond_dict):
    date_start = respond_dict["queryResult"]["outputContexts"][0]["parameters"]["date_start.original"]
    date_stop = respond_dict["queryResult"]["outputContexts"][0]["parameters"]["date_stop.original"]
    store1 = respond_dict["queryResult"]["outputContexts"][0]["parameters"]["store1.original"]
    store2 = respond_dict["queryResult"]["outputContexts"][0]["parameters"]["store2.original"]

    answer_str = f"ยอดขายระหว่างวันที่ {date_start} ถึง วันที่ {date_stop} ของร้าน {store1} คือ {total_sales_between_date(df, start = date_start, stop = date_stop, store=store1):,.2f} และ ร้าน {store2} คือ {total_sales_between_date(df, start = date_start, stop = date_stop, store=store2):,.2f} บาทจ้า"
    user_id = respond_dict['originalDetectIntentRequest']['payload']['data']['source']['userId']
    # Generate Graph
    image_name = each_total_sales_between_date(df, date_start, date_stop,[store1,store2])
    # Response Text
    line_bot_api.push_message(user_id, ImageSendMessage(
        original_content_url=f"{config['url']}/static/{image_name}",
        preview_image_url=f"{config['url']}/static/{image_name}"
    ))
    return answer_str

def sales_9(respond_dict):
    days = float(respond_dict["queryResult"]["outputContexts"][0]["parameters"]["days.original"])
    store = str(respond_dict["queryResult"]["outputContexts"][0]["parameters"]["store.original"])
    answer_str = f"ยอดขายย้อนหลัง {days:.0f} วัน ของร้านที่ {store} คือ {total_previous_sales(df, days = days, store = store):,.2f} บาทจ้า"
    print(answer_str)
    user_id = respond_dict['originalDetectIntentRequest']['payload']['data']['source']['userId']
    # Generate Graph
    image_name = each_previous_sales(df, days, store = [store])
    # Response
    reply_token = respond_dict['originalDetectIntentRequest']['payload']['data']['replyToken']
    line_bot_api.reply_message(
        reply_token,
        TextSendMessage(text=answer_str)
    )
    line_bot_api.push_message(user_id, ImageSendMessage(
        original_content_url=f"{config['url']}/static/{image_name}",
        preview_image_url=f"{config['url']}/static/{image_name}"
    ))
    return answer_str

def product_1(respond_dict):
    numbers = int(respond_dict["queryResult"]["outputContexts"][0]["parameters"]["numbers.original"])
    user_id = respond_dict['originalDetectIntentRequest']['payload']['data']['source']['userId']
    answer_str = ""
    image_name = popular_item(df, numbers)
    line_bot_api.push_message(user_id, ImageSendMessage(
        original_content_url=f"{config['url']}/static/{image_name}",
        preview_image_url=f"{config['url']}/static/{image_name}"
    ))
    return answer_str

def product_2(respond_dict):
    numbers = int(respond_dict["queryResult"]["outputContexts"][0]["parameters"]["numbers.original"])
    store = int(respond_dict["queryResult"]["outputContexts"][0]["parameters"]["store.original"])

    user_id = respond_dict['originalDetectIntentRequest']['payload']['data']['source']['userId']
    answer_str = ""
    image_name = popular_item(df, numbers,store=[store])
    line_bot_api.push_message(user_id, ImageSendMessage(
        original_content_url=f"{config['url']}/static/{image_name}",
        preview_image_url=f"{config['url']}/static/{image_name}"
    ))
    return answer_str

def product_3(respond_dict):
    # print('Start apriori')
    # check_prod = df['PROD_CODE'].value_counts().to_frame()
    # prod_list = check_prod[check_prod['PROD_CODE']>= 100].index
    # df_fillter = df[df['PROD_CODE'].isin(prod_list)]
    # df_pv = df_fillter.pivot_table(index='BASKET_ID' , columns = 'PROD_CODE' , values= 'QUANTITY' , aggfunc='sum').fillna(0)
    # for col in df_pv.columns:
    #     df_pv[col] = df_pv[col].apply(lambda x:1 if x > 0 else 0 )
    # frq_items = apriori(df_pv, min_support = 0.01, use_colnames = True)   
    # rules = association_rules(frq_items, metric ="lift", min_threshold = 1) 
    # rules = rules.sort_values(['support', 'lift'], ascending =[False, False])
    # antecedents = [''.join(list(x)) for x in rules["antecedents"]]
    # consequents = [''.join(list(x)) for x in rules["consequents"]]
    answer_str = f'''สินค้าที่ขายคู่กันบ่อย 5 อันดับแรก คือ
1. PRD0903678 กับ PRD0903052
2. PRD0903052 กับ PRD0903678
3. PRD0903052 กับ PRD0904358
4. PRD0904358 กับ PRD0903052
5. PRD0903052 กับ PRD0901265
    '''
    reply_token = respond_dict['originalDetectIntentRequest']['payload']['data']['replyToken']
    line_bot_api.reply_message(
        reply_token,
        TextSendMessage(text=answer_str)
    )
    return answer_str

def customer_1(respond_dict):
    answer_str = f"จำนวนลูกค้าทั้งหมด คือ {number_of_customer(df)}"
    return answer_str

def customer_2(respond_dict):
    store = int(respond_dict["queryResult"]["outputContexts"][0]["parameters"]["store.original"])
    answer_str = f"จำนวนลูกค้าของร้านที่ {store} คือ {number_of_customer(df,[store])}"
    return answer_str

def item_1(respond_dict):
    k = int(respond_dict["queryResult"]["outputContexts"][0]["parameters"]["k.original"])
    prod_spend_k, n_prod_spend_k, value_spend_k, total_spend = item_generated_percentage_income(df, k = k)
    answer_str = f"สินค้าที่ทำเงิน {k}% มีจำนวน {n_prod_spend_k:,.0f} คิดเป็นรายได้ {value_spend_k:,.0f} จากรายได้ทั้งหมด {total_spend:,.0f} คือ {' '.join(prod_spend_k[0:5])} ..."
    return answer_str

def item_2(respond_dict):
    k = int(respond_dict["queryResult"]["outputContexts"][0]["parameters"]["k.original"])
    store = int(respond_dict["queryResult"]["outputContexts"][0]["parameters"]["store.original"])
    prod_spend_k, n_prod_spend_k, value_spend_k, total_spend = item_generated_percentage_income(df, k = k, store = [store])
    answer_str = f"สินค้าที่ทำเงิน {k}% มีจำนวน {n_prod_spend_k:,.0f} คิดเป็นรายได้ {value_spend_k:,.0f} จากรายได้ทั้งหมด {total_spend:,.0f} คือ {' '.join(prod_spend_k[0:5])} ..."
    return answer_str

def total_sales_on_date(df, date, store = None):

    ''' ยอดขายรวมของวันที่ date (ของร้านที่ store)'''

    if store is None:
        selected_df = df[df['SHOP_DATE'] == date]
        sales = sum(selected_df['SPEND'])
    elif len(store) == 1:
        selected_df = df[(df['SHOP_DATE'] == date) & (df['STORE_CODE'] == 'STORE0000'+str(store))]
        sales = sum(selected_df['SPEND'])
    elif len(store) == 2:
        selected_df_1 = df[(df['SHOP_DATE'] == date) 
                         & (df['STORE_CODE'] == 'STORE0000'+str(store[0]))]
        
        selected_df_2 = df[(df['SHOP_DATE'] == date) 
                         & (df['STORE_CODE'] == 'STORE0000'+str(store[1]))]
        
        sales_each_1 = selected_df_1.groupby('SHOP_DATE')[['SPEND']].sum().reset_index()
        sales_each_2 = selected_df_2.groupby('SHOP_DATE')[['SPEND']].sum().reset_index()
        sales = sales_each_1.merge(sales_each_2, 
                                        left_on = 'SHOP_DATE', right_on = 'SHOP_DATE',
                                       suffixes = ('_store'+str(store[0]), '_store'+str(store[1])))
        
        
        #plot
        fig, ax = plt.subplots(figsize=(20, 10))
        labels = list(sales_each_1['SHOP_DATE'].dt.date)   
        x = np.arange(len(labels))  # the label locations
        width = 0.4  # the width of the bars
        rects1 = ax.bar(x - (1*width/2), sales_each_1['SPEND'], width, label='Total Sales store'+str(store[0]))
        rects2 = ax.bar(x + (1*width/2), sales_each_2['SPEND'], width, label='Total Sales store'+str(store[1]))
        rects = (rects1, rects2)
        
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Sales',fontsize=18)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=18)
        ax.legend(fontsize=18)
        ax.tick_params(axis="y", labelsize=18)
    
        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = round(rect.get_height(), 1)
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            size=20)
        if store is not None:
            if len(store) == 2:
                for i in rects:
                    autolabel(i)
            else:
                autolabel(rects)
        else:
            autolabel(rects)
        fig.tight_layout()
        image_name = str(time.time_ns()) + '.png'
        plt.savefig(image_name)    
    
    return sales

def total_sales_between_date(df, start, stop, store = None):

    ''' ยอดขายรวมระหว่างวันที่ start ถึง stop ของร้านที่ store'''

    if store is None:
        selected_df = df[(df['SHOP_DATE'] >= start) & (df['SHOP_DATE'] <= stop)]
        sales = sum(selected_df['SPEND'])
    else:
        selected_df = df[((df['SHOP_DATE'] >= start) & (df['SHOP_DATE'] <= stop)) & (df['STORE_CODE'] == 'STORE0000'+str(store))]
        sales = sum(selected_df['SPEND'])

    return sales

def total_previous_sales(df, days = 1, store = None):

    ''' ยอดขายรวมย้อนหลัง days วัน ของร้านที่ store'''

    date = max(df['SHOP_DATE']) - datetime.timedelta(days = days)
    if store is None:
        selected_df = df[df['SHOP_DATE'] >= date]
        sales = sum(selected_df['SPEND'])
    else:
        selected_df = df[(df['SHOP_DATE'] >= date) & (df['STORE_CODE'] == 'STORE0000'+str(store))]
        sales = sum(selected_df['SPEND'])

    return sales

def each_total_sales_between_date(df, start, stop, store = None):

    ''' ยอดขายแต่ละวันระหว่างวันที่ start ถึง stop (ของร้านที่ ['store1','store2'])'''
    
    if store is None:
        selected_df = df[(df['SHOP_DATE'] >= start) & (df['SHOP_DATE'] <= stop)]
        sales_each = selected_df.groupby('SHOP_DATE')[['SPEND']].sum().reset_index()
        
        #plot
        fig, ax = plt.subplots(figsize=(20, 10))
        labels = list(sales_each['SHOP_DATE'].dt.date)   
        x = np.arange(len(labels))  # the label locations
        width = 0.6  # the width of the bars
        rects = ax.bar(x - (0*width/2), sales_each['SPEND'], width, label='Total Sales')
        ax.set(ylim=(min(sales_each['SPEND']-200), max(sales_each['SPEND']+200)))

    elif len(store) == 1:
        selected_df = df[(df['SHOP_DATE'] >= start) 
                          & (df['SHOP_DATE'] <= stop) 
                          & (df['STORE_CODE'] == 'STORE0000'+str(store[0]))]
        sales_each = selected_df.groupby('SHOP_DATE')[['SPEND']].sum().reset_index()
        
        #plot
        fig, ax = plt.subplots(figsize=(20, 10))
        labels = list(sales_each['SHOP_DATE'].dt.date)   
        x = np.arange(len(labels))  # the label locations
        width = 0.6  # the width of the bars
        rects = ax.bar(x - (0*width/2), sales_each['SPEND'], width, label='Total Sales')
        ax.set(ylim=(min(sales_each['SPEND']-200), max(sales_each['SPEND']+200)))
    
    elif len(store) == 2:      
        selected_df_1 = df[(df['SHOP_DATE'] >= start) 
                           & (df['SHOP_DATE'] <= stop) 
                           & (df['STORE_CODE'] == 'STORE0000'+str(store[0]))]
        
        selected_df_2 = df[(df['SHOP_DATE'] >= start) 
                           & (df['SHOP_DATE'] <= stop)
                           & (df['STORE_CODE'] == 'STORE0000'+str(store[1]))]
        
        sales_each_1 = selected_df_1.groupby('SHOP_DATE')[['SPEND']].sum().reset_index()
        sales_each_2 = selected_df_2.groupby('SHOP_DATE')[['SPEND']].sum().reset_index()
        sales_each = sales_each_1.merge(sales_each_2, 
                                        left_on = 'SHOP_DATE', right_on = 'SHOP_DATE',
                                       suffixes = ('_store'+str(store[0]), '_store'+str(store[1])))

        #plot
        fig, ax = plt.subplots(figsize=(20, 10))
        labels = list(sales_each_1['SHOP_DATE'].dt.date)   
        x = np.arange(len(labels))  # the label locations
        width = 0.4  # the width of the bars
        rects1 = ax.bar(x - (1*width/2), sales_each_1['SPEND'], width, label='Total Sales store'+str(store[0]))
        rects2 = ax.bar(x + (1*width/2), sales_each_2['SPEND'], width, label='Total Sales store'+str(store[1]))
        rects = (rects1, rects2)
        
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Sales',fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=18)
    ax.legend(fontsize=18)
    ax.tick_params(axis="y", labelsize=18)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = round(rect.get_height(), 1)
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        size=20)
    if store is not None:
        if len(store) == 2:
            for i in rects:
                autolabel(i)
        else:
            autolabel(rects)
    else:
        autolabel(rects)
    fig.tight_layout()
    # plt.show()    
    image_name = str(time.time_ns()) + '.png'
    plt.savefig('static/'+image_name)
    return image_name

def each_previous_sales(df, days = 1, store = None):
    
    ''' ยอดขายแต่ละวันย้อนหลัง days วัน (ของร้านที่ ['store1','store2'])'''
    date = max(df['SHOP_DATE']) - datetime.timedelta(days = days)

    if store is None:
        selected_df = df[df['SHOP_DATE'] >= date]
        
        sales_each = selected_df.groupby('SHOP_DATE')[['SPEND']].sum().reset_index()
        
        #plot
        fig, ax = plt.subplots(figsize=(20, 10))
        labels = list(sales_each['SHOP_DATE'].dt.date)   
        x = np.arange(len(labels))  # the label locations
        width = 0.6  # the width of the bars
        rects = ax.bar(x - (0*width/2), sales_each['SPEND'], width, label='Total Sales')
        ax.set(ylim=(min(sales_each['SPEND']-200), max(sales_each['SPEND']+200)))

    elif len(store) == 1:
        selected_df = df[(df['SHOP_DATE'] >= date) & (df['STORE_CODE'] == 'STORE0000'+str(store[0]))]
        sales_each = selected_df.groupby('SHOP_DATE')[['SPEND']].sum().reset_index()
        
        #plot
        fig, ax = plt.subplots(figsize=(20, 10))
        labels = list(sales_each['SHOP_DATE'].dt.date)   
        x = np.arange(len(labels))  # the label locations
        width = 0.6  # the width of the bars
        rects = ax.bar(x - (0*width/2), sales_each['SPEND'], width, label='Total Sales')
        ax.set(ylim=(min(sales_each['SPEND']-200), max(sales_each['SPEND']+200)))
    
    elif len(store) == 2:      
        selected_df_1 = df[(df['SHOP_DATE'] >= date) 
                         & (df['STORE_CODE'] == 'STORE0000'+str(store[0]))]
        
        selected_df_2 = df[(df['SHOP_DATE'] >= date) 
                         & (df['STORE_CODE'] == 'STORE0000'+str(store[1]))]
        
        sales_each_1 = selected_df_1.groupby('SHOP_DATE')[['SPEND']].sum().reset_index()
        sales_each_2 = selected_df_2.groupby('SHOP_DATE')[['SPEND']].sum().reset_index()
        sales_each = sales_each_1.merge(sales_each_2, 
                                        left_on = 'SHOP_DATE', right_on = 'SHOP_DATE',
                                       suffixes = ('_store'+str(store[0]), '_store'+str(store[1])))
        
        #plot
        fig, ax = plt.subplots(figsize=(20, 10))
        labels = list(sales_each_1['SHOP_DATE'].dt.date)   
        x = np.arange(len(labels))  # the label locations
        width = 0.4  # the width of the bars
        rects1 = ax.bar(x - (1*width/2), sales_each_1['SPEND'], width, label='Total Sales store'+str(store[0]))
        rects2 = ax.bar(x + (1*width/2), sales_each_2['SPEND'], width, label='Total Sales store'+str(store[1]))
        rects = (rects1, rects2)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Sales',fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=18)
    ax.legend(fontsize=18)
    ax.tick_params(axis="y", labelsize=18)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = round(rect.get_height(), 1)
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        size=20)
    if store is not None:
        if len(store) == 2:
            for i in rects:
                autolabel(i)
        else:
            autolabel(rects)
    else:
        autolabel(rects)
    fig.tight_layout()
    image_name = str(time.time_ns()) + '.png'
    plt.savefig('static/'+image_name)    
        
    return image_name

def popular_item(df, n, store = None):

    ''' สินค้าไหนถูกซื้อบ่อยสุด n อันดับแรก'''
    
    if store is None:
        #select data
        pivot_df = df[['BASKET_ID', 'PROD_CODE']].groupby(['PROD_CODE','BASKET_ID']).size().unstack(fill_value = 0)
        support = pivot_df.sum(axis = 1)/df['BASKET_ID'].nunique()
        n_rank_support = support.sort_values(ascending = False)[0:n]
        n_rank_support.loc['else'] = 1 - n_rank_support.sum()
        
        #plot
        labels = n_rank_support.reset_index()['PROD_CODE']
        sizes = list(n_rank_support)
        fig1, ax1 = plt.subplots(figsize = (10,6))
        ax1.pie(sizes, explode = n_rank_support.reset_index().groupby('PROD_CODE').size()/20, labels=labels, autopct='%1.1f%%',
        startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        image_name = str(time.time_ns()) + '.png'
        plt.savefig('static/'+image_name) 
        
    else:
        #select data
        selected_df = df[(df['STORE_CODE'] == 'STORE0000'+str(store[0]))]
        sales_each = selected_df.groupby('SHOP_DATE')[['SPEND']].sum().reset_index()
        
        pivot_df = selected_df[['BASKET_ID', 'PROD_CODE']].groupby(['PROD_CODE','BASKET_ID']).size().unstack(fill_value = 0)
        support = pivot_df.sum(axis = 1)/selected_df['BASKET_ID'].nunique()
        n_rank_support = support.sort_values(ascending = False)[0:n]
        n_rank_support.loc['else'] = 1 - n_rank_support.sum()
        
        #plot
        labels = n_rank_support.reset_index()['PROD_CODE']
        sizes = list(n_rank_support)
        fig1, ax1 = plt.subplots(figsize = (10,6))
        ax1.pie(sizes, explode = n_rank_support.reset_index().groupby('PROD_CODE').size()/20, labels=labels, autopct='%1.1f%%',
        startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        image_name = str(time.time_ns()) + '.png'
        plt.savefig('static/'+image_name) 
        
    return image_name

def item_generated_percentage_income(df, k = 0.3, store = None):
    
    ''' สินค้า n อันดับแรกที่ทำเงิน k*100% จากยอดขายทั้งหมด '''
    
    if store is None:
        prod_spend_ratio = df[['PROD_CODE', 'SPEND']].groupby('PROD_CODE').sum()/df['SPEND'].sum()
        index = []
        sum_v = 0

        for i,v in enumerate(prod_spend_ratio.reset_index().sort_values('SPEND', ascending = False)['SPEND']):
            sum_v += v
            if sum_v >= k:
                break
            else:
                index.append(i)
        prod_spend_k = prod_spend_ratio.reset_index().sort_values('SPEND', ascending = False).iloc[index,0]
        value_spend_k = (prod_spend_ratio.reset_index().sort_values('SPEND', ascending = False).iloc[index,1]*(df['SPEND'].sum())).sum()
        total_spend = df['SPEND'].sum()
        n_prod_spend_k = len(prod_spend_k)
    else:
        selected_df = df[(df['STORE_CODE'] == 'STORE0000'+str(store[0]))]
        prod_spend_ratio = selected_df[['PROD_CODE', 'SPEND']].groupby('PROD_CODE').sum()/df['SPEND'].sum()
        index = []
        sum_v = 0

        for i,v in enumerate(prod_spend_ratio.reset_index().sort_values('SPEND', ascending = False)['SPEND']):
            sum_v += v
            if sum_v >= k:
                break
            else:
                index.append(i)
        prod_spend_k = list(prod_spend_ratio.reset_index().sort_values('SPEND', ascending = False).iloc[index,0])
        value_spend_k = (prod_spend_ratio.reset_index().sort_values('SPEND', ascending = False).iloc[index,1]*(selected_df['SPEND'].sum())).sum()
        total_spend = selected_df['SPEND'].sum()
        n_prod_spend_k = len(prod_spend_k)
    
    return prod_spend_k, n_prod_spend_k, value_spend_k, total_spend

def number_of_customer(df, store = None):
    
    ''' จำนวนลูกค้า '''
    
    if store is None:
        selected_df = df
    else:
        selected_df = df[(df['STORE_CODE'] == 'STORE0000'+str(store[0]))]
    n_customer = selected_df['CUST_CODE'].nunique()
    
    return n_customer

@app.route('/img', methods=['GET'])
def view_image():
    # generate_img("test.jpg"); #save inside static folder
    return '<img src=' + url_for('static',filename='test.png') + '>'

# ----- Main Flask -----
if __name__ == '__main__':
    #import and prep
    df = pd.read_csv('supermarket_data.csv')
    df['SHOP_DATE'] = pd.to_datetime(df['SHOP_DATE'], format = '%Y%m%d')
    port = int(os.getenv('PORT', 5000))
    print("Starting app on port %d" % port)
    app.run(debug=True, port=port, host='0.0.0.0', threaded=True)