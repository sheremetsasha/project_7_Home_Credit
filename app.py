import pickle
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import json
import shap
from shap import TreeExplainer, Explanation
#from shap.plots import waterfall
from flask import Flask, render_template, request
import random
from io import BytesIO
import base64


# create flask app
app = Flask(__name__)

#--------------------------------------------
# converters from plot to html, json, 
#--------------------------------------------
def shap_to_html(fig):
    return fig.html()

# convert matplotlib figure to html image
def fig_to_html(fig):
    b64 = fig_to_base64png(fig)
    return f"<img src=\"data:image/png;base64,{b64}\">"

def px_to_html(pxplot):
    htmlid = random.randint(0, 1e4)
    plotjson = json.dumps(pxplot, cls = plotly.utils.PlotlyJSONEncoder)
    return f"""
        <div id="chart{htmlid}" class="pxchart"></div>
        <script type='text/javascript'>
            Plotly.plot('chart{htmlid}',{plotjson},{{}});
        </script>
    """

def fig_to_base64png(fig):
    b = BytesIO()
    plt.savefig(b, format="png", bbox_inches='tight')
    b64 = base64.b64encode(b.getvalue()).decode()
    return b64

#-------------------------------------------
# load the model from disk
#-------------------------------------------
print("Loading model...")
with open('finalized_model.sav', 'rb') as f:
    loaded_model = pickle.load(f)

#-------------------------------------------
# load the data for prediction
#-------------------------------------------
print("Loading data...")
with open('data_clients.pickle', 'rb') as f:
    data = pickle.load(f)

#-------------------------------------------
#prediction
#-------------------------------------------
print("Model prediction...")
y_pred_lgbm = loaded_model.predict(data)

#-------------------------------------------
#local features with SHAP
#-------------------------------------------
print("Calculating local feture importnace...")

explainer_client = TreeExplainer(loaded_model)
shap_values_client = explainer_client(data)

exp = Explanation(shap_values_client.values[:,:,1], 
                  shap_values_client.base_values[:,1], 
                  data = data.values, 
                  feature_names = data.columns)

#Force plot
def get_shap_force_plot(i, num_feat):
    return shap.force_plot(explainer_client.expected_value[0], exp.values[i], 
                features = data.iloc[i], feature_names = data.columns)

#Waterfall plot
def get_waterfall_plot(exp, i, max_display = 6, show = False):
    fig = plt.figure()
    shap.plots.bar(exp[i], max_display = max_display, show = show)
    plt.gcf().set_size_inches(4.75,4)
    plt.title("Local feature importance")
    return fig

#Averaged local feature importance
def plot_same_customers(data, x, y, title):
    fig = px.bar(data, x = x, y = y, title = title)
    fig.update_layout(yaxis = {'autorange': 'reversed'}, width = 550, height = 400, showlegend = False)
    fig.add_traces(go.Scatter(x = round(data[:5].benchmark, 2), 
                          y = data[:5].feature, mode = "markers",
                          marker = dict(symbol = 'star', color = "red",
                                            line = dict(color = 'black', width = 1), size = 7), ))
    fig.update_layout(legend = dict(yanchor = "bottom", y = 0.05, xanchor = "right",x = 0.99))
    fig.update_layout(title = title, title_x = 0.75, title_y = 0.8)
    #fig.update_layout(yaxis_visible = False)
    
    return fig

#--------------------------------------------
#calculating score of the customer
#--------------------------------------------
def get_score(i):    
    score_client = round(np.sum(exp.values[i]),2)    
    return score_client

def get_feature_plot(client, x, y, title):
    fig = px.bar(client, x = x, y = y, title = title)
    fig.update_layout(yaxis = {'autorange': 'reversed'})
    return json.dumps(fig, cls = plotly.utils.PlotlyJSONEncoder)


#----------------------------------------------
#Global features plot
#----------------------------------------------

def feat_import_sns(data, x, y, title):
    fig = plt.figure(figsize = (3,2.5))
    sns.barplot(data = data, x = x, y = y)
    plt.title(title)
    return fig


def plot_feat_import_glob(data, x, y, title):
    fig = px.bar(data, x = x, y = y, title = title)
    fig.update_layout(yaxis = {'autorange': 'reversed'}, width = 550, height = 400)
    fig.update_layout(title = title, title_x = 0.5, title_y = 0.8)
    
    return fig

#----------------------------------------------
#Figures with multi-choice
#----------------------------------------------
#Function for plots
def plot_hist(i, y, xlabel, ylabel1 = 'Counts', ylabel2 = 'Density', 
                  mask1 = y_pred_lgbm == 0, mask2 = y_pred_lgbm == 1, 
                  label1 = 'TARGET = 0', label2 = 'TARGET = 1'):
    
    #figure = plt.figure(figsize = (9,3));
    fig = plt.figure(figsize = (7,2))
    
    plt.subplot(1, 2, 1) # row 1, col 2 index 1
    plt.style.use('fivethirtyeight')
    plt.hist(y, edgecolor = 'k', bins = 20)
    plt.axvline(round(y.iloc[i],2), color = 'k', linestyle = 'dashed', linewidth = 1)
    plt.rcParams.update({'font.size': 8})
    plt.xlabel(xlabel); plt.ylabel(ylabel1);

    plt.subplot(1, 2, 2) # index 2
    sns.kdeplot(y.loc[mask1], label = label1)
    sns.kdeplot(y.loc[mask2], label = label2)
    plt.axvline(round(y.iloc[i],2), color = 'k', linestyle = 'dashed', linewidth = 1)
    plt.rcParams.update({'font.size': 7})
    plt.xlabel(xlabel); plt.ylabel(ylabel2);
    
    
    return fig


def figs_to_select_html(figs):
    html = ""
    rnd = random.randint(0,1000)
    name = f"radio{rnd}"
    srcs = []
    for k, (txt, fig) in enumerate(figs):
        rnd = random.randint(0, 1000)
        htmlid = f"radioid{rnd}"
        srcs.append(fig)
        checked = "checked" if k == 0 else ""
        html += f"""
            <input type="radio" name="{name}" id="{htmlid}" {checked} onclick="changeimg({k})">
            <label for="{htmlid}">{txt}</label>
        """
    html += f"""
    <img id="radioimg" src="data:image/png;base64,{srcs[0]}"/>
    <script>
        var srcs = {json.dumps(srcs)};
        function changeimg(i){{
            document.getElementById("radioimg").src="data:image/png;base64,"+srcs[i];
        }}
    </script>
    """
    return html

# New plots

#-----------------------------------------
#Figure for bi-variant dependence
#-----------------------------------------

labels_2f = {"DAYS_BIRTH": "Age of customers, years", "DAYS_ID_PUBLISH": "Recency, years"},

def two_feat_plot(f, x1, y1, color='score',i = 12, labels = labels_2f):
    
    fig = px.scatter(f, x = x1, y = y1, color = 'score',labels = labels_2f)
    fig.update_layout(width = 550, height = 400, showlegend = False)
    fig.add_trace(go.Scatter(x = [f[x1].iloc[i]],
                        y = [f[y1].iloc[i]],
                        mode = 'markers',
                        marker = dict(symbol = 'star',
                                      color = "red",
                                      line = dict(color = 'black', width = 2),
                                      size = 30), 
                        showlegend = False))
    #fig.data = (fig.data[1],fig.data[0])
    return fig

#==============================================
#==============================================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/<int:i>/")
def api(i):
    #------------------------------------------
    #Data frame with local features importance
    #------------------------------------------
    feat_client = pd.DataFrame()
    feat_data = data.copy()
    feat_client["feature"] = feat_data.columns 
    feat_client["importance"] = exp.values[i]
    # sort by importance
    fc = feat_client.sort_values(by = "importance", ascending = False)
    # make a dict
    obj = {k:v for k,v in zip(list(fc["feature"]), list(fc["importance"]))}
    return json.dumps(obj)

@app.route("/chart/", methods = ["GET"])
def charts():
    i = int(request.args.get("id"))
    
    #----------------------------------------------
    #Local features plot for the customer with SHAP
    #----------------------------------------------
    explainer = shap_to_html(get_shap_force_plot(i, num_feat = 5))
    
    #----------------------------------------------
    #Data frame with local feature importane
    #----------------------------------------------
    feat_client_loc = pd.DataFrame()
    feat_client_loc_sup = data.copy()
    feat_client_loc["feature"] = feat_client_loc_sup.columns 
    feat_client_loc["importance"] = exp.values[i]
    
    
    feat_import = np.zeros(len(data.index))
    for k in range(len(feat_import)):
        feat_import[k] = round(np.sum(exp.values[k,:]),2)
    
    data_tot = data.copy()
    data_tot['score'] = feat_import
    mask_same_score = data_tot['score'] == data_tot["score"].iloc[i]
    
    #----------------------------------------------
    #Selecting 5 best local features
    #----------------------------------------------
    
    #Calculating mean feature importance for selected custmers
    mean_score = np.zeros(len(data.columns))
    for k in range(len(data.columns)):
        mean_score[k] = np.mean(exp.values[mask_same_score, k])
    feat_cust = pd.DataFrame()
    feat_cust["feature"] = data.columns 
    feat_cust["importance"] = exp.values[i]

    #Data frame with avereged feature importance
    feat_client_same_score = pd.DataFrame()
    feat_client_same_score["feature"] = data.columns 
    feat_client_same_score["importance"] = mean_score
    #Adding a column with values for the selected customer
    feat_client_same_score["benchmark"] = feat_cust["importance"]

    #Sorting with descending order
    best_feat_client_same_score = feat_client_same_score.copy()
    best_feat_client_same_score = best_feat_client_same_score.sort_values(by = "importance", ascending = False)
    best_feat_client_same_score
    
    #----------------------------------------------
    #Global features plot for the customer
    #----------------------------------------------
    feat_import_glob = pd.DataFrame()
    feat_import_glob_sup = data.copy()
    feat_import_glob['feature'] = feat_import_glob_sup.columns
    import_glob = loaded_model.feature_importances_
    tot_glob = import_glob.sum()
    feat_import_glob['importance'] = import_glob/tot_glob

    glob_feat_importance = feat_import_glob.sort_values(by = "importance", ascending = False)[:5]
    
    #------------------------------------------------
    #Data for plots with multi-choice
    #------------------------------------------------
    
    y_plt = [-1/365*data['DAYS_BIRTH'], data['AMT_CREDIT'], data['EXT_SOURCE_2'], 
         -1/365*data['DAYS_ID_PUBLISH'], data['PAYMENT_RATE']]
    
    x_label = ['Age, years', 'amount', 'external source 2', 'recency, years', 'payment rate']
    
    options = [('Age', 0), ('Reqested loan', 1), ('External source 2', 2), ('Recency', 3), ('Payment rate', 4)]

    figs = [
        (txt, fig_to_base64png(plot_hist(i, y_plt[idx], xlabel = x_label[idx])))
        for txt, idx in options
    ]
    radioplot = figs_to_select_html(figs)

    twofeat = two_feat_plot(-1/365*data_tot[:3000], "DAYS_BIRTH", 'DAYS_ID_PUBLISH',color = 'score',i = 12, labels = labels_2f)
    twofeat_html = px_to_html(twofeat)
    
    #==============================================
    #==============================================
    #First line of plots
    row1 = [
        #Plot 1
        fig_to_html(get_waterfall_plot(exp, i)),
        
        #Plot 2
        px_to_html(
            plot_same_customers(data = best_feat_client_same_score[:5],
                                x = "importance", y = "feature", 
                                title = 'Averged local feature importance')
        ),
        
        #Plot 3
        px_to_html(
            plot_feat_import_glob(data = glob_feat_importance[:5],
                                x = "importance", y = "feature", 
                                title = "Global features importance")
            ),
        
    ]

    #Line 2
    row2 = [
        #Plot one (multi-varee)
        radioplot,
        #Plot 2 (bi-variant)
        twofeat_html,
    ]
    
    return render_template("charts.html",
                            client_id = data.index[i],
                            result_pred = y_pred_lgbm[i],
                            score_client = get_score(i),
                            charts = [
                                [explainer],
                                row1,
                                row2,
                            ]
    )

if __name__ == "__main__":
    app.run()

