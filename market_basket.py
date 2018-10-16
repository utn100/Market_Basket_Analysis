import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objs as go
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx
import numpy as np

app = dash.Dash()
server = app.server
app.css.append_css({'external_url':'https://codepen.io/chriddyp/pen/bWLwgP.css'})
def encode_image(image_file):
    encoded = base64.b64encode(open(image_file, 'rb').read())
    return 'data:image/png;base64,{}'.format(encoded.decode())

df = pd.read_csv("Data/BreadBasket_DMS.csv")
# Drop row of NONE items
df.drop(df[df['Item']=='NONE'].index,inplace=True)
# Create month and hour columns
df['Month'] = df['Date'].apply(lambda x:x.split("-")[1])
df['Hour'] = df['Time'].apply(lambda x:int(str(pd.to_datetime(x).round('H')).split(" ")[1].split(":")[0]))
# Get counts of each item per hour for each months
byitem=df.groupby(["Month","Hour",'Item']).size().reset_index().sort_values(by="Hour")
byitem.rename(columns={0:'Total'},inplace=True)

df_agg = byitem.groupby(['Month','Hour','Item']).agg({'Total':sum})
g = df_agg['Total'].groupby(level=[0,1], group_keys=False)
topitem_hour=g.nlargest(5).reset_index()
colors=['#848484','#3d567f','#70b578','#8923c4','#ffd77a','#e6b2ff',
              '#ff009d','#7f8400','#a3f2ff','#c1093a','#6b7cc6','#fc4cff','#0f8c01',
              '#ff8a00','#aeffa5','#cc8a3b','#ff7a9d','#ad4747','#f9ffb2','#ff0202',
              '#00fff2','#3a51ff','#73f4d8','#fffa00','#ff6600','#00b2ff','#cb00ff',
              '#59ed28','#c7fce9','#d2d6d1','#d1cd9c','#9b6a47','#ff7777','#499b5a',
              '#c49c68','#349dad','#b200ff','#7c0048','#289bc9']
month_dict = {'01':'January','02':'February','03':'March','04':'April','10':'October','11':'November','12':'December'}

items = []
for i in df['Transaction'].unique():
    itemlist = list(set(df[df["Transaction"]==i]['Item']))
    if len(itemlist) > 0:
        items.append(itemlist)

tran_encoder = TransactionEncoder()
oht_ary = tran_encoder.fit(items).transform(items)
dataframe = pd.DataFrame(oht_ary, columns=tran_encoder.columns_)
print (dataframe.head())
support_thresholds = [0.005,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
lift_thresholds = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

app.layout = html.Div([
                html.Div([
                html.Div([
                dcc.Dropdown(id='items',
                            options=[{'label':i,'value':i} for i in list(df['Item'].unique())],
                            value = ['Coffee','Bread'],
                            multi= True)
                ]),
                dcc.Graph(id='itemgraph')]),
                html.Div([
                html.Div([
                dcc.Dropdown(id='months',
                            options=[{'label':month_dict[i],'value':i} for i in list(df['Month'].unique())],
                            value = '01')
                ]),
                dcc.Graph(id='composition-graph')]),
                html.Div([
                dcc.Dropdown(
                    id='support',
                    options=[{'label': str(i), 'value': i} for i in support_thresholds],
                    value=0.02
                    )], style={'width': '48%', 'display': 'inline-block'}),

                html.Div([
                dcc.Dropdown(
                id='lift',
                options=[{'label': str(i), 'value': i} for i in lift_thresholds],
                value=1
                )],style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),

                dcc.Graph(id="network"),
                html.Div([
                html.Table(id='table')
                ])], style={'padding':10})

@app.callback(Output('itemgraph', 'figure'),
            [Input('items', 'value')])
def update_itemgraph(item_toplot):
    traces=[go.Scatter(
    x = byitem[byitem['Item']==item]['Hour'].unique(),
    y = byitem[byitem['Item']==item].groupby("Hour")['Total'].mean().round(1),
    mode = 'markers+lines',
    name = item) for item in item_toplot]

    return {'data':traces,
            'layout':go.Layout(
                title = 'Average number of items sold at each hour')}

@app.callback(Output('composition-graph', 'figure'),
            [Input('months', 'value')])
def update_compositiongraph(month):
    month_item = topitem_hour[topitem_hour['Month']==month]['Item'].unique()
    byitem_month = byitem[byitem['Month']==month][['Hour','Item','Total']]
    byitem_month_pivot=pd.pivot_table(byitem_month,index='Hour',values='Total',columns='Item',fill_value=0,aggfunc='sum')[month_item]

    cols=list(byitem_month_pivot.sum(axis=0).reset_index().sort_values(by=0)['Item'])
    col_dicts = {}
    for col in cols:
        col_dicts[col] = colors[cols.index(col)]

    data = [go.Bar(x=byitem_month_pivot.index,y=byitem_month_pivot[col],name=col,marker={'color':col_dicts[col]}) for col in cols]
    layout = go.Layout(title="Number of Items sold in " + month_dict[month],barmode='stack')
    return {'data':data,'layout':layout}



@app.callback(
    Output('network', 'figure'),
    [Input('support', 'value'),
     Input('lift', 'value')])

def update_networkgraph(support_min, lift_min):
    frequent_itemsets = apriori(dataframe, use_colnames=True, min_support=support_min)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=lift_min)
    if rules.empty:
        return {'data':[],'layout':go.Layout(
          title='<br>No rules satisfy the chosen conditions',
          titlefont=dict(size=20))}
    else:

        rules.sort_values(by='lift',ascending=False,inplace=True)
        rules_to_show=len(rules)

        colormap = plt.get_cmap('Reds')
        colorlist=colormap(rules['lift']**2/(1 + rules['lift'] ** 2))
        colorlist=["rgb"+str(tuple(i[0:3])) for i in colorlist]

        def PointsInCircum(r,n):
            import math
            return [math.cos(2*math.pi/n*x)*r for x in np.arange(0,n+1)], [math.sin(2*math.pi/n*x)*r for x in np.arange(0,n+1)]
        cir_x,cir_y = PointsInCircum(40,rules_to_show*2)[0],PointsInCircum(40,rules_to_show*2)[1]

        G1 = nx.DiGraph()
        rule_lists=[]
        for i in range(rules_to_show):
            rule_lists.append('R'+str(i))

        for i in range (rules_to_show):
            G1.add_nodes_from(["R"+str(i)])
            for a in rules.iloc[i]['antecedents']:
                G1.add_nodes_from([a])
                G1.add_edge(a, "R"+str(i))
            for c in rules.iloc[i]['consequents']:
                G1.add_nodes_from([c])
                G1.add_edge("R"+str(i), c)
        sizes = []
        color_map = []
        j=0
        pos = {}
        texts = []
        index1 = -(int(rules_to_show/5)+1)
        index2 = int(rules_to_show/5+1)
        for node in G1:
            if node in rule_lists:
                color_map.append(colorlist[j])
                sizes.append(rules['confidence'].iloc[j]*50+5)
                pos[node]= [cir_x[index1],cir_y[index1]]
                texts.append(node+': '+ list(rules.iloc[j]['antecedents'])[0] + ' --> ' + list(rules.iloc[j]['consequents'])[0]
                    + '<br>'+'support = '+str(round(rules['support'].iloc[j],3))
                    + '<br>'+'confidence = '+str(round(rules['confidence'].iloc[j],2))
                    + '<br>'+'lift = '+str(round(rules['lift'].iloc[j],2)))
                index1 -= 1
                j += 1
            else:
                color_map.append('#f766ff')
                sizes.append(20)
                pos[node]= [cir_x[index2],cir_y[index2]]
                texts.append(node)
                index2 +=1

        x0s = []
        x1s = []
        y0s = []
        y1s = []
        for edge in G1.edges():
            x0s.append(pos[edge[0]][0])
            x1s.append(pos[edge[1]][0])
            y0s.append(pos[edge[0]][1])
            y1s.append(pos[edge[1]][1])

        node_trace = go.Scatter(
                x=[],
                y=[],
                text=texts,
                mode='markers',
                hoverinfo='text',
                marker=dict(
                    color=color_map,
                    size=sizes,
                    line=dict(width=2,color="black")))
        for node in G1:
            node_trace['x'].append(pos[node][0])
            node_trace['y'].append(pos[node][1])


        fig = {'data':[node_trace],
              'layout':go.Layout(
                title='<br>Network graph of market basket analysis - Association rule learning<br>',
                titlefont=dict(size=16),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    showarrow=True,
                    arrowhead=3,
                    arrowsize=1,
                    arrowwidth=1.2,
                    arrowcolor='#808080',
                    ax=x0s[i], ay=y0s[i], axref='x', ayref='y',
                    x=x1s[i], y=y1s[i], xref='x', yref='y',
                    yshift=-3) for i in range(len(x0s))],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))}

        return fig

@app.callback(Output('table','children'),
            [Input('support', 'value'),
             Input('lift', 'value')])
def update_table(support_min, lift_min):
    max_rows=50
    frequent_itemsets = apriori(dataframe, use_colnames=True, min_support=support_min)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=lift_min)
    rules['antecedents'] = rules['antecedents'].apply(lambda x:list(x)[0])
    rules['consequents']= rules['consequents'].apply(lambda x:list(x)[0])
    for col in ['antecedent support','consequent support', 'support', 'confidence', 'lift', 'leverage','conviction']:
        rules[col] = round(rules[col],3)
    columns = ['antecedents','consequents','support', 'confidence', 'lift']
    return html.Table([html.Tr([html.Th(col) for col in columns])] +
                        [html.Tr([html.Td(rules.iloc[i][col]) for col in columns]) for i in range(min(len(rules), max_rows))])

if __name__ == '__main__':
    app.run_server()
