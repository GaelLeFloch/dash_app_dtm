import dash
from dash.dependencies import Input, Output, State, Event
import dash_core_components as dcc
import dash_html_components as html
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
# from flask import send_from_directory
import pandas as pd
import numpy as np
import pickle
import gensim
from gensim import corpora
import re

app = dash.Dash()
app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})
server = app.server

# app.scripts.config.serve_locally = True
# app.css.config.serve_locally = True


def prep_data_to_plot(model, t2i, corpus, vis, topic):
    words = [wo for we, wo in model.show_topic(topic, 0, 20)]
    dd = {}
    for wo in words:
        dd[wo] = vis[periodes[0]][1][:, t2i[wo]]

    dd = pd.DataFrame(dd).transpose().sort_values(topic)
    xx, yy = np.meshgrid(
        ["Topic {}".format(i) for i in range(model.num_topics)],
        range(1, len(dd.index)+1))
    return xx, yy, dd


def load_model():
    loader = gensim.utils.SaveLoad()
    model = loader.load("DTMModel")
    return model


def load_corpus():
    corpus = pickle.load(open('corpus_geo.pkl', 'rb'))
    SW = ["pouvoir", "entrer", "partir", "faire", "grand"]
    corpus = [[x for x in sub if x not in SW] for sub in corpus]
    dictionary = corpora.Dictionary(corpus)
    corpus = [dictionary.doc2bow(text) for text in corpus]
    return corpus


def word_topic(xx, yy, dd, topic):
    traces = []
    for x, y, z, label in zip(xx, yy, dd.values, dd.index):
        c = ["#1876B2"]*len(xx)
        c[topic] = "red"
        traces.append(go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(size=z*17000, sizemode='area', color=c),
            name=label,
            hoverinfo='text',
            text=["({}, {})  {:04.3f}".format(a, label, b)
                  for a, b in zip(x, z)]
        ))

    layout = go.Layout(
        yaxis=dict(
            autorange=True,
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks='',
            #         showticklabels=False,
            tickvals=[k for k in range(1, 21)],
            ticktext=list(dd.index)
        ),
        showlegend=False,
        hovermode="closest",
        height=700,
        # width=500
    )

    fig = go.Figure(data=traces, layout=layout)
    return fig


def prep_data_wp(model, t2i, corpus, topic, vis=None):
    nb_words = 10
    time = 0

    words = [wo for we, wo in model.show_topic(topic, time, nb_words)]
    dd = {}
    # if vis==None:
    #     vis = {}
    for time in range(len(model.time_slices)):
        # if periodes[time] not in vis.keys():
        #     vis[periodes[time]] = model.dtm_vis(corpus, time)[1]
        for wo in words:
            if wo in dd.keys():
                dd[wo].append(vis[periodes[time]][1][topic, t2i[wo]])
            else:
                dd[wo] = []
                dd[wo].append(vis[periodes[time]][1][topic, t2i[wo]])
    dd = pd.DataFrame(dd).transpose()
    y_data = dd.values
    labels = list(dd.index)
    x_data = [periodes]*dd.shape[0]
    return x_data, y_data, labels


def word_period(x_data, y_data, labels):
    traces = []
    annotations = []
    for i in range(0, len(labels)):
        traces.append(go.Scatter(
            x=x_data[i],
            y=y_data[i],
            mode='lines',
            connectgaps=True,
            name=labels[i]
        ))
        traces.append(go.Scatter(
            x=[x_data[i][0], x_data[i][y_data.shape[1]-1]],
            y=[y_data[i][0], y_data[i][y_data.shape[1]-1]],
            mode='markers',
            #         marker=dict(color=colors[i]),
            hoverinfo="none"
            #         ,size=mode_size[i])
        ))

    for i, (y_trace, label) in enumerate(zip(y_data, labels)):
        # labeling the right_side of the plot
        annotations.append(dict(xref='paper', x=0.95, y=y_trace[y_data.shape[1]-1],
                                xanchor='left', yanchor='middle',
                                text=label,
                                font=dict(family='Arial',
                                          size=12,
                                          #                                             color=colors,
                                          ),
                                showarrow=False))
        annotations.append(dict(xref='paper', x=0.05, y=y_trace[0],
                                xanchor='right', yanchor='middle',
                                text=label,
                                font=dict(family='Arial',
                                          size=12,
                                          #                                             color=colors,
                                          ),
                                showarrow=False))

    layout = go.Layout(
        # width=1000,
        height=700,
        showlegend=False,
        hovermode="closest"
    )
    layout['annotations'] = annotations
    fig = go.Figure(data=traces, layout=layout)
    return fig


def topic_evolution(topic: int):
    """
    DOCSTRING
    """
    b = pd.DataFrame()
    b["date"] = data_all.date.sort_values().values
    b["weight"] = vis[periodes[0]][0][:, topic]
    val = b.groupby("date").mean().values
    # val = pd.DataFrame([vis[periodes[0]][0][:, topic], data_all.date.sort_values().values]).transpose().groupby(1).mean().values
    trace_scatter = [go.Scatter(x=data_all.date.sort_values().unique(),
                                y=val.ravel(),
                                mode='markers',
                                )]
    #layout = go.layout(hovermode='closest')
    return go.Figure(data=trace_scatter)


def terme_periode(model, t2i, terme):
    traces_ter = []
    termId = t2i[terme]
    for time_t in range(len(periodes)):
        prob_terme_topic = []
        for topic in range(model.num_topics):
            prob_terme_topic.append(vis[periodes[time_t]][1][topic][termId])

        traces_ter.append(go.Bar(x=["Topic {}".format(i) for i in range(
            model.num_topics)], y=prob_terme_topic, name=periodes[time_t]))

    #layout = go.layout(hovermode='closest')
    return go.Figure(data=traces_ter)


topic = 0

model = load_model()
corpus = load_corpus()
data_all = pd.read_csv(
    "data_geo.csv", sep="\t")

# periodes = ["1892-1909", "1910-1926", "1927-1942", "1942-1964", "1965-1976", "1977-1984", "1985-1993", "1994-2000", "2001-2007", "2008-2014"]
periodes = ['1892', '1901', '1909', '1919', '1926', '1933', '1942', '1956', '1964',
            '1972', '1976', '1980', '1984', '1988', '1993', '1997', '2000', '2004', '2007', '2011']

vis = {}
for time in range(len(model.time_slices)):
    vis[periodes[time]] = model.dtm_vis(corpus, time)

t2i = {model.id2word[i]: i for i in model.id2word.keys()}

xx, yy, dd = prep_data_to_plot(model, t2i, corpus, vis, topic)
x_data, y_data, labels = prep_data_wp(model, t2i, corpus, topic, vis)

traces = []
traces.append(go.Bar(x=["Topic {}".format(i) for i in range(
    model.num_topics)], y=np.sum(vis[periodes[0]][0], axis=0)))


app.layout = html.Div([
    html.H2("Dynamic Topic Modeling"),
    html.Pre(id='hover-data'),
    dcc.Slider(id='my-slider',
               min=0,
               max=9,
               step=1,
               value=0,),
    html.Div([
        html.Div([
            html.H3('Mots par topic'),
            dcc.Graph(id='plot1', figure=word_topic(xx, yy, dd, topic)),
        ], className="six columns"),

        html.Div([
            html.H3('Evolution des mots dans le temps'),
            dcc.Graph(id='plot2', figure=word_period(x_data, y_data, labels))
        ], className="six columns"),
    ], className="row"),
    html.Div([
        html.Div([
            html.H3('Poids des topics'),
            dcc.Graph(id='plot3', figure=go.Figure(data=traces))
        ], className="four columns"),
        html.Div([
            html.H3('Evolution du mot'),
            dcc.Input(id='word', value='milieu', type='text'),
            dcc.Checklist(id="check", options=[
                          {'label': 'Verrouiller', 'value': 'ver'}, ], values=["ver"]),
            dcc.Graph(id='plot4')
        ], className="four columns"),
        html.Div([
            html.H3('Poids des topics', id='pt'),
            dcc.Graph(id='plot5')
        ], className="four columns"),
    ], className="row"),
])

# @app.callback(
#     Output(component_id='plot1', component_property='figure'),
#     [Input(component_id='my-slider', component_property='value')]
# )
# def update_output_div(topic):
#     xx, yy, dd = prep_data_to_plot(model, corpus, vis, topic)
#     return word_topic(xx, yy, dd, topic)


@app.callback(
    Output('plot1', 'figure'),
    [Input('plot1', 'clickData')])
def display_hover_data_p1(clickData):
    if clickData is None:
        xx, yy, dd = prep_data_to_plot(model, t2i, corpus, vis, 0)
        return word_topic(xx, yy, dd, 0)
    else:
        xx, yy, dd = prep_data_to_plot(
            model, t2i, corpus, vis, clickData["points"][0]["pointNumber"])
        return word_topic(xx, yy, dd, clickData["points"][0]["pointNumber"])


@app.callback(
    Output('plot5', 'figure'),
    [Input('plot1', 'clickData')])
def update_data_p5(clickData):
    if clickData is None:
        return topic_evolution(0)
    else:
        return topic_evolution(clickData["points"][0]["pointNumber"])


@app.callback(
    Output('pt', 'children'),
    [Input('plot1', 'clickData')])
def display_hover_data_p5(clickData):
    return "Poids du {} dans le temps".format(clickData['points'][0]['x'])


@app.callback(
    Output('word', 'value'),
    [Input('plot1', 'clickData'),
     Input("check", "values")]
)
def update_val(clickData, check):
    if (len(check) == 0):
        word = re.findall(
            r"(\w+)", clickData['points'][0]['text'], re.UNICODE)[2]
        return "{}".format(word)


@app.callback(
    Output(component_id='plot4', component_property='figure'),
    [Input(component_id='word', component_property='value'),
     Input("check", "values"),
     Input('plot1', 'clickData')]
)
def update_output_div(input_value, check, clickData):
    if (len(check) == 0) and (clickData is not None):
        word = re.findall(
            r"(\w+)", clickData['points'][0]['text'], re.UNICODE)[2]
        #word = re.findall("[A-z]+", clickData['points'][0]['text'])[1]
        return terme_periode(model, t2i, word)
    else:
        return terme_periode(model, t2i, input_value)


@app.callback(
    Output('plot2', 'figure'),
    [Input('plot1', 'clickData'),
     Input('plot1', 'hoverData')])
def display_hover_data_p2(clickData, hoverData):
    if clickData is None:
        x_data, y_data, labels = prep_data_wp(model, t2i, corpus, 0, vis)
    else:
        x_data, y_data, labels = prep_data_wp(
            model, t2i, corpus, clickData["points"][0]["pointNumber"], vis)
    return word_period(x_data, y_data, labels)


if __name__ == '__main__':
    app.run_server(debug=True)
