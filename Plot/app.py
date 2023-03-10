from dash import Dash, html, dcc
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
import tensorflow as tf
import numpy as np
import json

app = Dash(__name__)

model = tf.keras.models.load_model('../Notebooks/LSTMModel2Layer.h5')

o_player_sequences, d_player_sequences = np.load('../Notebooks/trajectories.npy',allow_pickle=True)
sequences_ordered = np.load('../Notebooks/sequences_ordered.npy',allow_pickle=True)


with open('../Notebooks/player_pairs.json', 'r') as f:
    player_pairs_str = json.load(f)
    player_pairs = {tuple(key_str.split(',')): value for key_str, value in player_pairs_str.items()}

o_padding_value = [-1.0 for i in range(len([o_player_sequences[0][0]]))]
padded_o_seq = tf.keras.preprocessing.sequence.pad_sequences(o_player_sequences,padding='post', value=o_padding_value, dtype='float32',maxlen = 90)
masking_o_layer = tf.keras.layers.Masking(mask_value=-1)
masked_o_seq = masking_o_layer(padded_o_seq)

padding_value = [-1.0 for i in range(len([d_player_sequences[0][0]]))]
padded_d_seq = tf.keras.preprocessing.sequence.pad_sequences(d_player_sequences,padding='post', value=padding_value, dtype='float32',maxlen = 90)
masking_layer = tf.keras.layers.Masking(mask_value=-1)
masked_d_seq = masking_layer(padded_d_seq)
prev_clicks = 0
o_x, o_y,d_x,d_y = [],[],[],[]
los_x_arr,los_y_arr = [],[]
o_marker_colors = ['rgb(200, 200, 255)']
d_marker_colors = ['rgb(200, 255, 200)']


# define a function that generates x and y arrays based on user input
def generate_trajectory(start_x,start_y, d_start_x,d_start_y,index,los_x):
    #print(d_start_y)
    #print(start_y)
    index = int(index)
    if index<0:
        index = -1*index
    o_seq = o_player_sequences[index]
    expected_sequence = masked_d_seq[index]
    o_sequence_len = len(o_seq)

    #left_sideline = 53.3
    #right_sideline = 0.0

    #o_left_sideline_distance = abs(left_sideline - (start_y))
    #o_right_sideline_distance = abs(right_sideline - (start_y))
    #d_left_sideline_distance = abs(left_sideline - (start_y+d_start_y))
    #d_right_sideline_distance = abs(right_sideline - (start_y+d_start_y))
    #print(d_right_sideline_distance)
    #print(d_left_sideline_distance)

    for i in range(o_sequence_len):
        o_seq[i][6] = d_start_x
        o_seq[i][7] = d_start_y
        o_seq[i][8] = start_x
        o_seq[i][9] = start_y
        o_seq[i][10] = los_x
    padded_o_seq = tf.keras.preprocessing.sequence.pad_sequences([o_seq],padding='post', value=o_padding_value, dtype='float32',maxlen = 90)
    m_o_seq = masking_o_layer(padded_o_seq)
    predict_dataset = tf.data.Dataset.from_tensor_slices((m_o_seq,[expected_sequence]))
    predict_dataset_batched = predict_dataset.batch(1)
    predicted_sequence = model.predict(predict_dataset_batched)
    predicted_sequence = predicted_sequence[0]

    predicted_sequence_x = [x[0]+start_x+d_start_x for x in predicted_sequence]
    predicted_sequence_y = [x[1]+start_y+d_start_y for x in predicted_sequence]
    predicted_sequence_x = predicted_sequence_x[0:o_sequence_len]
    predicted_sequence_y = predicted_sequence_y[0:o_sequence_len]


    z = np.polyfit(predicted_sequence_x, predicted_sequence_y, 3)
    f = np.poly1d(z)

    # calculate new x's and y's
    x_new = np.linspace(predicted_sequence_x[0], predicted_sequence_x[-1], 50)
    y_new = f(x_new)

    x = [x[0]+start_x for x in o_seq]
    y = [x[1]+start_y for x in o_seq]
    return x, y,x_new,y_new


def los(x):
    x = [x,x]
    y = [0,53.3]
    return x,y
data = []

layout = go.Layout(title='Trajectory Plot', xaxis=dict(title='X'), yaxis=dict(title='Y'), legend=dict(x=1, y=1, traceorder='normal', font=dict(family='sans-serif', size=12, color='#000')))

fig = go.Figure(data=data, layout=layout)

app.layout = html.Div([
    html.Div([
        html.Label('Defensive Start X'),
        dcc.Input(id='start-x', type='number', value=2)
    ]),
    html.Div([
        html.Label('Defensive Start Y'),
        dcc.Input(id='start-y', type='number', value=0)
    ]),
    html.Div([
        html.Label('Sequence Index'),
        dcc.Input(id='seq-index', type='number', value=0)
    ]),
    html.Div([
            html.Div([
                html.Div([
                    dcc.Graph(
                        id='trajectory-graph', figure=fig, 
                        config={'displayModeBar': False, 'doubleClick': 'reset'},
                        clickData={'points': [{'customdata': 'Add Point'}]}
                    )]),
                    html.Div([
                    dcc.Slider(
                        id='los-slider',
                        min=0,
                        max=120,
                        step=1,
                        value=60,
                        marks={
                            0: {'label': '0', 'style': {'color': '#77b0b1'}},
                            20: {'label': '20'},
                            40: {'label': '40'},
                            60: {'label': '60'},
                            80: {'label': '80'},
                            100: {'label': '100'},
                            120: {'label': '120', 'style': {'color': '#f50'}}

                        }
                    )],style={'width':'94%','margin':'auto'})    
                    ],style={'display': 'inline-block', 'width': '95%'}),
                    html.Div([
                    dcc.Slider(
                        id='y-slider',
                        min=0,
                        max=53,
                        step=1,
                        value=26,
                        vertical = True,
                        marks={
                            0: {'label': '0', 'style': {'color': '#77b0b1'}},
                            10: {'label': '10'},
                            20: {'label': '20'},
                            30: {'label': '30'},
                            40: {'label': '40'},
                            50: {'label': '50', 'style': {'color': '#f50'}},
                        }
                    )]
            ,style={'display': 'inline-block', 'width': '5%'}),
            html.Button(id='play-button', children='Play', n_clicks=0, style={'margin-left': '30px', 'font-size': '20px'}),
            dcc.Interval(
                id='interval-component',
                interval=500,  # in milliseconds
                n_intervals=0,
                disabled = True
            )
    ]),
    html.Div(children=[
        html.Div(children=[
            dcc.Graph(
                id='plot1',
                figure={
                    'data': [{'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'Plot 1'}],
                    'layout': {'title': 'Plot 1'}
                }
            ),
            html.Button('Button 1', id='button1', n_clicks=0)
        ], style={'display': 'inline-block', 'width': '24%', 'text-align': 'center'}),
        html.Div(children=[
            dcc.Graph(
                id='plot2',
                figure={
                    'data': [{'x': [1, 2, 3], 'y': [1, 4, 3], 'type': 'bar', 'name': 'Plot 2'}],
                    'layout': {'title': 'Plot 2'}
                }
            ),
            html.Button('Button 2', id='button2', n_clicks=0)
        ], style={'display': 'inline-block', 'width': '24%', 'text-align': 'center'}),
        html.Div(children=[
            dcc.Graph(
                id='plot3',
                figure={
                    'data': [{'x': [1, 2, 3], 'y': [3, 2, 4], 'type': 'bar', 'name': 'Plot 3'}],
                    'layout': {'title': 'Plot 3'}
                }
            ),
            html.Button('Button 3', id='button3', n_clicks=0)
        ], style={'display': 'inline-block', 'width': '24%', 'text-align': 'center'}),
        html.Div(children=[
            dcc.Graph(
                id='plot4',
                figure={
                    'data': [{'x': [1, 2, 3], 'y': [2, 3, 1], 'type': 'bar', 'name': 'Plot 4'}],
                    'layout': {'title': 'Plot 4'}
                }
            ),
            html.Button('Button 4', id='button4', n_clicks=0)
        ], style={'display': 'inline-block', 'width': '24%', 'text-align': 'center'})
    ])
])

@app.callback(
    Output('interval-component', 'disabled'),
    Output('interval-component', 'n_intervals'),
    [Input('play-button', 'n_clicks'),Input('interval-component', 'n_intervals')],
    [State('interval-component','disabled')]
)
def start_stop_interval(n_clicks,n_intervals,disabled):
    global prev_clicks
    if n_clicks > prev_clicks:
        prev_clicks = n_clicks
        return False,0
    if not(disabled) and n_intervals > 18:
        return True, 0
    elif n_clicks > 0 and not(disabled):
        prev_clicks = n_clicks
        return False, n_intervals

    return disabled,n_intervals

@app.callback(
    Output('trajectory-graph', 'figure'),
    [Input('start-x', 'value'),
     Input('start-y', 'value'),
     Input('seq-index', 'value'),
     Input('los-slider', 'value'),
     Input('y-slider','value'),
     #Input('play-button', 'n_clicks'),
     Input('interval-component', 'n_intervals')],    
    [State('interval-component','disabled')]

)
def update_figure(start_x, start_y,index,los_x,y_pos, n_intervals,disabled):
    global o_x,o_y,d_x,d_y,los_x_arr,los_y_arr,o_marker_colors,d_marker_colors
    if not(disabled):
        seq_len = min(len(o_x),len(o_y),len(d_x),len(d_y))
        current_index = n_intervals*5
        if current_index < seq_len:
            new_ox = o_x[:current_index]
            new_oy = o_y[:current_index]
            new_dx = d_x[:current_index]
            new_dy = d_y[:current_index]
            o_trace = go.Scattergl(x=new_ox, y=new_oy, mode='markers', marker=dict(size=10, color=o_marker_colors), name='Offensive Player', showlegend=True, legendgroup='group1')
            d_trace = go.Scattergl(x=new_dx, y=new_dy, mode='markers', marker=dict(size=10, color=d_marker_colors), name='Defensive Player', showlegend=True, legendgroup='group1')
            los_trace = go.Scattergl(x=los_x_arr, y=los_y_arr, mode='lines', name='Line of Scrimmage', showlegend=True, legendgroup='group1')
            fig = go.Figure(data=[o_trace,d_trace,los_trace], layout=go.Layout(title='Trajectory Plot', xaxis=dict(title='X',range=[0,120]), yaxis=dict(title='Y',range=[0,53.3]),legend=dict(x=1, y=1, traceorder='normal', font=dict(family='sans-serif', size=12, color='#000'),orientation = 'h',xanchor = "right",yanchor="bottom")))
            return fig
        else:
            o_trace = go.Scattergl(x=o_x, y=o_y, mode='markers', marker=dict(size=10, color=o_marker_colors), name='Offensive Player', showlegend=True, legendgroup='group1')
            d_trace = go.Scattergl(x=d_x, y=d_y, mode='markers', marker=dict(size=10, color=d_marker_colors), name='Defensive Player', showlegend=True, legendgroup='group1')
            los_trace = go.Scattergl(x=los_x_arr, y=los_y_arr, mode='lines', name='Line of Scrimmage', showlegend=True, legendgroup='group1')
            fig = go.Figure(data=[o_trace,d_trace,los_trace], layout=go.Layout(title='Trajectory Plot', xaxis=dict(title='X',range=[0,120]), yaxis=dict(title='Y',range=[0,53.3]),legend=dict(x=1, y=1, traceorder='normal', font=dict(family='sans-serif', size=12, color='#000'),orientation = 'h',xanchor = "right",yanchor="bottom")))
            return fig
    o_x, o_y,d_x,d_y = generate_trajectory(los_x-1, y_pos,start_x,start_y,index,los_x)
    los_x_arr,los_y_arr = los(los_x)
    o_marker_colors = ['rgb(200, 200, 255)']
    d_marker_colors = ['rgb(200, 255, 200)']
    for i in range(1, len(o_x)):
        red_value = int(o_marker_colors[i-1].split(',')[0].split('(')[1]) - 2
        green_value = int(o_marker_colors[i-1].split(',')[1]) - 2
        #blue_value = int(marker_colors[i-1].split(',')[2].split(')')[0]) + 10
        o_marker_colors.append(f'rgb({red_value}, {green_value}, 255)')
    for i in range(1, len(d_x)):
        red_value = int(d_marker_colors[i-1].split(',')[0].split('(')[1]) - 2
        #green_value = int(marker_colors[i-1].split(',')[1]) - 2
        blue_value = int(d_marker_colors[i-1].split(',')[2].split(')')[0]) -2
        d_marker_colors.append(f'rgb({red_value}, 255, {blue_value})')
    o_trace = go.Scattergl(x=o_x, y=o_y, mode='markers', marker=dict(size=10, color=o_marker_colors), name='Offensive Player', showlegend=True, legendgroup='group1')
    d_trace = go.Scattergl(x=d_x, y=d_y, mode='markers', marker=dict(size=10, color=d_marker_colors), name='Defensive Player', showlegend=True, legendgroup='group1')
    los_trace = go.Scattergl(x=los_x_arr, y=los_y_arr, mode='lines', name='Line of Scrimmage', showlegend=True, legendgroup='group1')
    fig = go.Figure(data=[o_trace,d_trace,los_trace], layout=go.Layout(title='Trajectory Plot', xaxis=dict(title='X',range=[0,120]), yaxis=dict(title='Y',range=[0,53.3]),legend=dict(x=1, y=1, traceorder='normal', font=dict(family='sans-serif', size=12, color='#000'),orientation = 'h',xanchor = "right",yanchor="bottom")))
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)