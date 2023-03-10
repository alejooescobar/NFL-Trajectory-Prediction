from dash import Dash, html, dcc
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
import tensorflow as tf
import numpy as np

app = Dash(__name__)

model = tf.keras.models.load_model('../Notebooks/LSTMModel2Layer.h5')
o_player_sequences, d_player_sequences = np.load('../Notebooks/trajectories.npy',allow_pickle=True)
o_padding_value = [-1.0 for i in range(len([o_player_sequences[0][0]]))]
padded_o_seq = tf.keras.preprocessing.sequence.pad_sequences(o_player_sequences,padding='post', value=o_padding_value, dtype='float32',maxlen = 90)
masking_o_layer = tf.keras.layers.Masking(mask_value=-1)
masked_o_seq = masking_o_layer(padded_o_seq)

padding_value = [-1.0 for i in range(len([d_player_sequences[0][0]]))]
padded_d_seq = tf.keras.preprocessing.sequence.pad_sequences(d_player_sequences,padding='post', value=padding_value, dtype='float32',maxlen = 90)
masking_layer = tf.keras.layers.Masking(mask_value=-1)
masked_d_seq = masking_layer(padded_d_seq)

marker_colors = ['rgb(200, 200, 255)']
los_x_arr = []
los_y_arr = []
o_x = []
o_y = []
d_x = []
d_y = []

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
                disabled = False
            )
    ])
])

@app.callback(
    Output('interval-component', 'disabled'),
    [Input('play-button', 'n_clicks'),Input('interval-component', 'n_intervals')],
    [State('interval-component', 'disabled')]
)
def start_stop_interval(n_clicks, n_intervals,disabled):
    print(n_intervals)
    n_intervals = 0
    return not disabled

@app.callback(
    Output('trajectory-graph', 'figure'),
    [Input('start-x', 'value'),
     Input('start-y', 'value'),
     Input('seq-index', 'value'),
     Input('los-slider', 'value'),
     Input('y-slider','value'),
     Input('play-button', 'n_clicks'),
     Input('interval-component', 'n_intervals')],    
     [State('play-button', 'n_clicks')]
)
def update_figure(start_x, start_y,index,los_x,y_pos,n_clicks, n_intervals,prev_n_clicks):
    if n_clicks > 0:
        stop_interval = False
        print(n_intervals)
        #n_clicks = 0
    
    o_x, o_y,d_x,d_y = generate_trajectory(los_x-1, y_pos,start_x,start_y,index,los_x)
    los_x_arr,los_y_arr = los(los_x)
    marker_colors = ['rgb(200, 200, 255)']
    for i in range(1, len(o_x)):
        red_value = int(marker_colors[i-1].split(',')[0].split('(')[1]) - 2
        green_value = int(marker_colors[i-1].split(',')[1]) - 2
        #blue_value = int(marker_colors[i-1].split(',')[2].split(')')[0]) + 10
        marker_colors.append(f'rgb({red_value}, {green_value}, 255)')
    o_trace = go.Scattergl(x=o_x, y=o_y, mode='markers', marker=dict(size=10, color=marker_colors), name='Offensive Player', showlegend=True, legendgroup='group1')

    marker_colors = ['rgb(200, 255, 200)']
    for i in range(1, len(d_x)):
        red_value = int(marker_colors[i-1].split(',')[0].split('(')[1]) - 2
        #green_value = int(marker_colors[i-1].split(',')[1]) - 2
        blue_value = int(marker_colors[i-1].split(',')[2].split(')')[0]) -2
        marker_colors.append(f'rgb({red_value}, 255, {blue_value})')
    d_trace = go.Scattergl(x=d_x, y=d_y, mode='markers', marker=dict(size=10, color=marker_colors), name='Defensive Player', showlegend=True, legendgroup='group1')

    los_trace = go.Scattergl(x=los_x_arr, y=los_y_arr, mode='lines', name='Line of Scrimmage', showlegend=True, legendgroup='group1')
    fig = go.Figure(data=[o_trace,d_trace,los_trace], layout=go.Layout(title='Trajectory Plot', xaxis=dict(title='X',range=[0,120]), yaxis=dict(title='Y',range=[0,53.3]),legend=dict(x=1, y=1, traceorder='normal', font=dict(family='sans-serif', size=12, color='#000'),orientation = 'h',xanchor = "right",yanchor="bottom")))
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)