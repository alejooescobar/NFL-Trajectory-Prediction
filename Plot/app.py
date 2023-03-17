from dash import Dash, html, dcc
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
import tensorflow as tf
import numpy as np
import json
import pandas as pd
import ast
import dash
import dash_bootstrap_components as dbc


app = Dash(__name__,suppress_callback_exceptions=True,external_stylesheets=[dbc.themes.BOOTSTRAP])

model = tf.keras.models.load_model('../Notebooks/LSTMModel2Layer.h5')

o_player_sequences, d_player_sequences = np.load('../Notebooks/trajectories.npy',allow_pickle=True)
sequences_ordered = np.load('../Notebooks/sequences_ordered.npy',allow_pickle=True)
# getting the ordered sequences from the clustering
o_player_sequences_id = np.load('../Notebooks/o_player_sequences_id.npy',allow_pickle=True)

with open('../Notebooks/player_pairs.json', 'r') as f:
    player_pairs_str = json.load(f)
    player_pairs = {ast.literal_eval(key_str): value for key_str, value in player_pairs_str.items()}

with open('../Notebooks/player_pair_ids.json', 'r') as f:
    player_pair_ids_str = json.load(f)
    player_pair_ids = {ast.literal_eval(key_str): value for key_str, value in player_pair_ids_str.items()}

player_pair_ids = {k:v for k,v in player_pair_ids.items() if v}

trajectory_dict_keys = pd.DataFrame(list(player_pair_ids.keys()),columns=["gameId","playId"])
bdb_games = pd.read_csv("../NFLData/games.csv")
bdb_plays = pd.read_csv("../NFLData/plays.csv")
bdb_players = pd.read_csv("../NFLData/players.csv")
valid_games = trajectory_dict_keys[["gameId"]].drop_duplicates()
valid_plays = trajectory_dict_keys[["gameId","playId"]].drop_duplicates()

valid_games = pd.merge(bdb_games,valid_games,on="gameId",how="inner")
valid_plays = pd.merge(bdb_plays,valid_plays,on=["gameId","playId"],how="inner")


with open('../Notebooks/all_trajectory_dict.json', 'r') as f:
    all_trajectory_dict_json = json.load(f)
    all_trajectory_dict = {ast.literal_eval(key_str): value for key_str, value in all_trajectory_dict_json.items()}

o_padding_value = [-1.0 for _ in range(len([o_player_sequences[0][0]]))]
padded_o_seq = tf.keras.preprocessing.sequence.pad_sequences(o_player_sequences,padding='post', value=o_padding_value, dtype='float32',maxlen = 90)
masking_o_layer = tf.keras.layers.Masking(mask_value=-1)
masked_o_seq = masking_o_layer(padded_o_seq)

padding_value = [-1.0 for _ in range(len([d_player_sequences[0][0]]))]
padded_d_seq = tf.keras.preprocessing.sequence.pad_sequences(d_player_sequences,padding='post', value=padding_value, dtype='float32',maxlen = 90)
masking_layer = tf.keras.layers.Masking(mask_value=-1)
masked_d_seq = masking_layer(padded_d_seq)
prev_clicks = 0
o_x, o_y,d_x,d_y = [],[],[],[]
los_x_arr,los_y_arr = [],[]
o_marker_colors = ['rgb(200, 200, 255)']
d_marker_colors = ['rgb(200, 255, 200)']
cluster_options = [{'label': f'Cluster Center {i+1}', 'value': i} for i in range(8)]
cluster_center_options = [{'label': f'Sequence {i+1}', 'value': i} for i in range(8)]
game_options = []
for row in valid_games.itertuples():
    game_options.append({'label':f'{row.homeTeamAbbr} vs. {row.visitorTeamAbbr} playing home to {row.homeTeamAbbr} on {row.gameDate}','value':row.gameId})


def flipX(x,los_x):
    flipped_x = x - los_x
    flipped_x = -flipped_x
    flipped_x = flipped_x+los_x
    return flipped_x

def flipY(y,los_y):
    flipped_y = y - los_y
    flipped_y = -flipped_y
    flipped_y = flipped_y+los_y
    return flipped_y

# define a function that generates x and y arrays based on user input
def generate_trajectory(start_x,start_y, d_start_x,d_start_y,index,los_x,flip_x,flip_y):
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

    if flip_x:
        #print(f"Flipping each coordinate around {los_x}")
        predicted_sequence_x = [flipX(x[0]+(2*los_x-(start_x+d_start_x)),los_x) for x in predicted_sequence]
        x = [flipX(x[0]+(2*los_x-start_x),los_x) for x in o_seq]
    else:
        predicted_sequence_x = [x[0]+start_x+d_start_x for x in predicted_sequence]
        x = [x[0]+start_x for x in o_seq]
    if flip_y:
        predicted_sequence_y = [flipY(x[1]+(53.3-(start_y+d_start_y)),26.65) for x in predicted_sequence]
        y = [flipY(x[1]+(53.3-start_y),26.65) for x in o_seq]
    else:
        predicted_sequence_y = [x[1]+start_y+d_start_y for x in predicted_sequence]
        y = [x[1]+start_y for x in o_seq]
    predicted_sequence_x = predicted_sequence_x[0:o_sequence_len]
    predicted_sequence_y = predicted_sequence_y[0:o_sequence_len]


    z = np.polyfit(predicted_sequence_x, predicted_sequence_y, 3)
    f = np.poly1d(z)

    # calculate new x's and y's
    x_new = np.linspace(predicted_sequence_x[0], predicted_sequence_x[-1], 50)
    y_new = f(x_new)
    return x, y,x_new,y_new

def generate_play_trajectories(index,los_x):
    play_key = o_player_sequences_id[index] # game play o_player d_player
    #print(f"play key {play_key}",end="\n\n")
    o_player_id = int(play_key[2])
    d_player_id = int(play_key[3])
    #print(f"o_player_id: {o_player_id} d_player_id: {d_player_id}")
    teams = valid_plays[(valid_plays["gameId"] == play_key[0])&(valid_plays["playId"] == play_key[1])]
    #print(teams,end="\n\n")
    off_team = teams.iloc[0]['possessionTeam']
    def_team = teams.iloc[0]['defensiveTeam']
    actual_los = teams.iloc[0]['absoluteYardlineNumber']
    #print(off_team)
    #print(actual_los)
    off_play = all_trajectory_dict[(play_key[0],play_key[1],off_team)]
    def_play = all_trajectory_dict[(play_key[0],play_key[1],def_team)]
    off_play = {int(float(k)):v for k,v in off_play.items()}
    def_play = {int(float(k)):v for k,v in def_play.items()}
    #print(off_play.keys(),end="\n\n")
    #print(def_play.keys(),end="\n\n")
    o_player = off_play.pop(o_player_id,[])
    d_player = def_play.pop(d_player_id,[])
    if len(d_player) == 0 or len(o_player) == 0:
        #print(f"o_player: {o_player} d_player: {d_player}")
        return [],[],[],[],[],[],[],[],False,False
    off_seqs = list(off_play.values())
    def_seqs = list(def_play.values())
    off_seq_x = [x[0]-actual_los+los_x for i in range(len(off_seqs)) for x in off_seqs[i]]
    off_seq_y = [x[1] for i in range(len(off_seqs)) for x in off_seqs[i]]
    def_seq_x = [x[0]-actual_los+los_x for i in range(len(def_seqs)) for x in def_seqs[i]]
    def_seq_y = [x[1] for i in range(len(def_seqs)) for x in def_seqs[i]]
    
    flip_x = actual_los < o_player[0][0] # on the right side of los
    flip_y = 26.65 < o_player[0][1] # above halfway line

    original_o_player_x = [x[0]-actual_los+los_x for x in o_player]
    original_o_player_y = [x[1] for x in o_player]

    original_d_player_x = [x[0]-actual_los+los_x for x in d_player]
    original_d_player_y = [x[1] for x in d_player]

    
    return off_seq_x,off_seq_y,def_seq_x,def_seq_y,original_o_player_x,original_o_player_y,original_d_player_x,original_d_player_y,flip_x,flip_y



def los(x):
    x = [x,x]
    y = [0,53.3]
    return x,y
data = []

layout = go.Layout(title='Trajectory Plot', xaxis=dict(title='X'), yaxis=dict(title='Y'), legend=dict(x=1, y=1, traceorder='normal', font=dict(family='sans-serif', size=12, color='#000')))

fig = go.Figure(data=data, layout=layout)

app.layout = html.Div([
    html.Div(id="hidden-div", style={"display": "none"}),
    dbc.Container([
        html.H1(children='',id='plot-header'),
        html.H1(
            children="My Dash Application",
            className="center-align"
        ),
        html.Div(
            className="card-panel",
            children=[
                html.H3(
                    children="Description",
                    className="blue-grey-text text-darken-4"
                ),
                html.P(
                    children="This is a sample Dash application. It demonstrates how to create a simple layout using Dash HTML components.",
                    className="grey-text text-darken-3 flow-text"
                )
            ]
        )
    ],
    id = "info-component",
    fluid = True,
    className = "py-3"),

    html.Div([
    dcc.Tabs(id='tabs', value='seq-selection', children=[
        dcc.Tab(label='Sequence Selection', value='seq-selection'),
        dcc.Tab(label='Cluster Selection', value='cluster-selection'),
        dcc.Tab(label='Play Selection', value='play-selection'),
        ]),
    ]),
    html.Div([
        dcc.Tabs(id='plot-tabs', value='play-isolated', children=[
            dcc.Tab(label='Isolate Single Pair', value='play-isolated'),
            dcc.Tab(label='Full Play', value='play-full'),
            #dcc.Tab(label='Play Selection', value='play-selection'),
        ]) 
    ]),
    html.Div([
            html.Div([
                html.Div([
                    dcc.Graph(
                        id='trajectory-graph', figure=fig, 
                        config={'displayModeBar': False, 'doubleClick': 'reset'},
                        clickData={'points': [{'customdata': 'Add Point'}]},
                        style={'height': '800px'}
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
                        },
                        verticalHeight = 700
                    )]
            ,style={'display': 'inline-block', 'width': '5%','marginBottom':'3%'}),
            dcc.Interval(
                id='interval-component',
                interval=500,  # in milliseconds
                n_intervals=0,
                disabled = True
            )
    ]),
    html.Div([
        dbc.Button(id='play-button', children='View Play in Real Time', n_clicks=0)
    ],className = "d-grid gap-2 col-6 mx-auto my-4"),
    dbc.Container([
        dbc.Form(
            dbc.Row(
                [
                    dbc.Label('Defensive Start X',width = 'auto'),
                    dbc.Col(
                        dcc.Slider(id="start-x", min=0, max=10, step=0.5, value=0)
                    ),
                    dbc.Label('Defensive Start Y', width="auto"),
                    dbc.Col(
                        dcc.Slider(id="start-y", min=-5, max=5, step=0.5, value=0)
                    )
                ]
            )
        ,className="my-4"),
        dbc.Form(
            dbc.Row(
                [
                    dbc.Label('Sequence Index',width = 'auto'),
                    dbc.Col(
                        dbc.Input(id='seq-index', type='number', value=0,placeholder="Please select an integer to display a sequence on the Trajectory Plot.")
                    ),
                    dbc.Label('Original Sequence Toggle', width="auto"),
                    dbc.Col(
                    dbc.Checklist(
                        options=[
                            {"label": "Show Original", "value": True},
                        ],
                        value=[True],
                        id="original-switch",
                        switch=True,
                        )
                    )
                ]
            )
        ,className= "my-4")
    ],fluid=True),
    html.Div(id = "play-mode",children=[
                dcc.Dropdown(
                id='game-dropdown',
                options=game_options,
                value=None,
                placeholder='Select a Game in the season to view plays...',
                searchable=True),
                dcc.Dropdown(
                id='play-dropdown',
                options=[],
                value=None,
                placeholder='Select a Play in the selected game to plot...',
                searchable=True),
            ]),
    html.Div(
        id = "cluster-mode",
        children=[
            html.Div(
                    children=[
                    html.H2('Please select a cluster center below to view sequences in the selected cluster.',id='cluster-header'),
                    dcc.Dropdown(
                        id='cluster-dropdown',
                        options=cluster_options,
                        value=None,
                        placeholder='Select a cluster center...',
                        searchable=False),
                    html.H2('Please select a sequence below to display on the trajectory plot.',id='seq-header'),
                    dcc.Dropdown(
                        id='cluster-center-dropdown',
                        options=cluster_center_options,
                        value=None,
                        placeholder='Select a sequence to display...',
                        searchable=False),
                    ]
                ),
            html.Div(id='cluster-div',children=[
                html.Div(children=[
                        dcc.Graph(
                            id='cluster-graph-1',  
                            config={'displayModeBar': False, 'doubleClick': 'reset'},
                            clickData={'points': [{'customdata': 'Add Point'}]}
                    ),
                ], style={'display': 'inline-block', 'width': '24%', 'textAlign': 'center','position': 'relative'}),
                html.Div(children=[
                        dcc.Graph(
                            id='cluster-graph-2', 
                            config={'displayModeBar': False, 'doubleClick': 'reset'},
                            clickData={'points': [{'customdata': 'Add Point'}]}
                    ),
                ], style={'display': 'inline-block', 'width': '24%', 'textAlign': 'center'}),
                html.Div(children=[
                    dcc.Graph(
                        id='cluster-graph-3', 
                        config={'displayModeBar': False, 'doubleClick': 'reset'},
                        clickData={'points': [{'customdata': 'Add Point'}]}
                    ),
                ], style={'display': 'inline-block', 'width': '24%', 'textAlign': 'center'}),
                html.Div(children=[
                    dcc.Graph(
                        id='cluster-graph-4',  
                        config={'displayModeBar': False, 'doubleClick': 'reset'},
                        clickData={'points': [{'customdata': 'Add Point'}]}
                    ),
                ], style={'display': 'inline-block', 'width': '24%', 'textAlign': 'center'}),
                        html.Div(children=[
                    dcc.Graph(
                        id='cluster-graph-5', 
                        config={'displayModeBar': False, 'doubleClick': 'reset'},
                        clickData={'points': [{'customdata': 'Add Point'}]}
                    ),
                ], style={'display': 'inline-block', 'width': '24%', 'textAlign': 'center'}),
                html.Div(children=[
                    dcc.Graph(
                        id='cluster-graph-6',  
                        config={'displayModeBar': False, 'doubleClick': 'reset'},
                        clickData={'points': [{'customdata': 'Add Point'}]}
                    ),
                ], style={'display': 'inline-block', 'width': '24%', 'textAlign': 'center'}),
                html.Div(children=[
                    dcc.Graph(
                        id='cluster-graph-7', 
                        config={'displayModeBar': False, 'doubleClick': 'reset'},
                        clickData={'points': [{'customdata': 'Add Point'}]}
                    ),
                ], style={'display': 'inline-block', 'width': '24%', 'textAlign': 'center'}),
                html.Div(children=[
                    dcc.Graph(
                        id='cluster-graph-8',  
                        config={'displayModeBar': False, 'doubleClick': 'reset'},
                        clickData={'points': [{'customdata': 'Add Point'}]}
                    ),
                ], style={'display': 'inline-block', 'width': '24%', 'textAlign': 'center'})
            ],style={})
        ]
    ),
    html.Div(id="extra-space",children=[],style={"height":"400px"})
])

@app.callback(
    Output('interval-component', 'disabled'),
    Output('interval-component', 'n_intervals'),
    [Input('play-button', 'n_clicks'),Input('interval-component', 'n_intervals')],
    [State('interval-component','disabled')]
)
def start_stop_interval(n_clicks,n_intervals,disabled):
    global prev_clicks,o_x,o_y,d_x,d_y
    seq_len = min(len(o_x),len(o_y),len(d_x),len(d_y))
    if n_clicks > prev_clicks:
        prev_clicks = n_clicks
        return False,0
    if not(disabled) and n_intervals*5 > seq_len:
        return True, 0
    elif n_clicks > 0 and not(disabled):
        prev_clicks = n_clicks
        return False, n_intervals

    return disabled,n_intervals

@app.callback(
    Output('trajectory-graph', 'figure'),
    Output('plot-header','children'),
    [Input('start-x', 'value'),
     Input('start-y', 'value'),
     Input('seq-index', 'value'),
     Input('los-slider', 'value'),
     Input('y-slider','value'),
     Input('interval-component', 'n_intervals'),
     Input('plot-tabs','value'),
     Input('original-switch','value')],    
    [State('interval-component','disabled')]
)
def update_figure(start_x, start_y,index,los_x,y_pos, n_intervals,plot_type,show_original,disabled):
    if index is None:
        return dash.no_update,dash.no_update
    global o_x,o_y,d_x,d_y,off_seq_x,off_seq_y,def_seq_x,def_seq_y,oo_x,oo_y,od_x,od_y,los_x_arr,los_y_arr,o_marker_colors,d_marker_colors
    if not(disabled):
        seq_len = min(len(o_x),len(d_x),len(off_seq_x)//10,len(def_seq_x)//10,len(oo_x),len(oo_y))
        current_index = n_intervals*5
        if current_index < seq_len:
            new_ox = o_x[:current_index]
            new_oy = o_y[:current_index]
            new_dx = d_x[:current_index]
            new_dy = d_y[:current_index]

            # original team excluding the selected player sequences
            #otx_2d = 
            inner_length = len(off_seq_x)//10
            new_otx = []
            new_oty = []
            for i in range(10):
                counter = i * inner_length
                for j in range(inner_length):
                    if j < current_index and counter < len(off_seq_x):
                        new_otx.append(off_seq_x[counter])
                        new_oty.append(off_seq_y[counter])
                    else:
                        break
                    counter += 1
            inner_length = len(def_seq_x)//10
            new_dtx = []
            new_dty = []
            for i in range(10):
                counter = i * inner_length
                for j in range(inner_length):
                    if j < current_index and counter < len(def_seq_x):
                        new_dtx.append(def_seq_x[counter])
                        new_dty.append(def_seq_y[counter])
                    else:
                        break
                    counter += 1
            # original player sequences 
            new_oox = oo_x[:current_index]
            new_ooy = oo_y[:current_index]
            new_odx = od_x[:current_index]
            new_ody = od_y[:current_index]
            oo_trace = go.Scattergl(x=new_oox, y=new_ooy, mode='markers', marker=dict(size=10,color = '#333333'), name='Original Offensive Player', showlegend=True, legendgroup='group1')
            od_trace = go.Scattergl(x=new_odx, y=new_ody, mode='markers', marker=dict(size=10,color = '#003366'), name='Original Defensive Player', showlegend=True, legendgroup='group1') 
            o_team_trace = go.Scattergl(x=new_otx,y = new_oty,mode='markers', marker=dict(size=10,color = 'red'),name = 'Offensive Team', showlegend=True, legendgroup='group1')
            d_team_trace = go.Scattergl(x=new_dtx,y = new_dty,mode='markers', marker=dict(size=10,color = 'blue'),name = 'Defensive Team', showlegend=True, legendgroup='group1')
            o_trace = go.Scattergl(x=new_ox, y=new_oy, mode='markers', marker=dict(size=10, color=o_marker_colors), name='Offensive Player', showlegend=True, legendgroup='group1')
            d_trace = go.Scattergl(x=new_dx, y=new_dy, mode='markers', marker=dict(size=10, color=d_marker_colors), name='Defensive Player', showlegend=True, legendgroup='group1')
            los_trace = go.Scattergl(x=los_x_arr, y=los_y_arr, mode='lines', name='Line of Scrimmage', showlegend=True, legendgroup='group1')
            data = [los_trace]
            if plot_type == 'play-full':
                data = [los_trace,o_team_trace,d_team_trace]#oo_trace,od_trace,o_trace,d_trace]
            if show_original:
                data.append(oo_trace)
                data.append(od_trace)
            data.append(o_trace)
            data.append(d_trace)
            fig = go.Figure(data=data,
                    layout=go.Layout(title='Trajectory Plot', xaxis=dict(title='X',range=[-5,125]), 
                    yaxis=dict(title='Y',range=[-5,58.3]),
                    legend=dict(x=1, y=1, traceorder='normal', font=dict(family='sans-serif', size=12, color='#000'),orientation = 'h',xanchor = "right",yanchor="bottom"),
                    shapes=[
                        dict(
                            type='rect',
                            x0=0, y0=0, x1=120, y1=53.3,
                            line=dict(color='white', width=2),
                            fillcolor='#B4EEB4',
                            layer = 'below'
                        )
                    ]
                )
            )
            return fig,f"Viewing Plot for ID: {index}"
        else:
            oo_trace = go.Scattergl(x=oo_x, y=oo_y, mode='markers', marker=dict(size=10,color = '#333333'), name='Original Offensive Player', showlegend=True, legendgroup='group1')
            od_trace = go.Scattergl(x=od_x, y=od_y, mode='markers', marker=dict(size=10, color = '#003366'), name='Original Defensive Player', showlegend=True, legendgroup='group1') 
            o_team_trace = go.Scattergl(x=off_seq_x,y = off_seq_y,mode='markers', marker=dict(size=10,color = 'red'),name = 'Offensive Team', showlegend=True, legendgroup='group1')
            d_team_trace = go.Scattergl(x=def_seq_x,y = def_seq_y,mode='markers', marker=dict(size=10,color = 'blue'),name = 'Defensive Team', showlegend=True, legendgroup='group1')
            o_trace = go.Scattergl(x=o_x, y=o_y, mode='markers', marker=dict(size=10, color=o_marker_colors), name='Offensive Player', showlegend=True, legendgroup='group1')
            d_trace = go.Scattergl(x=d_x, y=d_y, mode='markers', marker=dict(size=10, color=d_marker_colors), name='Defensive Player', showlegend=True, legendgroup='group1')
            los_trace = go.Scattergl(x=los_x_arr, y=los_y_arr, mode='lines', name='Line of Scrimmage', showlegend=True, legendgroup='group1')
            data = [los_trace]
            if plot_type == 'play-full':
                data = [los_trace,o_team_trace,d_team_trace]#oo_trace,od_trace,o_trace,d_trace]
            if show_original:
                data.append(oo_trace)
                data.append(od_trace)
            data.append(o_trace)
            data.append(d_trace)
            fig = go.Figure(data=data,
                layout=go.Layout(title='Trajectory Plot', xaxis=dict(title='X',range=[-5,125]), 
                yaxis=dict(title='Y',range=[-5,58.3]),
                legend=dict(x=1, y=1, traceorder='normal', font=dict(family='sans-serif', size=12, color='#000'),orientation = 'h',xanchor = "right",yanchor="bottom"),
                shapes=[
                    dict(
                        type='rect',
                        x0=0, y0=0, x1=120, y1=53.3,
                        line=dict(color='white', width=2),
                        fillcolor='#B4EEB4',
                        layer = 'below'
                    )
                ]
            )
        )
            return fig,f"Viewing Plot for ID: {index}"
    
    off_seq_x,off_seq_y,def_seq_x,def_seq_y,oo_x,oo_y,od_x,od_y,flip_x,flip_y = generate_play_trajectories(index,los_x)
    o_x, o_y,d_x,d_y = generate_trajectory(los_x-1, y_pos,start_x,start_y,index,los_x,flip_x,flip_y)

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
    o_team_trace = go.Scattergl(x=off_seq_x,y = off_seq_y,mode='markers', marker=dict(size=10,color = 'red'),name = 'Offensive Team', showlegend=True, legendgroup='group1')
    d_team_trace = go.Scattergl(x=def_seq_x,y = def_seq_y,mode='markers', marker=dict(size=10,color = 'blue'),name = 'Defensive Team', showlegend=True, legendgroup='group1')
    oo_trace = go.Scattergl(x=oo_x, y=oo_y, mode='markers', marker=dict(size=10,color = '#333333'), name='Original Offensive Player', showlegend=True, legendgroup='group1')
    od_trace = go.Scattergl(x=od_x, y=od_y, mode='markers', marker=dict(size=10,color = '#003366'), name='Original Defensive Player', showlegend=True, legendgroup='group1')
    o_trace = go.Scattergl(x=o_x, y=o_y, mode='markers', marker=dict(size=10, color=o_marker_colors), name='Offensive Player', showlegend=True, legendgroup='group1')
    d_trace = go.Scattergl(x=d_x, y=d_y, mode='markers', marker=dict(size=10, color=d_marker_colors), name='Defensive Player', showlegend=True, legendgroup='group1')
    los_trace = go.Scattergl(x=los_x_arr, y=los_y_arr, mode='lines', name='Line of Scrimmage', showlegend=True, legendgroup='group1')
    data = [los_trace]
    if plot_type == 'play-full':
        data = [los_trace,o_team_trace,d_team_trace]#oo_trace,od_trace,o_trace,d_trace]
    if show_original:
        data.append(oo_trace)
        data.append(od_trace)
    data.append(o_trace)
    data.append(d_trace)
    fig = go.Figure(data=data, 
        layout=go.Layout(title='Trajectory Plot', xaxis=dict(title='X',range=[-5,125]), 
                yaxis=dict(title='Y',range=[-5,58.3]),
                legend=dict(x=1, y=1, traceorder='normal', font=dict(family='sans-serif', size=12, color='#000'),orientation = 'h',xanchor = "right",yanchor="bottom"),
                shapes=[
                    dict(
                        type='rect',
                        x0=0, y0=0, x1=120, y1=53.3,
                        line=dict(color='white', width=2),
                        fillcolor='#B4EEB4',
                        layer = 'below'
                    )
                ]
            )
        )
    return fig,f"Viewing Plot for ID: {index}"


@app.callback(
    Output('play-button','children'),
    [Input('interval-component', 'n_intervals'),
    Input('interval-component','disabled')]
)
def update_button_text(n_intervals,disabled):
    if disabled:
        return "View Play in Real Time"
    return f"{n_intervals*0.5} s"

@app.callback(
    Output('cluster-graph-1', 'figure'),
    Output('cluster-graph-2', 'figure'),
    Output('cluster-graph-3', 'figure'),
    Output('cluster-graph-4', 'figure'),
    Output('cluster-graph-5', 'figure'),
    Output('cluster-graph-6', 'figure'),
    Output('cluster-graph-7', 'figure'),
    Output('cluster-graph-8', 'figure'),
    Output('cluster-header','style'),
    Output('cluster-center-dropdown','style'),
    Output('seq-header','style'),
    [Input('cluster-dropdown','value')]
    )
def update_cluster_figure(value):
    cluster_figures = []
    if value in range(8):
        for i in range(8):
            current_c_index = sequences_ordered[value][i][1]
            seq = o_player_sequences[current_c_index]
            c_x = [x[0] for x in seq]
            c_y = [x[1] for x in seq]   
            c_trace = go.Scatter(x=c_x, y=c_y, mode='markers', name=f'Sequence {i}')
            cluster_figures.append(go.Figure(data=[c_trace], layout=go.Layout(title=f'Sequence {i}', xaxis=dict(title='X',range=[0,20]), yaxis=dict(title='Y',range=[-20,20]))))
        return cluster_figures[0],cluster_figures[1],cluster_figures[2],cluster_figures[3],cluster_figures[4],cluster_figures[5],cluster_figures[6],cluster_figures[7],{},{},{}

    for i in range(len(sequences_ordered)):
        current_c_index = sequences_ordered[i][0][1]
        seq = o_player_sequences[current_c_index]
        c_x = [x[0] for x in seq]
        c_y = [x[1] for x in seq]   
        c_trace = go.Scatter(x=c_x, y=c_y, mode='markers', name=f'Cluster {i}')
        cluster_figures.append(go.Figure(data=[c_trace], layout=go.Layout(title=f'Cluster {i} Sequence', xaxis=dict(title='X',range=[0,20]), yaxis=dict(title='Y',range=[-20,20]))))
    return cluster_figures[0],cluster_figures[1],cluster_figures[2],cluster_figures[3],cluster_figures[4],cluster_figures[5],cluster_figures[6],cluster_figures[7],{},{'display':'none'},{'display':'none'}

@app.callback(
    Output('cluster-center-dropdown','value'),
    [Input('cluster-dropdown','value')],
)
def reset_centerdrd(value_cluster):
    return None

@app.callback(
    Output('seq-index', 'value'),
    [Input('cluster-center-dropdown','value'),
    Input('play-dropdown','value')],
    State('cluster-dropdown','value'),
    State('seq-index', 'value'),
    State('game-dropdown','value')
)
def set_cluster_seq(c_seq,playId,cluster,index,gameId):
    if playId is not None and gameId is not None:
        return player_pair_ids[(gameId,playId)][0]
    if cluster is not None and c_seq is not None:
        return sequences_ordered[cluster][c_seq][1]
    else:
        return index

@app.callback(
    Output('play-dropdown', 'options'),
    Output('play-dropdown','style'),
    Output('play-dropdown','value'),
    [Input('game-dropdown','value')],
)
def set_cluster_seq(value):
    if value is None:
        return [],{'display':'none'},None
    game_play_df = valid_plays[(valid_plays['gameId'] == value)]
    play_drd_options = []
    for row in game_play_df.itertuples():
        play_drd_options.append({'label':row.playDescription,'value':row.playId})
    return play_drd_options,{},None

@app.callback(
        Output('play-mode', 'style'),
        Output('cluster-mode','style'),
        #Output(''),
        Input('tabs', 'value')
)
def render_tab_content(tab_clicked):
    if tab_clicked == 'seq-selection':
        return {'display':'none'},{'display':'none'}
    elif tab_clicked == 'cluster-selection':
        #returning cluster selection options
        return {'display':'none'},{}
    elif tab_clicked == 'play-selection':
        return {},{'display':'none'}

@app.callback(
    Output('game-dropdown','value'),
    Output('cluster-dropdown','value'),
    [Input('tabs', 'value')]
)
def set_cluster_seq(value):
    return None,None


if __name__ == '__main__':
    app.run_server(debug=True)