from dash import Dash, html, dcc,no_update
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
from tensorflow import keras,data as tfdata
from numpy import polyfit,linspace,poly1d
from pandas import read_csv
import dash_bootstrap_components as dbc
from os import path,environ
from pymongo import MongoClient
from flask_caching import Cache

#flask_server = Flask(__name__)
app = Dash(__name__,suppress_callback_exceptions=True,external_stylesheets=[dbc.themes.SIMPLEX, dbc.icons.BOOTSTRAP])
app.title = "NFL Defensive Trajectory Prediction"

my_dir = path.dirname(__file__)
mongo_username = environ.get("MONGO_USERNAME")
mongo_password = environ.get("MONGO_PASSWORD")
mclient = MongoClient(f"mongodb+srv://{mongo_username}:{mongo_password}@cpsc502.wv2dsv5.mongodb.net/?retryWrites=true&w=majority")
db = mclient.CPSC502
cluster_col = db.Cluster
seq_col = db.Sequence

play_fields = {
    "gameId":"Game identifier, unique.",
    "playId":"Play identifier, not unique across games.",
    "playDescription": "Description of play.",
    "quarter": "Game quarter.",
    "down": "Down (number).",
    "yardsToGo": "Distance needed for a first down.",
    "possessionTeam": "Team abbreviation of the team on offense with possession of the ball.",
    "defensiveTeam": "Team abbreviation of the team on defense.",
    "yardlineSide": "3-letter team code corresponding to line-of-scrimmage (los).",
    "yardlineNumber": "Yard line at line-of-scrimmage (los).",
    "gameClock": "Time on clock of play (MM:SS).",
    "preSnapHomeScore": "Home score prior to the play.",
    "preSnapVisitorScore": "Visiting team score prior to the play.",
    "passResult": "Dropback outcome of the play (C: Complete pass, I: Incomplete pass, S: Quarterback sack, IN: Intercepted pass, R: Scramble).",
    "penaltyYards": "yards gained by offense by penalty.",
    "prePenaltyPlayResult": "Net yards gained by the offense, before penalty yardage.",
    "playResult": "Net yards gained by the offense, including penalty yardage.",
    "foulName1": "Name of the 1st penalty committed during the play.",
    "foulNFLId1": "nflId of the player who comitted the 1st penalty during the play.",
    "absoluteYardlineNumber": "Distance from end zone for possession team.",
    "offenseFormation": "Formation used by possession team."
}

num_sequences = seq_col.count_documents({})


model = keras.models.load_model(f"{my_dir}/../Notebooks/models/LSTMModel2LayerFull.h5")



valid_games = read_csv("./data/valid_games.csv")
valid_plays = read_csv("./data/valid_plays.csv")



o_padding_value = [-1.0 for _ in range(9)] # Number of features in the offensive player sequences

masking_o_layer = keras.layers.Masking(mask_value=-1)
d_empty = [[0,0] for _ in range(90)]

### ==================================================================================




cluster_options = [{"label": f"Cluster Center {i+1}", "value": i} for i in range(8)]
cluster_center_options = [{"label": f"Sequence {i+1}", "value": i} for i in range(8)]
game_options = []
for row in valid_games.itertuples():
    game_options.append({"label":f"{row.homeTeamAbbr} vs. {row.visitorTeamAbbr} playing home to {row.homeTeamAbbr} on {row.gameDate}","value":row.gameId})

yard_lines = []
yard_line_annotations = []
current_yard = 0
decrease = False
for x in range(20, 110, 10):
    if current_yard >= 50:
        decrease = True
    if decrease:
        current_yard -= 10
    else:
        current_yard += 10
    yard_line_annotations.append(dict(x=x, y=-2, showarrow=False, text=current_yard,font={"color":"#333333","size":20}))
    yard_lines.append({
        "type": "line",
        "xref": "x",
        "yref": "y",
        "x0": x,
        "y0": 0,
        "x1": x,
        "y1": 53.3,
        "line": {"color": "white", "width": 4},
        "layer":"below",
        "editable":False
    })

rects = [
        dict(
            type="rect",
            x0=10, y0=0, x1=110, y1=53.3,
            line=dict(color="white", width=4),
            fillcolor="#B4EEB4",
            layer = "below",
            editable = False
        ),
        dict(
            type="rect",
            x0=0, y0=0, x1=10, y1=53.3,
            line=dict(color="#333333", width=2),
            fillcolor="#333333",
            layer = "below",
            editable = False
        ),
        dict(
            type="rect",
            x0=110, y0=0, x1=120, y1=53.3,
            line=dict(color="#333333", width=2),
            fillcolor="#333333",
            layer = "below",
            editable = False
        )
]
yard_line_annotations.append(dict(x=5, y=26.65, showarrow=False, text= "ENDZONE",font={"color":"white","size":30},textangle = -90))
yard_line_annotations.append(dict(x=115, y=26.65, showarrow=False, text= "ENDZONE",font={"color":"white","size":30},textangle = 90))


cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache-directory'
})

TIMEOUT = 60

@cache.memoize(timeout=TIMEOUT)
def get_doc(index):
    return seq_col.find_one({"seqIndex":index})

# define a function that generates x and y arrays based on user input
def generate_trajectory(start_x,start_y, d_start_x,d_start_y,index,los_x,flip_x,flip_y,smooth):
    index = int(index)
    if index<0 or index >= num_sequences:
        index = 0
    seq_doc = get_doc(index)
    o_seq = seq_doc["seqOSeqFull"]
    o_seq_start_x = seq_doc["seqOArrX"][0]
    o_seq_start_y = seq_doc["seqOArrY"][0]
    if d_start_x < 0:
        d_start_x = seq_doc["seqDArrX"][0]
        if flip_x:
            d_start_x = start_x-(d_start_x - o_seq_start_x)
        else:
            d_start_x = start_x+(d_start_x - o_seq_start_x)
    if d_start_y < 0:
        d_start_y = seq_doc["seqDArrY"][0]
        if flip_y:
            d_start_y = start_y-(d_start_y - o_seq_start_y)
        else:
            d_start_y = start_y+(d_start_y - o_seq_start_y)
    o_sequence_len = len(o_seq)
    o_left_sideline_distance = 53.3 - start_y
    if flip_x:
        endzone_distance = abs(start_x)
    else:
        endzone_distance = abs(120-start_x)

    for i in range(o_sequence_len):
        o_seq[i][3] = abs(d_start_x-start_x)
        o_seq[i][4] = abs(d_start_y-start_y)
        o_seq[i][5] = los_x
        o_seq[i][6] = o_left_sideline_distance
        o_seq[i][7] = start_y
        o_seq[i][8] = endzone_distance
    padded_o_seq = keras.preprocessing.sequence.pad_sequences([o_seq],padding="post", value=o_padding_value, dtype="float32",maxlen = 90)
    m_o_seq = masking_o_layer(padded_o_seq)
    predict_dataset = tfdata.Dataset.from_tensor_slices((m_o_seq,[d_empty]))
    predict_dataset_batched = predict_dataset.batch(1)
    predicted_sequence = model.predict(predict_dataset_batched)
    predicted_sequence = predicted_sequence[0]
    predicted_start_x = predicted_sequence[0][0]
    predicted_start_y = predicted_sequence[0][1]
    predicted_sequence = [[x[0] - predicted_start_x,x[1]-predicted_start_y] for x in predicted_sequence]
    if flip_x:
        predicted_sequence_x = [-x[0]+d_start_x for x in predicted_sequence]
        x = [-x[0]+start_x for x in o_seq]
    else:
        predicted_sequence_x = [x[0]+d_start_x  for x in predicted_sequence]
        x = [x[0]+start_x for x in o_seq]
    
    if flip_y:
        predicted_sequence_y = [-x[1]+d_start_y for x in predicted_sequence]
        y = [-x[1]+start_y for x in o_seq]
    else:
        predicted_sequence_y = [x[1]+d_start_y for x in predicted_sequence]
        y = [x[1]+start_y for x in o_seq]
    predicted_sequence_x = predicted_sequence_x[0:o_sequence_len]
    predicted_sequence_y = predicted_sequence_y[0:o_sequence_len]


    if smooth:
        z = polyfit(predicted_sequence_x, predicted_sequence_y, 3)
        f = poly1d(z)

        # calculate new x"s and y"s
        x_new = linspace(predicted_sequence_x[0], predicted_sequence_x[-1], 50)
        y_new = f(x_new)
    else:
        x_new = predicted_sequence_x
        y_new = predicted_sequence_y
    return x, y,x_new,y_new
    

def generate_play_trajectories(index,los_x):
    play = get_doc(index)
    play_info = valid_plays[(valid_plays["gameId"] == play["seqGameID"])&(valid_plays["playId"] == play["seqPlayID"])]
    actual_los = play_info.iloc[0]["absoluteYardlineNumber"]
    original_o_player_x,original_o_player_y = play["seqOArrX"],play["seqOArrY"]
    original_d_player_x,original_d_player_y = play["seqDArrX"],play["seqDArrY"]
    
    if len(original_o_player_x) == 0 or len(original_d_player_x) == 0:
        return [],[],[],[],[],[],[],[],False,False
    
    off_seq_x,off_seq_y = play["seqOTeamX"],play["seqOTeamY"]
    def_seq_x,def_seq_y = play["seqDTeamX"],play["seqDTeamY"]

    off_seq_x = [x-actual_los+los_x for x in off_seq_x]
    def_seq_x = [x-actual_los+los_x for x in def_seq_x]
    
    flip_x = actual_los < original_o_player_x[0] # on the right side of los
    flip_y = 26.65 < original_o_player_y[0] # above halfway line

    original_o_player_x = [x-actual_los+los_x for x in original_o_player_x]
    original_d_player_x = [x-actual_los+los_x for x in original_d_player_x]
    
    return off_seq_x,off_seq_y,def_seq_x,def_seq_y,original_o_player_x,original_o_player_y,original_d_player_x,original_d_player_y,flip_x,flip_y



def los(x):
    x = [x,x]
    y = [0,53.3]
    return x,y
data = []

layout = go.Layout(title="Trajectory Plot", xaxis=dict(title="X",showgrid=False,showticklabels=False), yaxis=dict(title="Y",showgrid=False,showticklabels=False), legend=dict(x=1, y=1, traceorder="normal", font=dict(family="sans-serif", size=12, color="#000")),plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",clickmode="none")

fig = go.Figure(data=data, layout=layout)

app.layout = html.Div([
    html.Div(id="hidden-div", style={"display": "none"}),
    html.Div(
        [
            html.Span(
                [
                    html.I(className="bi bi-info-circle-fill me-2", style={"margin-right": "0.5rem"}), 
                ],
                style= {"cursor": "pointer","font-size":"30px", "zIndex": "9999"},
                id="info-icon",
            ),
            dbc.Tooltip("Click for more information.",target="info-icon",placement="top"),
            dbc.Modal([
                    dbc.ModalHeader(html.Strong("Application Information")),
                    dbc.ModalBody(children=[
                        html.Span("The available Trajectory Selection modes are listed below. You are able to flip through these modes using the first set of tabs. Changing the selected sequence in any of these modes \
                                        will  generate a new prediction using the model and change the full play displayed  to match up the sequence to the appropriate play it belongs to. "),
                        html.Span(""),
                        html.Ul([
                            html.Li([
                                html.Span([
                                    html.Strong("Sequence Selection Mode:"), " Using the ",html.Strong("Sequence Index") ,f" text input you are able to select any available sequence index in the range [0,{num_sequences-1}],\
                                        The available sequences contain all sequences gathered from the Big Data Bowl data, which were used in model training and preprocessed in the project GitHub."
                                ])
                            ]),
                            html.Li([
                                html.Span([
                                    html.Strong("Cluster Selection Mode:"), " Using the dropdown menus below the ",html.Strong("Trajectory Plot")," a user may view and select one of the eight available cluster centers.\
                                         These cluster centers represent common trajectories ran by offensive players in all preprocessed offensive sequences. The clustering algorithm used Dynamic Time Warping (DTW) for accurate \
                                        clustering of sequences. Once a cluster center is selected, the top 8 sequences closest to that cluster center will be displayed. A user may then select a sequence close to that cluster sequence \
                                        to display the trajectory on the plot."
                                ])
                            ]),
                            html.Li([
                                html.Span([
                                    html.Strong("Play Selection Mode:"), " Using the dropdown menus below the ",html.Strong("Trajectory Plot")," a user may view the available games from the selectable sequences. The user may select a game \
                                        and view the plays in that game to display the sequences from the selected play on the plot."
                                ])
                            ])
                        ]),
                        html.Span("The available Trajectory Visualization modes are listed below. You are able to flip through these modes using the second set of tabs."),
                        html.Span(""),
                        html.Ul([
                            html.Li([
                                html.Span([
                                    html.Strong("Isolated Play Visualization:"), " Selecting the ", html.Strong("Isolated Play Visualization"), " tab will hide the sequences of other offensive/defensive players on the play to show the \
                                        selected offensive/defensive player pair in isolation. The offensive trajectory and predicted defensive player trajectory will always be visible, and the original trajectories can be toggled on/off \
                                            to compare with the model predictions."
                                ])
                            ]),
                            html.Li([
                                html.Span([
                                    html.Strong("Full Play Visualization:"), " Selecting the ", html.Strong("Full Play Visualization"), " tab will show the sequences of other offensive/defensive players on the play to show the play in full. \
                                        This mode will hide the original offensive/defensive trajectories which can be toggled on/off to compare with the model predictions."
                                ])
                            ])
                        ]),
                        html.Span("In addition to these modes, the user may make adjustments to the positioning of the sequences using the rest of the controls. Hover over a control to view more information about the effects of the control. \
                                  Users may also show/hide the original sequence to compare with model results, and play the sequences on the plot in real time to visualize the play motion.")
                        ]
                    ,id = "modal-content"),
                    dbc.ModalFooter(
                    )],
                    id="info-modal",
                    size="lg",
                    is_open=False
                ),
            ],
        style={"position": "fixed", "bottom": "20px", "right": "20px"},
    ),
    dbc.Container([
        html.H1(
            children="",
            className="center-align"
        ),
        html.Div(
            className="card-panel m-5",
            children=[
                html.H1("NFL Defensive Trajectory Predictive Tool",className = "m-4 display-3"),
                html.P(
                "This application is for academic and research purposes only.",
                className="lead m-2",
                ),
                html.Hr(className="my-2"),
                html.P(
                    children=["This application was created as a proof of concept for a Computer Science Research Project at the University of Calgary.\
                               The purpose of this application is to demonstrate the usage of a Long-Short Term Memory (LSTM) TensorFlow model for sequence to sequence prediction of \
                              football players, using real-time NFL data from the ",
                                 html.A("2023 Big Data Bowl competition",href="https://www.kaggle.com/competitions/nfl-big-data-bowl-2023/data"),
                              ". The full GitHub repository with data preprocessing, model architecture, etc... can be found ",html.A("here",href="https://github.com/alejooescobar/CPSC502") ,
                              ". All libraries used to produce this application including ", html.A("Dash",href="https://dash.plotly.com/introduction"),", ",html.A("TensorFlow",href="https://www.tensorflow.org/"),
                              ", "," etc... are Open-source and free to use. Please send any inquiries or suggestions to Alejandro Escobar (",html.A("alejoesc2000@gmail.com",href="mailto:alejoesc2000@gmail.com"),")."],
                    className="m-4"
                ),
                html.P(
                    className = "m-4",
                    children=["The predictive capabilities of the model are demonstrated in the ",html.B("Trajectory Plot"),". The LSTM Model used in this project predicts the trajectory sequence of a defensive player in response to the movement of an\
                              offensive player. The sequences used for training and visualization are gathered from the Big Data Bowl data and filtered by official position to isolate the Wide Receivers as the only possible offensive player sequences. \
                              The responding defensive player is assigned to a corresponding offensive sequence using euclidean distance comparisons at various times throughout the sequence. The possible defensive positions are filtered to Tight End,\
                              Defensive Back and Cornerback. This is important to accurately identify defensive players that are man-marking an offensive player, which improves model accuracy for defensive trajectory prediction."]
                )
            ]
        )
    ],
    id = "info-component",
    fluid = True,
    className = "py-3 text-center"),

    html.Div([
    dbc.Tabs(id="tabs", active_tab="seq-selection", children=[
        dbc.Tab(label="Sequence Selection", tab_id="seq-selection",tab_class_name="mx-auto",style = {"font-size":"30px"}),
        dbc.Tab(label="Cluster Selection", tab_id="cluster-selection",tab_class_name="mx-auto",style = {"font-size":"30px"}),
        dbc.Tab(label="Play Selection", tab_id="play-selection",tab_class_name="mx-auto",style = {"font-size":"30px"}),
        ],        
        style={
            "width": "100%",
            "display": "flex",
            "justify-content": "center",
            "margin": "20px 0",
        }),
    ]),
    html.Div([
        dbc.Tabs(id="plot-tabs", active_tab="play-isolated", children=[
            dbc.Tab(label="Isolate Single Pair", tab_id="play-isolated",tab_class_name="mx-auto",style = {"font-size":"30px"}),
            dbc.Tab(label="Full Play", tab_id="play-full",tab_class_name="mx-auto",style = {"font-size":"30px"}),
        ],     
        style={
            "width": "100%",
            "display": "flex",
            "justify-content": "center",
            "margin": "20px 0",
        }) 
    ]),
    html.Div([
            html.Div([
                html.Div([
                        dbc.Spinner(
                            dcc.Graph(
                                id="trajectory-graph", figure=fig, 
                                style={"height": "800px","z-index": "9000"}
                            ),
                            type="grow",
                            delay_hide=100,
                            show_initially=True,
                            size= "md",
                            color="primary"
                        )
                        ]
                    ),
                    html.Div([
                    dcc.Slider(
                        id="los-slider",
                        min=0,
                        max=120,
                        step=1,
                        value=60,
                        marks={
                            0: {"label": "0", "style": {"color": "#77b0b1"}},
                            20: {"label": "20"},
                            40: {"label": "40"},
                            60: {"label": "60"},
                            80: {"label": "80"},
                            100: {"label": "100"},
                            120: {"label": "120", "style": {"color": "#f50"}}

                        }
                    )],style={"width":"94%","margin":"auto"}),
                    html.Div([
                    dcc.Slider(
                        id="ostart-slider",
                        min=0,
                        max=120,
                        step=1,
                        value=59,
                        marks={
                            0: {"label": "0", "style": {"color": "#77b0b1"}},
                            20: {"label": "20"},
                            40: {"label": "40"},
                            60: {"label": "60"},
                            80: {"label": "80"},
                            100: {"label": "100"},
                            120: {"label": "120", "style": {"color": "#f50"}}

                        }
                    )],style={"width":"94%","margin":"auto"})   
                     
                    ],style={"display": "inline-block", "width": "95%"}),
                    dbc.Tooltip(
                        "Adjust the line of scrimmage (LOS) to shift the play and sequences accordingly.",
                        target="los-slider",
                        placement="bottom",
                    ),
                    dbc.Tooltip(
                        "Adjust the starting x position of the offensive player.",
                        target="ostart-slider",
                        placement="bottom",
                    ),
                    html.Div([
                    dcc.Slider(
                        id="y-slider",
                        min=0,
                        max=53,
                        step=1,
                        value=26,
                        vertical = True,
                        marks={
                            0: {"label": "0", "style": {"color": "#77b0b1"}},
                            10: {"label": "10"},
                            20: {"label": "20"},
                            30: {"label": "30"},
                            40: {"label": "40"},
                            50: {"label": "50", "style": {"color": "#f50"}},
                        },
                        verticalHeight = 700
                    ),
                    dbc.Tooltip(
                        "Adjust the starting y position of the offensive player. This will subsequently shift the defensive player starting position.",
                        target="y-slider",
                        placement="left",
                    )]
            ,style={"display": "inline-block", "width": "5%","marginBottom":"3%"}),
    ]),
    html.Div([
        dbc.Button(id="adjust-defender-button", children="View Suggested Defender Starting Positions"),
        dbc.Button(id="play-info-button", children="View Play Information"),
        dbc.Modal(
            [
                dbc.ModalHeader(html.Strong("Play Information")),
                dbc.ModalBody(children=[],
                              id="play-info-modal-body"),
                dbc.ModalFooter(
                    ""
                ),
            ],
            id="play-info-modal",
            size="xl"
        )

    ],className = "d-grid gap-2 col-6 mx-auto my-5"),
    dbc.Container([
        dbc.Form(
            dbc.Row(
                [
                    dbc.Tooltip(
                        "The offset of the defensive player's starting position x coordinate, relative to the offensive player's starting position x coordinate.",
                        target="defensive-x-label",
                        placement="top",
                    ),
                    dbc.Col(
                        [dbc.Label("Defensive Start X",width = "auto",id="defensive-x-label",style={"textDecoration": "underline", "cursor": "pointer"},className="d-flex justify-content-center align-items-center"),
                        dbc.Input(id="start-x", value=-1,type="number",className="d-flex justify-content-center align-items-center")]
                    ),
                    dbc.Tooltip(
                        "The offset of the defensive player's starting position y coordinate, relative to the offensive players starting position y coordinate.",
                        target="defensive-y-label",
                        placement="top",
                    ),
                    dbc.Col(
                        [dbc.Label("Defensive Start Y", width="auto",id="defensive-y-label",style={"textDecoration": "underline", "cursor": "pointer"},className="d-flex justify-content-center align-items-center"),
                        dbc.Input(id="start-y", value=-1,type="number",className="d-flex justify-content-center align-items-center")]
                    )
                ]
            )
        ,className="my-4"),
        dbc.Form(
            dbc.Row(
                [
                    dbc.Tooltip(
                        f"Select an index within the range [0,{num_sequences-1}] to display an offensive sequence and the model-predicted sequence.",
                        target="seq-index-label",
                        placement="top",
                    ),
                    dbc.Col(
                        [dbc.Label("Sequence Index",width = "auto",id="seq-index-label",style={"textDecoration": "underline", "cursor": "pointer"},className="d-flex justify-content-center align-items-center"),
                        dbc.Input(id="seq-index", type="number", value=0,placeholder="Please select an integer to display a sequence on the Trajectory Plot.",className="d-flex justify-content-center align-items-center")],
                        align="center"
                    ),
                    dbc.Col(
                        [dbc.Label("Original Sequence Toggle", width="auto",id="seq-toggle",style={"textDecoration": "underline", "cursor": "pointer"},className="d-flex justify-content-center align-items-center"),
                        dbc.Checklist(
                            options=[
                                {"label": "Show Original", "value": True},
                            ],
                            value=[True],
                            id="original-switch",
                            switch=True,
                            className="d-flex justify-content-center align-items-center")],
                        align="center"
                        ),
                    dbc.Tooltip(
                        "Toggle to show or hide the original defensive/offensive player pair to compare with the model prediction.",
                        target="seq-toggle",
                        placement="top",
                    ),
                    dbc.Col([
                        dbc.Label("Predicted Sequence Smoothing Toggle", width="auto",id="smooth-toggle",style={"textDecoration": "underline", "cursor": "pointer"},className="d-flex justify-content-center align-items-center"),
                        dbc.Checklist(
                            options=[
                                {"label": "Smooth Curve", "value": True},
                            ],
                            value=[True],
                            id="smooth-switch",
                            switch=True,
                            className="d-flex justify-content-center align-items-center")
                        ],
                        align="center"
                        ),
                        dbc.Tooltip(
                            "Toggle to smooth the predicted curve or show the raw data points.",
                            target="smooth-toggle",
                            placement="top",
                        ),
                ]
            ,justify="center",align="center")
        ,className= "my-4")
    ],fluid=True),
    html.Div(id = "play-mode",children=[
                html.Div(id="extra-space-game",children=[],style={"height":"100px"}),
                html.H2("Please select a game below to view plays in the selected game.",id="game-select-header",className = "my-4 mx-auto text-center"),
                dcc.Dropdown(
                id="game-dropdown",
                options=game_options,
                value=None,
                placeholder="Select a Game in the season to view plays...",
                searchable=True,
                className = "mx-auto my-4 w-50"),
                html.H2("Please select a play below to display the play on the trajectory plot.",id="play-select-header",className = "my-4 mx-auto text-center"),
                dcc.Dropdown(
                id="play-dropdown",
                options=[],
                value=None,
                placeholder="Select a Play in the selected game to plot...",
                searchable=True,
                className = "mx-auto my-4 w-50"),
            ]),
    html.Div(
        id = "cluster-mode",
        children=[
            html.Div(id="extra-space-cluster",children=[],style={"height":"100px"}),
            html.Div(
                    children=[
                    html.H2("Please select a cluster center below to view sequences in the selected cluster.",id="cluster-header",className = "my-4 mx-auto text-center"),
                    dcc.Dropdown(
                        id="cluster-dropdown",
                        options=cluster_options,
                        value=None,
                        placeholder="Select a cluster center...",
                        searchable=False,
                        className = "mx-auto my-4 w-50"),
                    html.H2("Please select a sequence below to display on the trajectory plot.",id="seq-header",className = "my-4 mx-auto text-center"),
                    dcc.Dropdown(
                        id="cluster-center-dropdown",
                        options=cluster_center_options,
                        value=None,
                        placeholder="Select a sequence to display...",
                        searchable=False,
                        className = "mx-auto my-4 w-50"),
                    ]
                ),
            html.Div(id="cluster-div",children=[
                html.Div(children=[
                        dcc.Graph(
                            id="cluster-graph-1",  
                            config={"displayModeBar": False, "doubleClick": "reset"},
                            clickData={"points": [{"customdata": "Add Point"}]}
                    ),
                ], style={"display": "inline-block", "width": "24%", "textAlign": "center","position": "relative"}),
                html.Div(children=[
                        dcc.Graph(
                            id="cluster-graph-2", 
                            config={"displayModeBar": False, "doubleClick": "reset"},
                            clickData={"points": [{"customdata": "Add Point"}]}
                    ),
                ], style={"display": "inline-block", "width": "24%", "textAlign": "center"}),
                html.Div(children=[
                    dcc.Graph(
                        id="cluster-graph-3", 
                        config={"displayModeBar": False, "doubleClick": "reset"},
                        clickData={"points": [{"customdata": "Add Point"}]}
                    ),
                ], style={"display": "inline-block", "width": "24%", "textAlign": "center"}),
                html.Div(children=[
                    dcc.Graph(
                        id="cluster-graph-4",  
                        config={"displayModeBar": False, "doubleClick": "reset"},
                        clickData={"points": [{"customdata": "Add Point"}]}
                    ),
                ], style={"display": "inline-block", "width": "24%", "textAlign": "center"}),
                        html.Div(children=[
                    dcc.Graph(
                        id="cluster-graph-5", 
                        config={"displayModeBar": False, "doubleClick": "reset"},
                        clickData={"points": [{"customdata": "Add Point"}]}
                    ),
                ], style={"display": "inline-block", "width": "24%", "textAlign": "center"}),
                html.Div(children=[
                    dcc.Graph(
                        id="cluster-graph-6",  
                        config={"displayModeBar": False, "doubleClick": "reset"},
                        clickData={"points": [{"customdata": "Add Point"}]}
                    ),
                ], style={"display": "inline-block", "width": "24%", "textAlign": "center"}),
                html.Div(children=[
                    dcc.Graph(
                        id="cluster-graph-7", 
                        config={"displayModeBar": False, "doubleClick": "reset"},
                        clickData={"points": [{"customdata": "Add Point"}]}
                    ),
                ], style={"display": "inline-block", "width": "24%", "textAlign": "center"}),
                html.Div(children=[
                    dcc.Graph(
                        id="cluster-graph-8",  
                        config={"displayModeBar": False, "doubleClick": "reset"},
                        clickData={"points": [{"customdata": "Add Point"}]}
                    ),
                ], style={"display": "inline-block", "width": "24%", "textAlign": "center"})
            ],style={})
        ]
    ),
    html.Div(id="extra-space",children=[],style={"height":"400px"})
])


@app.callback(
    Output("trajectory-graph", "figure"),
    [Input("start-x", "value"),
     Input("start-y", "value"),
     Input("seq-index", "value"),
     Input("los-slider", "value"),
     Input("plot-tabs","active_tab"),
     Input("original-switch","value"),
     Input("adjust-defender-button","n_clicks"),
     Input("smooth-switch","value")],    
     State("ostart-slider","value"),
     State("y-slider","value"),
     
)
def update_figure(start_x, start_y,index,los_x,plot_type,show_original,n_clicks,smooth,o_start_x,y_pos):
    print(f"Triggered, {o_start_x} {y_pos} {start_x} {start_y}")
    if index is None:
        return no_update
    if index < 0 or index >= num_sequences:
        return no_update
    los_x_arr,los_y_arr = los(los_x)
    los_trace = go.Scatter(x=los_x_arr, y=los_y_arr, mode="lines", name="Line of Scrimmage", showlegend=True, legendgroup="group1")
    off_seq_x,off_seq_y,def_seq_x,def_seq_y,oo_x,oo_y,od_x,od_y,flip_x,flip_y = generate_play_trajectories(index,los_x)
    o_x, o_y,d_x,d_y = generate_trajectory(o_start_x, y_pos,start_x,start_y,index,los_x,flip_x,flip_y,smooth)

    
    inner_length = len(def_seq_x)//10
    off_seq_x_separate = []
    off_seq_y_separate = []
    counter = 0
    for _ in range(10):
        inner_x = []
        inner_y = []
        for _ in range(inner_length):
            inner_x.append(off_seq_x[counter])
            inner_y.append(off_seq_y[counter])
            counter += 1
        off_seq_x_separate.append(inner_x)
        off_seq_y_separate.append(inner_y)

    inner_length = len(def_seq_x)//10
    counter = 0
    def_seq_x_separate = []
    def_seq_y_separate = []
    for _ in range(10):
        inner_x = []
        inner_y = []
        for _ in range(inner_length):
            inner_x.append(def_seq_x[counter])
            inner_y.append(def_seq_y[counter])
            counter += 1
        def_seq_x_separate.append(inner_x)
        def_seq_y_separate.append(inner_y)

    #o_marker_colors = ["rgb(200, 200, 255)"]
    #d_marker_colors = ["rgb(255, 215, 0)"]
    #for i in range(1, len(o_x)):
    #    red_value = int(o_marker_colors[i-1].split(",")[0].split("(")[1]) - 2
    #    green_value = int(o_marker_colors[i-1].split(",")[1]) - 2
    #    #blue_value = int(marker_colors[i-1].split(",")[2].split(")")[0]) + 10
    #    o_marker_colors.append(f"rgb({red_value}, {green_value}, 255)")
    #for i in range(1, len(d_x)):
    #    red_value = int(d_marker_colors[i-1].split(",")[0].split("(")[1]) - 2
    #    green_value = int(d_marker_colors[i-1].split(",")[1]) - 2
    #    #blue_value = int(d_marker_colors[i-1].split(",")[2].split(")")[0]) -2
    #    d_marker_colors.append(f"rgb({red_value}, {green_value}, 0)")

    oo_trace = go.Scatter(x=oo_x, y=oo_y, mode="markers", marker=dict(size=10,color = "#333333"), name="Original Offensive Player", showlegend=True, legendgroup="group1")
    od_trace = go.Scatter(x=od_x, y=od_y, mode="markers", marker=dict(size=10,color = "#003366"), name="Original Defensive Player", showlegend=True, legendgroup="group1")
    o_trace = go.Scatter(x=o_x, y=o_y, mode="markers", marker=dict(size=10, color="#800000"), name="Offensive Player", showlegend=True, legendgroup="group1",selected=dict(marker=dict(color='red',size=10)))

    d_trace = go.Scatter(x=d_x, y=d_y, mode="markers", marker=dict(size=10, color="#FBC02D "), name="Defensive Player", showlegend=True, legendgroup="group1")
    
    if flip_x:
        d_position_x = []
        d_position_y = []
        i = los_x-1
        while i >= los_x-3:
            j = y_pos - 3
            while j <= y_pos + 3:
                d_position_x.append(i)
                d_position_y.append(j)
                j+=1
            i-=0.5
    else:
        d_position_x = []
        d_position_y = []
        i = los_x+1
        while i <= los_x+3:
            j = y_pos - 3
            while j <= y_pos + 3:
                d_position_x.append(i)
                d_position_y.append(j)
                j+=1
            i+=0.5

    d_position_trace = go.Scatter(x=d_position_x, y=d_position_y, mode="markers", marker=dict(size=10, color="green",symbol = "x"),text="Click to update the defensive starting position.", name="Defensive Player Positions", showlegend=True, legendgroup="group1")
    if n_clicks:
        fig = go.Figure(
            data=[o_trace,los_trace,d_position_trace], 
            layout=go.Layout(
                    title="Trajectory Plot", 
                    xaxis=dict(title="X",range=[min(los_x-2,o_x[0]-2,los_x+6,o_x[-1]-2),max(los_x+2,o_x[0]+2,los_x+6,o_x[-1])+2],showgrid=False,showticklabels=False), 
                    yaxis=dict(title="Y",range=[min(o_y[0]-2,y_pos-7,o_y[-1]-2),max(o_y[0]+2,y_pos+7,o_y[-1]+2)],showgrid=False,showticklabels=False),
                    legend=dict(x=1, y=1, traceorder="normal", font=dict(family="sans-serif", size=12, color="#000"),orientation = "h",xanchor = "right",yanchor="bottom"),
                    shapes=rects+yard_lines,
                    annotations=yard_line_annotations,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    clickmode="event"

                )
            )
        return fig
    
    frames = []
    i = 0
    current_index = 5
    inner_length = len(off_seq_x)//10
    seq_len = min(len(o_x),len(d_x),len(off_seq_x)//10,len(def_seq_x)//10,len(oo_x),len(oo_y))

    while current_index < seq_len:
        data = [los_trace,

                ]

        if plot_type == "play-full":
            for i in range(len(def_seq_x_separate)):
                data.append(go.Scatter(x=off_seq_x_separate[i][:current_index],y = off_seq_y_separate[i][:current_index]))
                data.append(go.Scatter(x=def_seq_x_separate[i][:current_index],y = def_seq_y_separate[i][:current_index]))

        if show_original:
            data.append(go.Scatter(x=oo_x[:current_index],y=oo_y[:current_index]))
            data.append(go.Scatter(x=od_x[:current_index],y=od_y[:current_index]))
        data.append(go.Scatter(x=o_x[:current_index], y=o_y[:current_index]))
        data.append(go.Scatter(x=d_x[:current_index],y=d_y[:current_index]))
        frames.append(go.Frame(data=data))
        current_index += 5
    data = [los_trace]
    if plot_type == "play-full":
        for i in range(len(def_seq_x_separate)):
            data.append(go.Scatter(x=off_seq_x_separate[i],y = off_seq_y_separate[i]))
            data.append(go.Scatter(x=def_seq_x_separate[i],y = def_seq_y_separate[i]))
    if show_original:
        data.append(go.Scatter(x=oo_x,y=oo_y))
        data.append(go.Scatter(x=od_x,y=od_y))
    data.append(go.Scatter(x=o_x, y=o_y))
    data.append(go.Scatter(x=d_x, y=d_y))
    frames.append(go.Frame(data=data))

    data = []
    if plot_type == "play-full":
        for i in range(len(def_seq_x_separate)):
            if i == 0:
                data.append(go.Scatter(x=off_seq_x_separate[i],y = off_seq_y_separate[i],mode="markers", marker=dict(size=10,color = "red"),name = "Offensive Team", showlegend=True, legendgroup="group1"))
                data.append(go.Scatter(x=def_seq_x_separate[i],y = def_seq_y_separate[i],mode="markers", marker=dict(size=10,color = "blue"),name = "Defensive Team", showlegend=True, legendgroup="group1"))
            else:
                data.append(go.Scatter(x=off_seq_x_separate[i],y = off_seq_y_separate[i],mode="markers", marker=dict(size=10,color = "red"),showlegend=False,name = "Offensive Team"))
                data.append(go.Scatter(x=def_seq_x_separate[i],y = def_seq_y_separate[i],mode="markers", marker=dict(size=10,color = "blue"),showlegend=False,name = "Defensive Team"))
    if show_original:
        data.append(oo_trace)
        data.append(od_trace)
    data = [los_trace] + data + [o_trace,d_trace]

    fig = go.Figure(
            data=data, 
            layout=go.Layout(
                    title="Trajectory Plot", 
                    xaxis=dict(title="X",range=[0,120],showgrid=False,showticklabels=False), 
                    yaxis=dict(title="Y",range=[-5,58.3],showgrid=False,showticklabels=False),
                    legend=dict(x=1, y=1, traceorder="normal", font=dict(family="sans-serif", size=12, color="#000"),orientation = "h",xanchor = "right",yanchor="bottom"),
                    shapes=rects+yard_lines,
                    annotations=yard_line_annotations,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    updatemenus=[
                        dict(
                        type="buttons",
                        buttons=[
                                dict(label="Play",
                                method="animate",
                                args=[None]
                                )
                            ],
                            direction="left",
                            pad={"l": 10, "t": 57},
                            showactive=True,
                            x=0,
                            xanchor="left",
                            y=1,
                            yanchor="top"
                        )
                    ],
                    clickmode="none"

                ),
                frames = frames,
            )
    return fig



@app.callback(
    Output("cluster-graph-1", "figure"),
    Output("cluster-graph-2", "figure"),
    Output("cluster-graph-3", "figure"),
    Output("cluster-graph-4", "figure"),
    Output("cluster-graph-5", "figure"),
    Output("cluster-graph-6", "figure"),
    Output("cluster-graph-7", "figure"),
    Output("cluster-graph-8", "figure"),
    Output("cluster-header","style"),
    Output("cluster-center-dropdown","style"),
    Output("seq-header","style"),
    [Input("cluster-dropdown","value")]
    )
def update_cluster_figure(value):
    cluster_figures = []
    i = 0
    if value in range(8):

        for seq in cluster_col.find({"clusterID":value}):
            c_trace = go.Scatter(x=seq["clusterSeqX"], y=seq["clusterSeqY"], mode="markers", name=f"Sequence {i+1}")
            cluster_figures.append(go.Figure(data=[c_trace], layout=go.Layout(title=f"Sequence {i+1}", xaxis=dict(title="X",range=[0,20]), yaxis=dict(title="Y",range=[-20,20]))))
            i+= 1
        return cluster_figures[0],cluster_figures[1],cluster_figures[2],cluster_figures[3],cluster_figures[4],cluster_figures[5],cluster_figures[6],cluster_figures[7],{},{},{}
    for seq in cluster_col.find({"clusterID":-1}):
        c_trace = go.Scatter(x=seq["clusterSeqX"], y=seq["clusterSeqY"], mode="markers", name=f"Cluster {i+1}")
        cluster_figures.append(go.Figure(data=[c_trace], layout=go.Layout(title=f"Cluster {i+1} Sequence", xaxis=dict(title="X",range=[0,20]), yaxis=dict(title="Y",range=[-20,20]))))
        i+=1
    return cluster_figures[0],cluster_figures[1],cluster_figures[2],cluster_figures[3],cluster_figures[4],cluster_figures[5],cluster_figures[6],cluster_figures[7],{},{"display":"none"},{"display":"none"}

@app.callback(
    Output("cluster-center-dropdown","value"),
    [Input("cluster-dropdown","value")],
)
def reset_centerdrd(value_cluster):
    return None

@app.callback(
    Output("seq-index", "value"),
    [Input("cluster-center-dropdown","value"),
    Input("play-dropdown","value")],
    State("cluster-dropdown","value"),
    State("seq-index", "value"),
    State("game-dropdown","value")
)
def set__seq(c_seq,playId,cluster,index,gameId):
    if playId is not None and gameId is not None:
        return seq_col.find_one({"seqGameID":gameId,"seqPlayID":playId})["seqIndex"]
    if cluster is not None and c_seq is not None:
        return cluster_col.find_one({"clusterID":cluster,"clusterOrder":c_seq})["clusterOriginalID"]
    else:
        return index

@app.callback(
    Output("play-dropdown", "options"),
    Output("play-dropdown","style"),
    Output("play-dropdown","value"),
    Output("play-select-header","style"),
    [Input("game-dropdown","value")],
)
def set_play_dropdown(value):
    if value is None:
        return [],{"display":"none"},None,{"display":"none"}
    game_play_df = valid_plays[(valid_plays["gameId"] == value)]
    play_drd_options = []
    for row in game_play_df.itertuples():
        play_drd_options.append({"label":row.playDescription,"value":row.playId})
    return play_drd_options,{},None,{}

@app.callback(
        Output("play-mode", "style"),
        Output("cluster-mode","style"),
        Input("tabs", "active_tab")
)
def render_tab_content(tab_clicked):
    if tab_clicked == "seq-selection":
        return {"display":"none"},{"display":"none"}
    elif tab_clicked == "cluster-selection":
        #returning cluster selection options
        return {"display":"none"},{}
    elif tab_clicked == "play-selection":
        return {},{"display":"none"}

@app.callback(
    Output("game-dropdown","value"),
    Output("cluster-dropdown","value"),
    [Input("tabs", "active_tab")]
)
def set_cluster_seq(value):
    return None,None

@app.callback(
    Output("info-modal", "is_open"),
    [Input("info-icon", "n_clicks")]
)
def toggle_modal(n_clicks):
    if n_clicks is None:
        return False
    return True

@app.callback(
    Output("play-info-modal-body","children"),
    [Input("seq-index", "value")]
)
def update_modal_body(index):
    if index is None:
        return no_update
    if index < 0 or index > num_sequences:
        return no_update
    modal_children = [html.Span(["Here you can find play information collected for the displayed trajectories on the ",html.Strong("Trajectory Plot"),". This information is collected \
                    from the Big Data Bowl dataset and games/plays are uniquely identified by their assigned Big Data Bowl ID's."]),html.Span("")]
    seq = get_doc(index)
    playId = seq["seqPlayID"]
    gameId = seq["seqGameID"]
    play = valid_plays[(valid_plays["gameId"] == gameId)&(valid_plays["playId"] == playId)]
    if play.empty:
        modal_children.append(html.Span(f"No information is available for the play with gameId {gameId} and playId {playId}"))
        return modal_children
    table_rows = []
    for field in play_fields:
        table_rows.append(
            html.Tr([
                html.Td(field),
                html.Td(play.iloc[0][field]),
                html.Td(play_fields[field])
            ])
        )

    modal_children.append(dbc.Table(
        [html.Thead(["Field", "Value", "Field Description"])]+
        [html.Tbody(table_rows)],
        bordered=True,
        striped=True,
        hover=True,
        responsive=True,
        className="m-auto",
    ))
    return modal_children

@app.callback(
    Output("play-info-modal", "is_open"),
    [Input("play-info-button", "n_clicks")],
    [State("play-info-modal", "is_open")],
)
def toggle_modal(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

    
@app.callback(
    Output("los-slider","value"),
    Output("ostart-slider","value"),
    Output("y-slider","value"),
    [Input("seq-index","value")],
)
def display_click_data(index):
    los = no_update
    o_x = no_update
    o_y = no_update
    los = 60
    if index:
        seq_doc = get_doc(index)
        play_info = valid_plays[(valid_plays["gameId"] == seq_doc["seqGameID"])&(valid_plays["playId"] == seq_doc["seqPlayID"])]
        los = play_info.iloc[0]["absoluteYardlineNumber"]
        o_x = seq_doc["seqOArrX"][0]
        o_y = seq_doc["seqOArrY"][0]
    return los,o_x,o_y

@app.callback(
    Output("start-x","value"),
    Output("start-y","value"),
    Output("adjust-defender-button","n_clicks"),
    Output('trajectory-graph', 'clickData'),
    [Input("adjust-defender-button","n_clicks"),
    Input('trajectory-graph', 'clickData'),
    Input("ostart-slider","value"),
    Input("y-slider","value")],
    State("seq-index","value")
)
def display_click_data(n_clicks,clickData,o_start_x,o_start_y,index):
    d_x = no_update
    d_y = no_update
    new_nclicks = no_update
    new_cd = no_update
    o_x = no_update
    o_y = no_update
    if clickData:
        d_x = clickData['points'][0]['x']
        d_y = clickData['points'][0]['y']
        new_nclicks = 0
        new_cd = None
    else:
        if not(index) or index < 0 or index > num_sequences-1:
            index = 0
        seq_doc = get_doc(index)
        o_x = seq_doc["seqOArrX"][0]
        o_y = seq_doc["seqOArrY"][0]
        d_x = seq_doc["seqDArrX"][0]
        d_y = seq_doc["seqDArrY"][0]
        d_x = o_start_x + (d_x-o_x)
        d_y = o_start_y + (d_y-o_y)
    return d_x,d_y,new_nclicks,new_cd






if __name__ == "__main__":
    app.run_server(debug=True)