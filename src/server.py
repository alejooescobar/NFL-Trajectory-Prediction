from dashapp import app
from waitress import serve

serve(app.server,port=80,url_scheme = 'http')
