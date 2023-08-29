from dashapp import app
from waitress import serve
# server for dash app
serve(app.server,port=8080,url_scheme = 'http')
