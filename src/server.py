from dashapp import app
from waitress import serve

serve(app.server,port=8080,url_scheme = 'http')
