from dashapp import app
from waitress import serve

serve(app.server,port=8000,url_scheme = 'https')
