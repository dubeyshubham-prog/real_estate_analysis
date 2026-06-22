from fastapi.templating import Jinja2Templates

from config.settings import Config


templates = Jinja2Templates(directory=str(Config.TEMPLATES_DIR))
