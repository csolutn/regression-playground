import os

from dotenv import load_dotenv
from flask import Flask, render_template

load_dotenv()


def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev")

    @app.route("/")
    def index():
        return render_template("index.html")

    return app
