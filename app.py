from flask import Flask
from interface import create_interface

app = Flask(__name__)

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_port=7860)
