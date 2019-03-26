import collections
import random
import string
import uuid

from flask import Flask, json, request
from flask_cors import CORS


APP = Flask(
    __name__,
    static_url_path='',
    static_folder='static'
)

CORS_OBJ = CORS(APP, resources={r"/api/*": {"origins": "*"}})

DOOR_ID = "497079449313791"

VISITOR = collections.namedtuple("Visitor", ["name", "unlock_code"])

DATA = {
    "5823647720": VISITOR("Alice", "7c91befe-4ac4-49ae-a867-228383ee8fd8"),
    "5790323022": VISITOR("Bob", "54a1701d-81d2-4382-81be-1a39bb5a4569"),
    "4286357830": VISITOR("Catrina", "45dbe456-bb65-4cc2-9de0-421cb2bc92e8"),
    "8639881096": VISITOR("Destiny", "82732f4e-27e3-493b-8a1a-437025eb9f4f"),
    "3899201564": VISITOR("Edgar H.", "3db3807c-6a65-4d4d-ac46-002be30185b6"),
    "1909294305": VISITOR("Fredd", "71084cf9-4e42-4055-b952-f9017efac359"),
}

HELP = """<h1>Test API Help</h1>
<p>Below is a description of the provided test APIs/</p>
<dl>
  <dt>/api/visitors</dt>
  <dd>A list of currently open visitor keys</dd>
  <dt>/api/visitors/{<em>visitor_key</em>}</dt>
  <dd>The lock authentication key for visitor with key <em>visitor_key</em></dd>
</dl>
"""

def generate_digit_string(size):
    if size <= 0:
        return ''

    return ''.join((random.choice(string.digits) for _ in range(size)))


@APP.route("/api/")
def api_index():
    """ Return a page defining available API calls. """
    return HELP


@APP.route("/api/visitors/")
def api_visitors():
    """ API call for homeowners to get a list of visitors. """
    return json.jsonify(
        status="success",
        data={
            "door_id": DOOR_ID,
            "visitor_keys": {
                key: value.name
                for key, value in DATA.items()
            },
        },
        message=None
    )


@APP.route("/api/visitors/new/", methods=["POST"])
def api_visitors_post():
    post_data = json.loads(request.data.decode("utf-8"))

    try:
        name = post_data["name"]
        if not name:
            raise AttributeError
    except (AttributeError, KeyError):
        return json.jsonify(
            status="failure",
            data=None,
            message="No name was provided"
        )

    # Set up the new visitor inside the "database"
    visitor_key = generate_digit_string(10)

    DATA[visitor_key] = VISITOR(name, uuid.uuid4())

    return json.jsonify(
        status="success",
        data={
            "door_id": DOOR_ID,
            "visitor": {
                "name": name,
                "key": visitor_key,
            }
        },
        message=None
    )

@APP.route("/api/visitors/del/", methods=["DELETE"])
def api_visitors_delete():
    request_data = json.loads(request.data.decode("utf-8"))

    try:
        visitor_id = request_data["visitor_id"]
    except KeyError:
        return json.jsonify(
            status="failure",
            data=None,
            message="Invalid payload was provided"
        )

    try:
        del DATA[visitor_id]
    except KeyError:
        return json.jsonify(
            status="failure",
            data=None,
            message="Invalid visitor key was provided"
        )

    return json.jsonify(
        status="success",
        data=None,
        message="Successfully removed visitor {}".format(visitor_id)
    )



@APP.route("/api/visitors/<string:door_id>/<string:visitor_id>/")
def api_visitors_get(door_id, visitor_id):
    """ API call for visitor to get an unlock key. """
    try:
        return json.jsonify(
            status="success",
            data={
                "unlock_key": DATA[visitor_id].unlock_code,
            },
            message=None
        )
    except KeyError:
        return json.jsonify(
            status="failure",
            data=None,
            message="Invalid visitor key was requested!"
        )


if __name__ == "__main__":
    APP.run(port=8080, debug=True)
