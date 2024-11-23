import json
import inspect
import json
from typing import Any, Type

import threading
from django.db.models.base import Model
from rest_framework.response import Response

from rest_framework import status


def run_in_background(func):
    """
    Run a function in the background using a thread. This is useful when you want to run a function without blocking the main thread.
    """

    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.start()

    return wrapper


def load_json_file(path: str) -> dict:
    """
    Load JSON file
    """
    with open(path) as f:
        return json.load(f)


def dump_json_file(data: dict, path: str):
    """
    Dump JSON file
    """
    with open(path, "w") as f:
        json.dump(data, f)


def make_success_response(
    data: dict[str, Any] | list[Any] | dict[int, Any] | None = None, message: str = ""
) -> Response:
    """
    Make a success response with 200 status code. The message is optional and will be empty by default. The message is shown on the frontend with a toast based on the status of the response.
    """
    return Response(
        {
            "status": "success",
            "message": message,
            "data": data,
        },
        status=status.HTTP_200_OK,
    )


def make_info_response(
    data: dict[str, Any] | list[Any] | dict[int, Any] | None = None, message: str = ""
) -> Response:
    """
    Make a info response with 308 status code. The message is optional and will be empty by default. The message is shown on the frontend with a toast based on the status of the response.
    """
    return Response(
        {
            "status": "info",
            "message": message,
            "data": data,
        },
        status=status.HTTP_308_PERMANENT_REDIRECT,
    )


def make_warning_response(
    data: dict[str, Any] | list[Any] | dict[int, Any] | None = None, message: str = ""
) -> Response:
    """
    Make a warning response with 307 status code. The message is optional and will be empty by default. The message is shown on the frontend with a toast based on the status of the response.
    """
    return Response(
        {
            "status": "warning",
            "message": message,
            "data": data,
        },
        status=status.HTTP_307_TEMPORARY_REDIRECT,
    )


def make_error_response(
    data: dict[str, Any] | list[Any] | dict[int, Any] | None = None, message: str = ""
) -> Response:
    """
    Make a error response with 400 status code. The message is optional and will be empty by default. The message is shown on the frontend with a toast based on the status of the response.
    """
    return Response(
        {
            "status": "error",
            "message": message,
            "data": data,
        },
        status=status.HTTP_400_BAD_REQUEST,
    )


def make_unauthorized_response(
    data: dict[str, Any] | list[Any] | dict[int, Any] | None = None, message: str = ""
) -> Response:
    """
    Make a unauthorized response with 401 status code. The message is optional and will be empty by default. The message is shown on the frontend with a toast based on the status of the response.
    """
    return Response(
        {
            "status": "unauthorized",
            "message": message,
            "data": data,
        },
        status=status.HTTP_401_UNAUTHORIZED,
    )


def synchronized(func):
    """Decorator to make a function thread-safe."""
    func.__lock__ = threading.Lock()

    def synced_func(*args, **kwargs):
        with func.__lock__:
            return func(*args, **kwargs)

    return synced_func


def color_print(text: Any, color: str = "OKGREEN"):
    """Print text in color"""

    class bcolors:
        HEADER = "\033[95m"
        OKBLUE = "\033[94m"
        OKCYAN = "\033[96m"
        OKGREEN = "\033[92m"
        WARNING = "\033[93m"
        FAIL = "\033[91m"
        ENDC = "\033[0m"
        BOLD = "\033[1m"
        UNDERLINE = "\033[4m"

    if color == "green":
        color = "OKGREEN"
    if color == "red":
        color = "FAIL"
    if color == "blue":
        color = "OKBLUE"
    if color == "yellow":
        color = "WARNING"
    if color == "cyan":
        color = "OKCYAN"
    if color == "purple":
        color = "HEADER"

    print(f"{getattr(bcolors, color.upper())}{text}{bcolors.ENDC}")


def print_test_header(test_name):
    color_print(
        f"""\n
        ----------------------------------------------------------------------------
                                    test: {test_name}
        ----------------------------------------------------------------------------
            """,
        "OKCYAN",
    )
    color_print("## =>  started", "WARNING")


def print_test_passed():
    color_print("## =>  passed", "OKGREEN")


def print_test_failed():
    color_print("## =>  failed", "FAIL")


def get_model_field_names_list(model: Type[Model]) -> list[str]:

    return [field.name for field in model._meta.get_fields() if field.concrete]  # type: ignore


def get_list(data: int | list[int] | str | list[str] | None) -> list:
    if not data:
        return []

    if isinstance(data, str) or isinstance(data, int):
        return [data]

    return data


def make_object_hashmap(data: list[dict], key1: str, key2: str | None = None) -> dict:
    hashmap = {}
    for item in data:
        if item[key1] not in hashmap:
            hashmap[item[key1]] = dict(item) if not key2 else {}

        if key2 and item[key2] not in hashmap[item[key1]]:
            hashmap[item[key1]][item[key2]] = dict(item)

    return hashmap


def make_list_hashmap(data: list[dict], key1: str, key2: str | None = None) -> dict:
    hashmap = {}
    for item in data:
        if item[key1] not in hashmap:
            hashmap[item[key1]] = []
        if not key2:
            hashmap[item[key1]].append(dict(item))
        else:
            if item[key2] not in hashmap[item[key1]]:
                hashmap[item[key1]].append(item[key2])

    return hashmap


def make_key_hashmap(
    data: list[dict], key1: str, key2: str, key3: str | None = None
) -> dict:
    hashmap = {}

    for item in data:
        if item[key1] not in hashmap:
            hashmap[item[key1]] = {}

        if item[key2] not in hashmap[item[key1]]:
            if not key3:
                hashmap[item[key1]] = item[key2]
            else:
                hashmap[item[key1]][item[key2]] = {}

        if key3 and item[key3] not in hashmap[item[key1]][item[key2]]:
            hashmap[item[key1]][item[key2]] = item[key3]

    return hashmap


def jsonify(data: dict[str, Any] | list[Any] | dict[int, Any]) -> str:
    return json.dumps(data, indent=4, default=str)


def debug_print(
    data: dict[str, Any] | list[Any] | dict[int, Any], color: str = "green"
) -> None:
    frame = inspect.currentframe()
    try:
        var_name = [var_name for var_name, var_val in frame.f_back.f_locals.items() if var_val is data][0]  # type: ignore
    except Exception:
        var_name = "data"

    if color == "green":
        color_print(f"{var_name} = {jsonify(data)}", "OKGREEN")
    if color == "red":
        color_print(f"{var_name} = {jsonify(data)}", "FAIL")
    if color == "blue":
        color_print(f"{var_name} = {jsonify(data)}", "OKBLUE")
    if color == "yellow":
        color_print(f"{var_name} = {jsonify(data)}", "WARNING")
    if color == "cyan":
        color_print(f"{var_name} = {jsonify(data)}", "OKCYAN")
    if color == "purple":
        color_print(f"{var_name} = {jsonify(data)}", "HEADER")
