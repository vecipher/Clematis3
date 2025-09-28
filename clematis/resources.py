# clematis/resources.py
from importlib.resources import files


def fixtures_dir():
    # .../site-packages/clematis/fixtures
    return files(__package__).joinpath("fixtures")


def llm_fixtures_dir():
    return fixtures_dir().joinpath("llm")
