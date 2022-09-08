"""Predicatorobot slack bot code.

To launch on supercloud (e.g., after a downtime has ended):

`rm -f nohup.out && nohup python scripts/launch_slack_bot.py &`
"""

import abc
import os
import re
import socket
import subprocess
from typing import Callable, Dict, List, Optional, Type
from urllib.request import urlopen

import requests
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

REPO_NAME = "Learning-and-Intelligent-Systems/predicators"
ANALYSIS_CMD = "python scripts/analyze_results_directory.py"
LAUNCH_CMD = ("python scripts/supercloud/launch.py --config nightly.yaml "
              "--user $USER --on_supercloud")
MAX_CHARS_PER_MESSAGE = 3500  # actual limit is 4000, but we keep a buffer
GITHUB_SEARCH_RESPONSE_MAX_FILE_MATCHES = 3
SUPERCLOUD_LOGIN_SERVER = "login-2"  # can also use login-3 or login-4
CONDA = "/state/partition1/llgrid/pkg/anaconda/anaconda3-2021b/bin/activate"

SUPERCLOUD_USER_TO_DIR = {
    "ronuchit": "/home/gridsan/ronuchit/predicators/",
    "tslvr": "/home/gridsan/tslvr/predicators/",
    "njk": "/home/gridsan/njk/GitHub/research/predicators/",
    "wmcclinton": "/home/gridsan/wmcclinton/GitHub/predicators/",
    "wshen": "/home/gridsan/wshen/predicators/",
}

# Load all environment variables. We do this early on to make sure they exist.
PREDICATORS_SLACK_BOT_TOKEN = os.environ["PREDICATORS_SLACK_BOT_TOKEN"]
PREDICATORS_SLACK_BOT_SIGNING_SECRET = os.environ[
    "PREDICATORS_SLACK_BOT_SIGNING_SECRET"]
PREDICATORS_SLACK_APP_TOKEN = os.environ["PREDICATORS_SLACK_APP_TOKEN"]
GITHUB_TOKEN = os.environ["GITHUB_ACCESS_TOKEN"]

# Initialize main App object.
app = App(token=PREDICATORS_SLACK_BOT_TOKEN,
          signing_secret=PREDICATORS_SLACK_BOT_SIGNING_SECRET)


class Response:
    """A response subclass defines a method for generating a message and a
    method for optionally generating the name of a file to upload."""

    def __init__(self, query: str, inquirer: str) -> None:
        self._query = query
        self._inquirer = inquirer

    @abc.abstractmethod
    def get_message_chunks(self) -> List[str]:
        """Return a list of messages.

        No message should be longer than MAX_CHARS_PER_MESSAGE. The
        self._chunk_message() helper function may be helpful here.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def get_filename(self) -> Optional[str]:
        """Optionally return the name of a file to upload."""
        raise NotImplementedError("Override me!")

    @staticmethod
    def _chunk_message(message: str, prefix: str,
                       include_code_blocks: bool) -> List[str]:
        """Slack enforces a per-message character limit when using the API, so
        we need to split up our message into chunks.

        We do this by line to preserve the quality of the output.
        """
        chunks = [prefix]
        if include_code_blocks:
            chunks[-1] += "```"
        for line in message.split("\n"):
            if len(chunks[-1]) + len(line) > MAX_CHARS_PER_MESSAGE:
                if include_code_blocks:
                    chunks[-1] += "```"
                chunks.append("")
                if include_code_blocks:
                    chunks[-1] += "```"
            chunks[-1] += line + "\n"
        if include_code_blocks:
            chunks[-1] += "```"
        return chunks


class DefaultResponse(Response):
    """A default response, for when the query wasn't understood."""

    def get_message_chunks(self) -> List[str]:
        ret = (f"Sorry <@{self._inquirer}>, I'm pretty dumb. I couldn't "
               f"understand your query ({self._query}). Right now, here's "
               "what I can understand:\n"
               "- `analyze <supercloud username>` or "
               "`analysis <supercloud username>` to run analysis script\n"
               "- `progress <supercloud username>` to get current progress\n"
               "- `launch <supercloud username>` to launch experiments\n"
               "- `cs <any string to search on our github repo>`, e.g., "
               "`cs def flush_cache`\n"
               "- `tom`\n")
        return [ret]

    def get_filename(self) -> Optional[str]:
        return None


class TomEmojiResponse(Response):
    """A Tom emoji response!"""

    def get_message_chunks(self) -> List[str]:
        return [":tom:"]

    def get_filename(self) -> Optional[str]:
        return None


class GithubSearchResponse(Response):
    """A response that looks for the queried search string on Github."""

    def __init__(self, query: str, inquirer: str, search_string: str) -> None:
        super().__init__(query, inquirer)
        self._search_string = search_string

    def get_message_chunks(self) -> List[str]:
        query = (
            f'https://api.github.com/search/code?q="{self._search_string}"'
            f'+in:file+extension:py+repo:{REPO_NAME}')
        headers = {"Authorization": f"token {GITHUB_TOKEN}"}
        response = requests.request("GET", query, headers=headers,
                                    timeout=10).json()
        chunks = [f'Github matches for string "{self._search_string}":']
        num_matches = len(response["items"])
        if num_matches > GITHUB_SEARCH_RESPONSE_MAX_FILE_MATCHES:
            chunks.append(
                f"There are {num_matches} files matching this search "
                "string. Can you be more specific?")
            return chunks
        # Github's code search API doesn't return which lines match. So we
        # have to search through the files ourselves to find matches.
        for item in response["items"]:
            query = item["url"]
            response = requests.request("GET",
                                        query,
                                        headers=headers,
                                        timeout=10).json()
            output = urlopen(response["download_url"]).read()
            lines = output.decode("utf-8").split("\n")
            html_url = response["html_url"]
            for i, line in enumerate(lines):
                if self._search_string in line:
                    chunks.append(f"{html_url}#L{i+1}")
        chunks.append(f"<@{self._inquirer}> I'm done listing matches.")
        return chunks

    def get_filename(self) -> Optional[str]:
        return None


class SupercloudResponse(Response):
    """An abstract response for supercloud that handles SSH and SCP stuff."""

    def __init__(self, query: str, inquirer: str, user: str) -> None:
        super().__init__(query, inquirer)
        self._user = user
        if "gridsan" not in os.getcwd():
            self._not_on_supercloud = True
        elif user not in SUPERCLOUD_USER_TO_DIR:
            self._invalid_user = True
            self._not_on_supercloud = False
        else:
            self._invalid_user = False
            self._not_on_supercloud = False

    def get_message_chunks(self) -> List[str]:
        if self._not_on_supercloud:
            return [
                f"Sorry <@{self._inquirer}>, I'm not running on "
                f"supercloud!"
            ]
        if self._invalid_user:
            return [
                f"Sorry <@{self._inquirer}>, {self._user} is not "
                "registered. You may need to update SUPERCLOUD_USER_TO_DIR."
            ]
        # There are a bunch of steps that need to be done correctly in order
        # for the SSHing to work out the way we want. The ultimate goal is
        # to launch each command in self._get_commands() sequentially, and
        # store all the stdout/stderr into output.txt.
        user = self._user
        dir_name = SUPERCLOUD_USER_TO_DIR[user]
        assert dir_name.endswith("predicators/")
        dir_name_up = dir_name[:-12]
        # 1) Always start by clearing the current output.txt.
        commands = ["rm -f output.txt"]
        for command in self._get_commands():
            # 2) SSH into the user's login server.
            commands.append(
                f"ssh {user}@{SUPERCLOUD_LOGIN_SERVER} "
                # 3) Change into the desired directory and source
                # the user's conda environment.
                f"'cd {dir_name} && source {CONDA} predicators "
                # 4) This step is tricky! For some reason, when you
                # SSH, the PYTHONPATH isn't preserved, so we update
                # it here to be the directory one level up from the
                # predicators repository, so that the imports in our
                # repository work as expected (from predicators...).
                f"&& export PYTHONPATH=$PYTHONPATH:{dir_name_up} "
                # 5) Similarly, update the PYTHONHASHSEED.
                "&& export PYTHONHASHSEED=0 "
                # 6) Run the actual command, appending both stdout
                # and stderr onto the end of output.txt.
                f"&& {command}' >> output.txt 2>&1")
        cmd_str = " && ".join(commands)
        print(f"Running SSH command: {cmd_str}", flush=True)
        subprocess.getoutput(cmd_str)
        # This should be impossible unless self._get_commands() was empty.
        assert os.path.exists("output.txt")
        return self._supercloud_get_message_chunks()

    def get_filename(self) -> Optional[str]:
        if self._not_on_supercloud or self._invalid_user:
            return None
        return self._supercloud_get_filename()

    def _scp_filename(self, path: str) -> None:
        """Helper method that SCPs over a file with the given path.

        The given path is relative to the directory stored in
        SUPERCLOUD_USER_TO_DIR.
        """
        user = self._user
        dir_name = SUPERCLOUD_USER_TO_DIR[user]
        cmd_str = f"scp {user}@{SUPERCLOUD_LOGIN_SERVER}:{dir_name}/{path} ."
        print(f"Running SCP command: {cmd_str}", flush=True)
        subprocess.getoutput(cmd_str)

    @abc.abstractmethod
    def _get_commands(self) -> List[str]:
        """Return a list of commands to run from within the directory stored in
        SUPERCLOUD_USER_TO_DIR.

        Each command's output will be appended to the end of output.txt.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _supercloud_get_message_chunks(self) -> List[str]:
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _supercloud_get_filename(self) -> Optional[str]:
        raise NotImplementedError("Override me!")


class SupercloudLaunchResponse(SupercloudResponse):
    """A response that wipes saved data on supercloud and launches
    experiments."""

    def _get_commands(self) -> List[str]:
        return [("git stash && git checkout master && git pull && "
                 "rm -f results/* logs/* saved_approaches/* "
                 f"saved_datasets/* && {LAUNCH_CMD}")]

    def _supercloud_get_message_chunks(self) -> List[str]:
        num_launched = 0
        with open("output.txt", "r", encoding="utf-8") as f:
            for line in f.readlines():
                if line.startswith("Running command:"):
                    num_launched += 1
        return [
            f"<@{self._inquirer}>: Launched {num_launched} job arrays on "
            f"{self._user}'s supercloud."
        ]

    def _supercloud_get_filename(self) -> Optional[str]:
        return None


class SupercloudProgressResponse(SupercloudResponse):
    """A response that gets the number of jobs running on supercloud, and the
    number of results generated."""

    def _get_commands(self) -> List[str]:
        return ["squeue | wc -l", "ls results/ | wc -l"]

    def _supercloud_get_message_chunks(self) -> List[str]:
        lines = []
        with open("output.txt", "r", encoding="utf-8") as f:
            for line in f.readlines():
                try:
                    lines.append(int(line.strip()))
                except ValueError:
                    pass
        if len(lines) != 2:
            return [
                f"Sorry <@{self._inquirer}>, malformed output.txt. "
                f"Check that the SSH command printed out on the server "
                f"is working as expected."
            ]
        num_jobs = lines[0] - 1
        num_res = lines[1]
        return [
            f"<@{self._inquirer}>: On {self._user}'s supercloud, "
            f"{num_jobs} jobs are currently running, and there "
            f"are {num_res} files in results/."
        ]

    def _supercloud_get_filename(self) -> Optional[str]:
        return None


class SupercloudAnalysisResponse(SupercloudResponse):
    """A response that runs analysis on supercloud."""

    def __init__(self, query: str, inquirer: str, user: str) -> None:
        super().__init__(query, inquirer, user)
        self._generated_csv = False

    def _get_commands(self) -> List[str]:
        return [ANALYSIS_CMD]

    def _supercloud_get_message_chunks(self) -> List[str]:
        with open("output.txt", "r", encoding="utf-8") as f:
            output = f.read()
        if "No data found" in output:
            return [
                f"<@{self._inquirer}>: No data in results/ on "
                f"{self._user}'s supercloud."
            ]
        if "AGGREGATED DATA OVER SEEDS:" not in output:
            return [
                f"Sorry <@{self._inquirer}>, malformed output.txt. "
                f"Check that the SSH command printed out on the server "
                f"is working as expected."
            ]
        self._generated_csv = True
        message = output[output.index("Git commit hashes"):].split(
            "Wrote out")[0].strip("\n")
        chunks = self._chunk_message(
            message,
            prefix=(f"<@{self._inquirer}>: Here are results on "
                    f"{self._user}'s supercloud.\n"),
            include_code_blocks=True)
        return chunks

    def _supercloud_get_filename(self) -> Optional[str]:
        if not self._generated_csv:
            return None
        self._scp_filename("results_summary.csv")
        return "results_summary.csv"


def _get_response_object(query: str, inquirer: str) -> Response:
    """Parse the query to find the appropriate Response class.

    Return an instantiation of that class.
    """
    match = re.match(r"(analysis|analyze|progress|launch) (\w+)", query)
    if match is not None:
        if match.groups()[0] in ("analysis", "analyze"):
            cls: Type[SupercloudResponse] = SupercloudAnalysisResponse
        elif match.groups()[0] == "launch":
            cls = SupercloudLaunchResponse
        else:
            assert match.groups()[0] == "progress"
            cls = SupercloudProgressResponse
        user = match.groups()[1]
        return cls(query, inquirer, user)
    match = re.match(r"cs (.+)", query)
    if match is not None:
        search_string = match.groups()[0]
        return GithubSearchResponse(query, inquirer, search_string)
    match = re.match(r"tom", query)
    if match is not None:
        return TomEmojiResponse(query, inquirer)
    return DefaultResponse(query, inquirer)


@app.event("app_mention")
def _callback(ack: Callable[[], None], body: Dict) -> None:
    """This callback is triggered whenever someone @mentions this bot."""
    ack()  # all callback functions must run this
    event = body["event"]
    query = event["text"]
    inquirer = event["user"]
    channel_id = event["channel"]
    home_dir = os.path.expanduser("~")
    host_name = socket.gethostname()
    bot_user_id = app.client.auth_test().data["user_id"]  # type: ignore
    assert f"<@{bot_user_id}" in query
    query = query.replace(f"<@{bot_user_id}>", "").strip()
    print(f"Got query from user {inquirer}: {query}", flush=True)
    pid = os.getpid()
    # Post an initial response, so the inquirer knows this bot is alive.
    app.client.chat_postMessage(
        channel=channel_id,
        text=(f"Hello from {host_name}:{home_dir} (PID: {pid})! Generating a "
              f"response to your query, <@{inquirer}>."))
    # Generate response object from query.
    response = _get_response_object(query, inquirer)
    # Post messages and upload file (if provided).
    chunks = response.get_message_chunks()
    assert isinstance(chunks, list)
    for message in chunks:
        assert isinstance(message, str)
        app.client.chat_postMessage(channel=channel_id, text=message)
    fname = response.get_filename()
    if fname is not None:
        app.client.files_upload(channels=channel_id, file=fname, title=fname)
    print("Finished handling query", flush=True)


if __name__ == "__main__":
    SocketModeHandler(app, PREDICATORS_SLACK_APP_TOKEN).start()  # type: ignore
