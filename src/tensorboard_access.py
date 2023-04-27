import json
import multiprocessing
import os
import subprocess
import time


def is_process_running(process_cmd):
    try:
        process_output = subprocess.check_output(["pgrep", "-f", process_cmd])
        return bool(process_output.strip())
    except subprocess.CalledProcessError:
        return False


def start_tensorboard():
    pool = multiprocessing.Pool(processes=10)
    cmds = []

    if not is_process_running("tensorboard"):
        cmds.append("tensorboard --logdir ./runs/ --host 0.0.0.0 --port 6006 &")

    if not is_process_running("ngrok"):
        cmds.append("ngrok http 6006 &")

    if cmds:
        [pool.apply_async(os.system, args=(cmd,), callback=None) for cmd in cmds]


def get_public_url(retries=10, delay=1):
    if not is_process_running("tensorboard") or not is_process_running("ngrok"):
        start_tensorboard()

    for _ in range(retries):
        try:
            # Execute the curl command and capture the output
            curl_output = subprocess.check_output(["curl", "-s", "http://localhost:4040/api/tunnels"])

            # Load the JSON data from the curl output
            tunnels_data = json.loads(curl_output)

            # Get the public URL from the JSON data
            public_url = tunnels_data['tunnels'][0]['public_url']
            return public_url
        except (subprocess.CalledProcessError, IndexError):
            time.sleep(delay)
    raise RuntimeError("Failed to get the public URL after {} attempts".format(retries))
