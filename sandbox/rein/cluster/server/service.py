import web
import httplib2
import time
import json

"""
RUN THIS BY
sudo docker pull rein/rllab-q && sudo docker run -t --net="host" rein/rllab-q
on master kubernetes server
"""

urls = (
    '/', 'index'
)


def make_pod_json(container_name, expt_name, index, command):
    podparts = []
    podparts.extend([expt_name, "%.4i" % index])
    podname = "-".join(podparts)

    command = ["/bin/bash", "-c", command]
    pod_json = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": podname,
            "labels": {
                "expt": expt_name
            },
        },
        "spec": {
            "containers": [
                {
                    "name": "foo",
                    "image": container_name,
                    "command": command,
                    "resources": {"requests": {"cpu": "1"}},
                    "imagePullPolicy": "Always",
                }
            ],
            "restartPolicy": "Never",
        }
    }
    return pod_json


class index:
    counter = 0

    def POST(self):
        # Retrieve POST data.
        data = web.input(_method='post')
        self.counter += 1
        container_name = data['container_name']
        expt_name = data['expt_name']
        command = data['command']
        print(container_name)
        print(expt_name)
        print(command)
        pod_json = make_pod_json(container_name, expt_name,
                                 self.counter, command)
        pod_json_str = json.dumps(pod_json, indent=1)
        print(pod_json_str)
        h = httplib2.Http('.cache')
        resp, content = h.request("http://localhost:8080/api/v1/namespaces/default/pods",
                                  "POST",
                                  pod_json_str)
        return 'Submitting:\n' + pod_json_str + '\n\n' + str(resp) + '\n\n' + str(content)

#     def GET(self):
#         h = httplib2.Http('.cache')
#         f = open('job.json', 'r')
#         data = f.read()
#         print(data)
#         resp, content = h.request("http://localhost:8080/api/v1/namespaces/default/pods",
#                                   "POST",
#                                   data)
# return 'Submitting:\n' + data + '\n\n' + str(resp) + '\n\n' +
# str(content)


if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()
