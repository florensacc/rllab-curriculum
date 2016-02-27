# Copyright 2016 The Kubernetes Authors All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM ubuntu:14.04
RUN apt-get update
RUN apt-get install -y python
RUN apt-get install -y python-pip
RUN pip install web.py
RUN pip install httplib2

ADD ./service.py /service.py
ADD ./job.json /job.json

# Local port, not same as host port, which is mapped by docker run -p host:local image
# EXPOSE 8080
# not needed if --net="host"

# Don't touch the port, it is fixed in the AWS kubernetes startup script (annotated @rein)
CMD python service.py 8888
