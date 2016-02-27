import web
import httplib2
import time

urls = (
  '/', 'index'
)

class index:
    def GET(self):
        h = httplib2.Http('.cache')
        f = open('job.json', 'r')
        data = f.read()
	print(data)
        resp, content = h.request("http://localhost:8080/api/v1/namespaces/default/pods", 
                          "POST", 
                          data)
        return 'Submitting:\n' + data + '\n\n' + str(resp) + '\n\n' + str(content)


if __name__ == "__main__": 
    app = web.application(urls, globals())
    app.run()  
