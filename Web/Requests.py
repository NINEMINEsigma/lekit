import requests as base
import urllib3

class light_requests(object):
    def __init__(self, retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 504)):
        self.session = base.Session()
        retry_strategy = urllib3.Retry(
            total=retries,
            status_forcelist=status_forcelist,
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=backoff_factor,
        )
        adapter = base.adapters.HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def get(self, url, timeout=30, headers=None, **kwargs):
        return self.session.get(url, timeout=timeout, headers=headers, **kwargs)
    
    def post_data(self, url, data=None, timeout=30, **kwargs):
        return self.session.post(url, data=data, timeout=timeout, **kwargs)
    
    def post_json(self, url, json=None, timeout=30, **kwargs):
        return self.session.post(url, json=json, timeout=timeout, **kwargs)
    
    def put(self, url, data=None, timeout=30, **kwargs):
        return self.session.put(url, data=data, timeout=timeout, **kwargs)
    
    def delete(self, url, timeout=30, **kwargs):
        return self.session.delete(url, timeout=timeout, **kwargs)
    

if __name__ == "__main__":
    # Example usage
    light = light_requests()
    url = 'http://localhost:8080'
    
    response = light.get(url)
    response = light.put(url)
    response = light.post_json(url,{'one':'two'})
    response = light.put(url)
    response = light.post_data(url,{'key':'value'})
    response = light.get(url)
    response = light.put(url)