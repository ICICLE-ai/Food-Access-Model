from fastapi import Request, FastAPI
from pyinstrument import Profiler
from pyinstrument.renderers.html import HTMLRenderer
from pyinstrument.renderers.speedscope import SpeedscopeRenderer
from typing import Callable
import os
import logging

class FassProfiler(Profiler):
    profile_type_to_ext = {"html": "html", "speedscope": "speedscope.json"}
    profile_type_to_renderer = {
        "html": HTMLRenderer,
        "speedscope": SpeedscopeRenderer,
    }
    
    
    
    def __init__(self, interval = 0.001, async_mode = "enabled", use_timing_thread = None, filename=None):
        super().__init__(interval, async_mode, use_timing_thread)
        self.filename = filename
        
    def output(self, renderer = SpeedscopeRenderer()):
        profile_output =  super().output(renderer)
        
        if self.filename is not None:
            with open(self.filename, "w") as out:
                out.write(profile_output)

        return profile_output


def register_middlewares(app: FastAPI):

    @app.middleware("http")
    async def profile_request(request: Request, call_next: Callable):
        """Profile the current request

        Taken from https://pyinstrument.readthedocs.io/en/latest/guide.html#profile-a-web-request-in-fastapi
        with small improvements.

        """
        request_path = request.url.path
        # we map a profile type to a file extension, as well as a pyinstrument profile renderer
        profile_type_to_ext = {"html": "html", "speedscope": "speedscope.json"}
        profile_type_to_renderer = {
            "html": HTMLRenderer,
            "speedscope": SpeedscopeRenderer,
        }

        # if the `profile=true` HTTP query argument is passed, we profile the request
        if request.query_params.get("profile", True):

            print("PROFILING REQUEST", request_path)
            # The default profile format is speedscope
            profile_type = request.query_params.get("profile_format", "speedscope")

            # we profile the request along with all additional middlewares, by interrupting
            # the program every 1ms1 and records the entire stack at that point
            
            # we dump the profiling into a file
            extension = profile_type_to_ext[profile_type]
            renderer = profile_type_to_renderer[profile_type]()
            
            
            # Ensure the profiling directory exists

            os.makedirs("profiling/api", exist_ok=True)
            # Create a unique filename if the file already exists
            base_filename = f"profiling/{request_path.lstrip('/')}.{extension}"

            filename = base_filename
            counter = 1
            while os.path.exists(filename):
                filename = f"{base_filename.rsplit('.', 1)[0]}_{counter}.{extension}"
                counter += 1
            
            with FassProfiler(interval=0.001, async_mode="enabled", filename=filename) as profiler:
                response = await call_next(request)


            #print output
            print("### PROFILING INFO")
            print(profiler.output(renderer=renderer))
            print("### PROFILING INFO")

            # with open(filename, "w") as out:
            #     out.write(profiler.output(renderer=renderer))
                
            return response

        # Proceed without profiling
        return await call_next(request)