# Design Prompt
I would like to write a project called LLM server that serves local LLMs over a port for use in other apps and projects on my computer. It will implement a basic API described below, including response streaming and model loading and unloading. I want the simplest, most concise implementation you can manage. The less code the better, large amounts of code are hard to read, understand, and maintain.

The project will use the following structure (in addition to the git files, as this is a git repo):
```md
llm-server/
- models/
- server.py
- server.log
- llm.sh
- README.md
- plan.md (this file, will not be in the final project)
```

## models/
A sub-directory in the project that holds available model weight files.

## server.py
The main source code for the project, which does the following:
- Starts the server to listen on a port specified at the top of the file.
- Allows models to be loaded into ram, after which they are automatically unloaded when their ttl expires.
  - If multiple requests are made to the same model, the ttl should be updated each time to the highest ttl received. e.g. A load request loads a model with a ttl of 30. 20 seconds later (the ttl is now 10), a run request with a ttl of 15 arrives. After the request, the ttl is set to 15 because the request had a larger ttl than the remaining value.
- Allows streaming inference from loaded models, which are managed using a python library.
- Implements an API that responds to requests using the following types and parameters:
  - **list**: Lists all available LLM models, both loaded and not loaded.
    - **loaded** (optional): specify true (list only loaded models), false (list only unloaded models), or any (default, list all models).
  - **load**: Loads a model into ram so that it is ready for use.
    - **model** (required): specifies the model name, in the same format given by the list request.
    - **ttl** (optional): The amount of time, in seconds, that the model should remain loaded after the request finishes. The default should be specified in a variable at the top of this file. Set it to 300s for now.
  - **run**: Runs a model with the given prompt and streams the response back over the API to the requesting client.
    - **prompt** (required): specifies the text prompt that the model will respond to.
    - **model** (required): Same as load request.
    - **ttl** (optional): Same as load request.
    - **autoload** (optional): true if an unloaded model should be loaded for the request. If false (the default), the request returns an error if the model is not loaded.

## server.log
A plain text debug and analytics log file that records important events, including the following events listed below. All logs should include a date-time stamp.
- Server start
- Server stop
- Request received
- Request completed
- Model loaded
- Model unloaded
- Any other important events

## llm.sh
A simple script that allows requests to be sent programmaticly from a command line, primarily intended for testing purposes. This should allow all the requests, as well as their parameters, to be specified and should handle receiving and displaying the responses (e.g. printing the streamed tokens as they arrive from a run request, listing models, etc.). It should use the API and send requests over the port. It should not call the server directly.

## readme.md
A simple, concise doc file explaining how to use this project and especially how to interact with the API. This is intended to be read by coding agents, who are attempting to build with this project, so it should be as short as possible. Provide the necessary context and trim any filler, make every word count for this file.
