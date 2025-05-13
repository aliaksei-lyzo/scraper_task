1. Raw estimation - at a glance doesn't seem too difficult. I think 2-3 days will be enough to complete. Risk factors - just a basic familiarity with python and AI solutions frameworks. Huge advantage - AI assistants.

2. Approaching the solution:
I intend to utilize GitHub Copilot agentic capabilities (GPT-4.1) to assist in the development process.
For that i will create:
- a copilot-instructions.md file to provide the agent with the necessary context and instructions.
- a PLANNING.md file to outline the project structure, goals, and constraints.
- a TASK.md file to keep track of tasks and progress.
- a REPORT.md file to document the process and findings during the completion of this project

Then I will ask Copilot to help picking the best libraries and tools for the task, create a list of tasks, and start implementing the solution step by step. I will also ask it to help with writing tests for each component as I go along.

So, here we go.

00:00 Started the project. Copilot advised to use Streamlit for the UI and ChromaBD as Vector.

00:05 Copilot generated task list and project structure. I will start with the first task - setting up the Python virtual environment and installing the required libraries. He estimated total time of 4h 35min for the project. I think it is a bit optimistic, but let's see how it goes.

00:20 CP struggled to install dependencies using poetry - faced issues with permissons, which it couldn't resolve. I switched to Admin mode in VS Code, then all

00:45 GPT-4.1 messed up with the set up of venv and poetry and couldn't resolve it. Claude Sonnet 3.7 solved the issue in 1 minute. I will use Claude for the rest of the project for now.

01:00 Claude 3.7 is amazing - quickly resolved the IDE issue of connecting to the correct python interpreter.

01:38 I have carefully studied and learnt the code for extration of text from news urls. I have fixed a couple of issues myself - a note not to ask for these fixes in chat-agentic mode, as it will mess the whole file (or at lest save and commit beforehand). Inline chat works better here.

02:38 I have been struggling with Copilot for a while now. First it keeps entering commands for bash, when I'm using PowerShell. Then it keeps installing packages by modifying pyproject.toml and lockfile, or tries to use pip. Tried fixing it by modifying the copilod-instructions.md file, we'll see how it goes. Also a huge issue is that the Copilot Chat UI is crazy - when I try to end the session, I save all files but when I close the chat (although i click keep everywhere possible) it still reverts all changes! I lost all changes 2 or 3 times already. Trying to figure out how to fix it, in the meantime the possible way is to commit the changes and then cancel out reverting.

03:00 Now we're talking. Fed the agent with docs for the Langchain 0.3.0 and with minor fixes it generated a working code for news summarization. Unit tests are also working, however I didn't check the quality of the tests. Summaries are looking good.

3:14 Pretty quickly extracted the models to a separate file, fixed the tests. Noice!

4:14 Spent some time debugging and setting up the database connection. Lost all changes a couple of times, but found a workaround - starting a new session and then deleting the old one works.

4:56 Finally got the database working. Had to fix a couple of typing issues (seems like copilot doesn't connect to Pylance automatically) and there was an issue with error handling in db setup. Also faced problems with venv in different terminals - but it's just me learning python.

5:12 Quickly done the semantic search task, but not tested yet - wanna test with UI altogether.

5:32 Finalized the UI, had to do a bit of fixes, but overall it was pretty straightforward. I think the UI is looking good, searches are working.